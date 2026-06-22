"""Unit tests for enrichment-v2 (schema / prompts / validator / extractor / behavioral).

No network: the OpenAI client is monkeypatched with a canned structured response, so
the full extraction path (parse → validate → cache round-trip → variant propagation →
parquet) is exercised offline. Also asserts the anti-recoding guarantee (no metadata in
the prompt) and the gap rank maps.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import EnrichmentV2Config
from src.knowledge.enrichment_v2 import (
    E2_LLM_AXES,
    build_e2_json_schema,
    build_e2_messages,
    estimate_e2_cost,
    extract_e2_pilot,
    render_e2_row,
    validate_e2,
)
from src.knowledge.enrichment_v2.prompts import strip_fabric_words, strip_metadata
from src.knowledge.enrichment_v2.schema import (
    PRICE_TIER_RANK,
    TREND_LOOK_RANK,
    TREND_PHASE_RANK,
)

CANNED = {
    "e2_occasion_primary": "Date-romantic",
    "e2_occasion_secondary": ["Party-festive", "Going-out-evening"],
    "e2_occasion_formality": 3,
    "e2_fit_intent": "Body-skimming",
    "e2_body_ease": 2,
    "e2_care_burden": 4,
    "e2_care_flags": ["Delicate-trims", "Wrinkle-prone"],
    "e2_price_look": 4,
    "e2_trend_look": "Current",
}


# --------------------------------------------------------------------------- schema
def test_schema_is_strict_and_complete():
    sch = build_e2_json_schema()
    assert sch["additionalProperties"] is False
    assert set(sch["required"]) == set(E2_LLM_AXES) == set(sch["properties"])


def test_estimate_cost_under_budget():
    assert estimate_e2_cost(500) < 0.5  # pilot budget guard


# --------------------------------------------------------------------------- prompts
def test_no_metadata_leak_in_messages():
    rep = {
        "article_id": "0123",
        "product_code": "012",
        "prod_name": "Slim cotton tee",
        "detail_desc": "Jersey top in soft cotton, 95% cotton 5% elastane. Machine wash. V-neck.",
    }
    blob = json.dumps(build_e2_messages(rep, image_base64="ZmFrZQ==")).lower()
    for leak in ("product_type_name", "colour_group_name", "garment_group_name", "section_name"):
        assert leak not in blob
    for fabric in ("cotton", "elastane", "machine wash"):
        assert fabric not in blob, f"recoding channel not stripped: {fabric}"


def test_strip_helpers():
    assert strip_fabric_words("Slim cotton tee") == "Slim tee"
    assert "cotton" not in strip_metadata("Made of 100% cotton. Round neck.").lower()
    assert "neck" in strip_metadata("Made of 100% cotton. Round neck.").lower()


def test_image_part_present_and_absent():
    rep = {"prod_name": "Tee", "detail_desc": ""}
    with_img = build_e2_messages(rep, "ZmFrZQ==")
    assert any(p.get("type") == "input_image" for p in with_img[1]["content"])
    without = build_e2_messages(rep, None)
    assert all(p.get("type") != "input_image" for p in without[1]["content"])


# --------------------------------------------------------------------------- validator
def test_validator_good_and_bad():
    assert validate_e2(CANNED).is_valid
    bad = dict(CANNED)
    bad["e2_trend_look"] = "Trendy"  # out-of-enum -> warning
    bad.pop("e2_body_ease")  # missing -> error
    vr = validate_e2(bad)
    assert not vr.is_valid
    assert any("e2_body_ease" in e for e in vr.errors)
    assert any("out-of-enum" in w for w in vr.warnings)


def test_render_roundtrip():
    row = render_e2_row(CANNED)
    assert row["e2_care_flags"] == ["Delicate-trims", "Wrinkle-prone"]
    assert row["e2_occasion_primary"] == "Date-romantic"


# --------------------------------------------------------------------------- gaps
def test_gap_rank_maps_monotone():
    assert PRICE_TIER_RANK["T1"] < PRICE_TIER_RANK["T5"]
    assert TREND_LOOK_RANK["Dated"] < TREND_LOOK_RANK["Emerging"]
    assert TREND_PHASE_RANK["Declining"] < TREND_PHASE_RANK["Emerging"]
    assert TREND_PHASE_RANK["Insufficient"] is None


# --------------------------------------------------------------------------- extractor (mocked)
class _FakeUsage:
    input_tokens = 380
    output_tokens = 150


class _FakeResp:
    output_text = json.dumps(CANNED)
    usage = _FakeUsage()


class _FakeResponses:
    async def create(self, **kwargs):  # noqa: ANN003
        return _FakeResp()


class _FakeClient:
    def __init__(self, *a, **k):
        self.responses = _FakeResponses()


def test_extract_pilot_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeClient)

    # 3 product_codes, 2 SKUs each (variant propagation must copy to both).
    rows = []
    for c in range(3):
        for v in range(2):
            rows.append(
                {
                    "article_id": f"{c}{v}00000001",
                    "product_code": f"pc{c}",
                    "prod_name": f"Product {c}",
                    "detail_desc": "A nice top. Round neck.",
                }
            )
    articles = pd.DataFrame(rows)
    config = EnrichmentV2Config(concurrency=4, max_cost_usd=0.5, checkpoint_interval=10)

    result = asyncio.run(extract_e2_pilot(articles, Path("/nonexistent_images"), tmp_path, config))

    assert result.n_product_codes == 3
    assert result.n_articles == 6  # propagated to all SKUs
    assert result.n_api_calls == 3
    assert result.coverage == 1.0

    out = pd.read_parquet(tmp_path / "enrichment_v2_llm.parquet")
    assert len(out) == 6
    for col in E2_LLM_AXES:
        assert col in out.columns
    # list field survived the cache JSON round-trip
    flags = out["e2_care_flags"].iloc[0]
    assert list(flags) == ["Delicate-trims", "Wrinkle-prone"]
    # every SKU of a product_code carries the same LLM value (propagation)
    assert out.groupby("product_code")["e2_occasion_primary"].nunique().max() == 1


def test_extract_pilot_resume(tmp_path, monkeypatch):
    """Second run with everything cached makes zero API calls."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeClient)
    articles = pd.DataFrame(
        [
            {
                "article_id": "0100000001",
                "product_code": "pc0",
                "prod_name": "Tee",
                "detail_desc": "x",
            }
        ]
    )
    config = EnrichmentV2Config(concurrency=2, max_cost_usd=0.5)
    r1 = asyncio.run(extract_e2_pilot(articles, Path("/none"), tmp_path, config))
    assert r1.n_api_calls == 1
    r2 = asyncio.run(extract_e2_pilot(articles, Path("/none"), tmp_path, config))
    assert r2.n_api_calls == 0 and r2.n_cache_hits == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
