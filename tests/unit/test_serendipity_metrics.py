"""Unit tests for the R-10 serendipity/novelty/long-tail metrics in `_probe_common.py`.

Pure functions over a top-k index matrix + popularity/tail/unexpected masks + ground-truth sets.
Synthetic, deterministic, no I/O. The headline distinction tested: novelty is a cheap list
property; the relevance-grounded hit metrics (tail-hit, serendipity) only count items that are
BOTH recommended-in-topk AND in the user's ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from witnesses._probe_common import (
    hit_count,
    item_novelty,
    longtail_exposure,
    serendipitous_hit_count,
    tail_hit_count,
)


def test_item_novelty_higher_for_unpopular():
    # 4 items: item 0 very popular, item 3 very rare → recommending rare = higher novelty
    pop_prob = np.array([0.5, 0.25, 0.20, 0.05])
    topk_pop = np.array([[0, 1]])  # popular pair
    topk_rare = np.array([[2, 3]])  # rare pair
    nov_pop = item_novelty(topk_pop, pop_prob)
    nov_rare = item_novelty(topk_rare, pop_prob)
    assert nov_rare[0] > nov_pop[0]
    # exact: mean(-log2 pop) for rare = mean(-log2 0.2, -log2 0.05)
    assert nov_rare[0] == pytest.approx((-np.log2(0.20) - np.log2(0.05)) / 2)


def test_longtail_exposure_fraction():
    tail_mask = np.array([False, False, True, True])  # items 2,3 are tail
    topk = np.array([[0, 2, 3]])  # 2 of 3 are tail
    assert longtail_exposure(topk, tail_mask)[0] == pytest.approx(2 / 3)


def test_hit_count_only_counts_gt_items():
    canon = np.array(["a", "b", "c", "d"])
    gts = [{"b", "d"}]
    topk = np.array([[0, 1, 2]])  # a(no), b(hit), c(no)
    assert hit_count(topk, gts, canon, k=3)[0] == 1.0


def test_tail_hit_requires_relevance_and_tail():
    canon = np.array(["a", "b", "c", "d"])
    tail_mask = np.array([False, True, True, False])  # b,c are tail
    gts = [{"a", "b"}]  # a and b are relevant
    topk = np.array([[0, 1, 2]])  # a=hit-but-head, b=hit-and-tail, c=tail-but-not-relevant
    # only b qualifies (relevant AND tail)
    assert tail_hit_count(topk, gts, canon, tail_mask, k=3)[0] == 1.0


def test_serendipity_requires_relevance_and_unexpected():
    canon = np.array(["a", "b", "c"])
    gts = [{"a", "b"}]  # a,b relevant
    topk = np.array([[0, 1, 2]])
    unexpected = np.array([[False, True, True]])  # position0 expected, 1&2 unexpected
    # a=hit-but-expected(no), b=hit-and-unexpected(yes), c=unexpected-but-not-relevant(no)
    assert serendipitous_hit_count(topk, gts, canon, unexpected, k=3)[0] == 1.0


def test_novelty_not_inflated_by_relevance():
    """Novelty is a pure list property — independent of GT (the honesty point: cheap to inflate)."""
    pop_prob = np.array([0.4, 0.3, 0.2, 0.1])
    topk = np.array([[3, 2]])  # rare items
    nov = item_novelty(topk, pop_prob)
    # same list, regardless of whether they are hits → novelty is GT-independent
    assert nov[0] == pytest.approx((-np.log2(0.1) - np.log2(0.2)) / 2)


def test_metrics_deterministic_and_vectorized_over_users():
    canon = np.array([f"i{j}" for j in range(6)])
    tail_mask = np.array([False, False, True, True, True, True])
    gts = [{"i2", "i5"}, {"i0"}]
    topk = np.array([[2, 0, 5], [0, 3, 1]])
    a = tail_hit_count(topk, gts, canon, tail_mask, k=3)
    b = tail_hit_count(topk, gts, canon, tail_mask, k=3)
    assert np.array_equal(a, b)
    assert a.tolist() == [2.0, 0.0]  # user0: i2,i5 both tail-hits; user1: i0 is hit but head


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
