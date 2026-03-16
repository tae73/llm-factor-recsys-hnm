"""Unit tests for src/data/preprocessing module."""
import tempfile
from pathlib import Path

import duckdb
import pytest

from src.data.preprocessing import (
    load_and_convert_articles,
    load_and_convert_customers,
    load_and_convert_transactions,
    validate_raw_data,
)


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create minimal CSV files for testing."""
    # articles.csv with leading-zero article_ids
    (tmp_path / "articles.csv").write_text(
        "article_id,product_code,prod_name,product_type_no,product_type_name,"
        "product_group_name,graphical_appearance_no,graphical_appearance_name,"
        "colour_group_code,colour_group_name,perceived_colour_value_id,"
        "perceived_colour_value_name,perceived_colour_master_id,"
        "perceived_colour_master_name,department_no,department_name,"
        "index_code,index_name,index_group_no,index_group_name,"
        "section_no,section_name,garment_group_no,garment_group_name,detail_desc\n"
        "0108775015,108775,Strap top,253,Vest top,Garment Upper body,1010016,Solid,"
        "9,Black,4,Dark,5,Black,1676,Jersey Basic,A,Ladieswear,1,Ladieswear,"
        "16,Womens Everyday Basics,1002,Jersey Basic,Nice top\n"
        "0108775044,108775,Strap top (1),253,Vest top,Garment Upper body,1010016,Solid,"
        "10,White,3,Light,9,White,1676,Jersey Basic,A,Ladieswear,1,Ladieswear,"
        "16,Womens Everyday Basics,1002,Jersey Basic,\n"
    )

    # customers.csv with some nulls
    (tmp_path / "customers.csv").write_text(
        "customer_id,FN,Active,club_member_status,fashion_news_frequency,age,postal_code\n"
        "c001,1.0,1.0,ACTIVE,Regularly,25,12345\n"
        "c002,,,,NONE,30,67890\n"
        "c003,1.0,,PRE-CREATE,,22,11111\n"
    )

    # transactions_train.csv
    (tmp_path / "transactions_train.csv").write_text(
        "t_dat,customer_id,article_id,price,sales_channel_id\n"
        "2020-01-15,c001,0108775015,0.050847,2\n"
        "2020-06-20,c001,0108775044,0.030508,1\n"
        "2020-07-10,c002,0108775015,0.050847,2\n"
        "2020-09-02,c003,0108775044,0.030508,1\n"
    )
    return tmp_path


def test_validate_raw_data(sample_data_dir):
    con = duckdb.connect()
    result = validate_raw_data(con, sample_data_dir)
    con.close()

    assert result["articles"]["n_rows"] == 2
    assert result["customers"]["n_rows"] == 3
    assert result["transactions"]["n_rows"] == 4
    assert result["transactions"]["date_range"] is not None


def test_articles_leading_zeros(sample_data_dir, tmp_path):
    output_dir = tmp_path / "output"
    con = duckdb.connect()
    path = load_and_convert_articles(con, sample_data_dir, output_dir)

    # Check article_id preserves leading zeros
    rows = con.execute(f"SELECT article_id FROM read_parquet('{path}')").fetchall()
    article_ids = [r[0] for r in rows]
    assert "0108775015" in article_ids
    assert "0108775044" in article_ids
    con.close()


def test_articles_null_detail_desc(sample_data_dir, tmp_path):
    output_dir = tmp_path / "output"
    con = duckdb.connect()
    path = load_and_convert_articles(con, sample_data_dir, output_dir)

    nulls = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}') WHERE detail_desc IS NULL"
    ).fetchone()[0]
    assert nulls == 0

    # Second article had empty detail_desc in CSV
    row = con.execute(
        f"SELECT detail_desc FROM read_parquet('{path}') WHERE article_id = '0108775044'"
    ).fetchone()
    assert row[0] == ""
    con.close()


def test_customers_null_handling(sample_data_dir, tmp_path):
    output_dir = tmp_path / "output"
    con = duckdb.connect()
    path = load_and_convert_customers(con, sample_data_dir, output_dir)

    # c002 had all nulls for FN, Active, club_member_status
    row = con.execute(
        f"SELECT fashion_news_frequency, Active, club_member_status "
        f"FROM read_parquet('{path}') WHERE customer_id = 'c002'"
    ).fetchone()
    assert row[0] == "NONE"
    assert row[1] == 0
    assert row[2] == "UNKNOWN"
    con.close()


def test_transactions_temporal_order(sample_data_dir, tmp_path):
    output_dir = tmp_path / "output"
    con = duckdb.connect()
    path = load_and_convert_transactions(con, sample_data_dir, output_dir)

    dates = con.execute(
        f"SELECT t_dat FROM read_parquet('{path}')"
    ).fetchall()
    date_list = [str(r[0]) for r in dates]
    assert date_list == sorted(date_list)
    con.close()


def test_transactions_article_id_varchar(sample_data_dir, tmp_path):
    output_dir = tmp_path / "output"
    con = duckdb.connect()
    path = load_and_convert_transactions(con, sample_data_dir, output_dir)

    rows = con.execute(f"SELECT article_id FROM read_parquet('{path}')").fetchall()
    for r in rows:
        assert isinstance(r[0], str)
        assert r[0].startswith("0")
    con.close()
