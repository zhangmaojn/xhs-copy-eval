from pathlib import Path

from xhs_eval.product_catalog import (
    ProductRecord,
    catalog_summary,
    find_product,
    load_product_catalog,
    select_products,
)

CATALOG = Path("data/products/alibaba_products_20260908.jsonl")


def test_real_catalog_shape_and_distribution() -> None:
    records = load_product_catalog(CATALOG)
    summary = catalog_summary(records)

    assert summary["products"] == 200
    assert summary["unique_product_ids"] == 200
    assert len(summary["categories"]) == 20
    assert set(summary["categories"].values()) == {10}
    assert summary["ad_records"] == 0
    assert summary["unparsed_prices"] == 0
    assert summary["unparsed_minimum_orders"] == 0


def test_grounded_brief_separates_source_claims() -> None:
    records = load_product_catalog(CATALOG)
    product = records[0]
    brief = product.grounded_brief()

    assert brief["product_id"] == product.product_id
    assert brief["source"]["url"] == product.source_url
    assert brief["unverified_source_claims"][0] == product.product_name
    assert "不得把" in brief["generation_rule"]
    assert find_product(records, product.record_id) == product


def test_product_id_must_match_record_and_url() -> None:
    data = load_product_catalog(CATALOG)[0].model_dump(mode="json")
    data["record_id"] = "ali-wrong"

    try:
        ProductRecord.model_validate(data)
    except ValueError as exc:
        assert "record_id" in str(exc)
    else:
        raise AssertionError("invalid record_id was accepted")


def test_select_products_by_category() -> None:
    records = load_product_catalog(CATALOG)
    selected = select_products(records, category="茶具", limit=3)

    assert len(selected) == 3
    assert {row.category for row in selected} == {"茶具"}
