from __future__ import annotations

import argparse
import csv
from pathlib import Path

from xhs_eval.product_catalog import load_product_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将商品 JSONL 快照导出为便于查看的 CSV")
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_product_catalog(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "platform",
        "category",
        "product_id",
        "product_name",
        "listing_description",
        "price_text",
        "price_currency",
        "price_min",
        "price_max",
        "min_order_text",
        "min_order_quantity",
        "min_order_unit",
        "supplier_name",
        "supplier_meta",
        "rating_text",
        "review_text",
        "selling_points",
        "badges_text",
        "is_ad",
        "source_url",
        "image_url",
        "captured_at",
        "search_query",
        "source_language",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            price_min, price_max, currency = record.price_range()
            moq_quantity, moq_unit = record.minimum_order()
            description_parts = [record.product_name, *record.selling_points]
            writer.writerow(
                {
                    "record_id": record.record_id,
                    "platform": record.platform,
                    "category": record.category,
                    "product_id": record.product_id,
                    "product_name": record.product_name,
                    "listing_description": "；".join(dict.fromkeys(description_parts)),
                    "price_text": record.price_text,
                    "price_currency": currency or "",
                    "price_min": price_min if price_min is not None else "",
                    "price_max": price_max if price_max is not None else "",
                    "min_order_text": record.min_order_text,
                    "min_order_quantity": moq_quantity if moq_quantity is not None else "",
                    "min_order_unit": moq_unit or "",
                    "supplier_name": record.supplier_name,
                    "supplier_meta": record.supplier_meta.replace("\n", " / "),
                    "rating_text": record.rating_text,
                    "review_text": record.review_text.replace("\n", " / "),
                    "selling_points": " | ".join(record.selling_points),
                    "badges_text": record.badges_text.replace("\n", " / "),
                    "is_ad": record.is_ad,
                    "source_url": record.source_url,
                    "image_url": record.image_url,
                    "captured_at": record.captured_at.isoformat(),
                    "search_query": record.search_query,
                    "source_language": record.source_language,
                }
            )
    print(f"exported={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
