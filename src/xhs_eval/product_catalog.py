from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xhs_eval.io import DataValidationError, load_models

PRICE_RE = re.compile(r"US\$\s*([\d,.]+)(?:\s*-\s*([\d,.]+))?")
MOQ_RE = re.compile(r"最低起订量[:：]\s*([\d,.]+)\s*(.+)")


class ProductRecord(BaseModel):
    """A timestamped marketplace-listing snapshot, not an independently verified product."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    platform: Literal["Alibaba.com"]
    category: str = Field(min_length=1)
    search_query: str = Field(min_length=1)
    captured_at: datetime
    source_language: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    image_url: str = Field(min_length=1)
    price_text: str = Field(min_length=1)
    min_order_text: str = Field(min_length=1)
    supplier_name: str = Field(min_length=1)
    supplier_url: str = Field(min_length=1)
    supplier_meta: str = ""
    selling_points: list[str] = Field(default_factory=list)
    rating_text: str = ""
    review_text: str = ""
    badges_text: str = ""
    is_ad: bool
    raw_listing_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_match_source(self) -> ProductRecord:
        if self.record_id != f"ali-{self.product_id}":
            raise ValueError("record_id must be 'ali-' followed by product_id")
        if self.product_id not in self.source_url:
            raise ValueError("source_url must contain product_id")
        return self

    def price_range(self) -> tuple[Decimal | None, Decimal | None, str | None]:
        match = PRICE_RE.search(self.price_text)
        if not match:
            return None, None, None
        minimum = Decimal(match.group(1).replace(",", ""))
        maximum = Decimal((match.group(2) or match.group(1)).replace(",", ""))
        return minimum, maximum, "USD"

    def minimum_order(self) -> tuple[Decimal | None, str | None]:
        match = MOQ_RE.search(self.min_order_text)
        if not match:
            return None, None
        return Decimal(match.group(1).replace(",", "")), match.group(2).strip()

    def grounded_brief(self) -> dict[str, Any]:
        """Return facts and seller claims separately for safe copy generation."""

        price_min, price_max, currency = self.price_range()
        moq_quantity, moq_unit = self.minimum_order()
        return {
            "record_id": self.record_id,
            "product_id": self.product_id,
            "category": self.category,
            "product_name": self.product_name,
            "source": {
                "platform": self.platform,
                "url": self.source_url,
                "captured_at": self.captured_at.isoformat(),
            },
            "listing_facts": {
                "price_text": self.price_text,
                "price_currency": currency,
                "price_min": str(price_min) if price_min is not None else None,
                "price_max": str(price_max) if price_max is not None else None,
                "min_order_text": self.min_order_text,
                "min_order_quantity": (str(moq_quantity) if moq_quantity is not None else None),
                "min_order_unit": moq_unit,
                "supplier_name": self.supplier_name,
            },
            "unverified_source_claims": [self.product_name, *self.selling_points],
            "generation_rule": (
                "不得把卖家标题或卖点中的功效、材质、认证等主张升级为已验证事实；"
                "价格与起订量必须注明为采集时页面快照。"
            ),
        }


def load_product_catalog(path: str | Path) -> list[ProductRecord]:
    source = Path(path)
    records = load_models(source, ProductRecord)
    record_ids = [row.record_id for row in records]
    product_ids = [row.product_id for row in records]
    duplicate_records = sorted(key for key, count in Counter(record_ids).items() if count > 1)
    duplicate_products = sorted(key for key, count in Counter(product_ids).items() if count > 1)
    if duplicate_records:
        raise DataValidationError(f"{source}: duplicate record_id: {', '.join(duplicate_records)}")
    if duplicate_products:
        raise DataValidationError(
            f"{source}: duplicate product_id: {', '.join(duplicate_products)}"
        )
    return records


def catalog_summary(records: list[ProductRecord]) -> dict[str, Any]:
    categories = Counter(row.category for row in records)
    missing_price_parse = sum(row.price_range()[0] is None for row in records)
    missing_moq_parse = sum(row.minimum_order()[0] is None for row in records)
    return {
        "valid": True,
        "products": len(records),
        "unique_product_ids": len({row.product_id for row in records}),
        "categories": dict(sorted(categories.items())),
        "ad_records": sum(row.is_ad for row in records),
        "unparsed_prices": missing_price_parse,
        "unparsed_minimum_orders": missing_moq_parse,
    }


def find_product(records: list[ProductRecord], product_id: str) -> ProductRecord:
    normalized = product_id.removeprefix("ali-")
    for record in records:
        if record.product_id == normalized:
            return record
    raise KeyError(f"unknown product_id: {product_id}")


def select_products(
    records: list[ProductRecord], *, category: str | None = None, limit: int = 10
) -> list[ProductRecord]:
    if limit < 1:
        raise ValueError("limit must be positive")
    selected = (row for row in records if category is None or row.category == category)
    return list(selected)[:limit]
