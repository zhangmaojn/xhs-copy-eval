# Read-only product catalog

Use this reference only for catalog-grounded copy or product selection.

## Configured snapshot

- Project: the `xhs-copy-eval` repository containing `pyproject.toml`
- Catalog: `data/products/alibaba_products_20260908.jsonl` relative to the project root
- Scope: 200 Alibaba.com natural search-result cards captured on 2026-09-08
- Coverage: 20 consumer/lifestyle categories, 10 unique product IDs per category

This is a time-stamped source-of-record for experiments, not independent verification of seller
claims and not a live inventory feed.

## Retrieve, then write

For an exact product ID, run from the project directory:

```bash
uv run xhs-eval product-brief \
  --catalog data/products/alibaba_products_20260908.jsonl \
  --product-id '<id>'
```

For product selection by category:

```bash
uv run xhs-eval catalog-sample \
  --catalog data/products/alibaba_products_20260908.jsonl \
  --category '<category>' \
  --limit 5
```

Use the selected `product_id` in `product-brief` before drafting. Never draft from the compact
sample listing alone when the grounded brief can be retrieved.

## Evidence rules

- `listing_facts` are facts about what the page displayed at `captured_at`; price and minimum
  order are wholesale snapshot values, not current consumer retail offers.
- `unverified_source_claims` are seller wording. Attribute, soften, omit, or verify them; never
  upgrade healing, skincare, material, certification, quality, sales, delivery, or suitability
  language into established fact.
- Do not infer missing material, size, color, origin, inventory, consumer price, usage, or results.
- Preserve the exact product ID and `source.url` in internal output metadata when the user asks
  for traceability.
- For real publication, ask for or perform a fresh source check when current price, stock, or
  specifications matter. The snapshot alone is sufficient for a fixed offline evaluation.
