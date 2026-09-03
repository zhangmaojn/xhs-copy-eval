from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from xhs_eval.models import EvalExample, Prediction

ModelT = TypeVar("ModelT", bound=BaseModel)


class DataValidationError(ValueError):
    """Raised when an input file is malformed or internally inconsistent."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise DataValidationError(f"{source}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def load_models(path: str | Path, model_type: type[ModelT]) -> list[ModelT]:
    source = Path(path)
    models: list[ModelT] = []
    for line_number, row in enumerate(read_jsonl(source), start=1):
        try:
            models.append(model_type.model_validate(row))
        except ValidationError as exc:
            raise DataValidationError(f"{source}:{line_number}: {exc}") from exc
    return models


def load_examples(path: str | Path) -> list[EvalExample]:
    examples = load_models(path, EvalExample)
    _ensure_unique_ids(examples, Path(path))
    return examples


def load_predictions(path: str | Path) -> list[Prediction]:
    predictions = load_models(path, Prediction)
    _ensure_unique_ids(predictions, Path(path))
    return predictions


def _ensure_unique_ids(rows: Iterable[BaseModel], source: Path) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        row_id = str(row.id)
        if row_id in seen:
            duplicates.add(row_id)
        seen.add(row_id)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise DataValidationError(f"{source}: duplicate ids: {joined}")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    lines: list[str] = []
    for row in rows:
        if isinstance(row, BaseModel):
            value = row.model_dump(mode="json")
        else:
            value = row
        lines.append(json.dumps(value, ensure_ascii=False))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, "\n".join(lines) + ("\n" if lines else ""))


def _atomic_write(target: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, target)
