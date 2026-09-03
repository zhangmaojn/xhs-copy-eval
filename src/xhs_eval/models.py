from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConstraintChecks(BaseModel):
    required_terms: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    min_chars: int | None = Field(default=None, ge=0)
    max_chars: int | None = Field(default=None, ge=1)
    require_emoji: bool = False
    min_hashtags: int = Field(default=0, ge=0)
    max_hashtags: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_length_range(self) -> ConstraintChecks:
        if (
            self.min_chars is not None
            and self.max_chars is not None
            and self.min_chars > self.max_chars
        ):
            raise ValueError("min_chars cannot exceed max_chars")
        return self


class EvalExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    brief: str = Field(min_length=1)
    facts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    checks: ConstraintChecks = Field(default_factory=ConstraintChecks)
    reference: str = Field(min_length=1)
    split: Literal["train", "validation", "test"] = "test"
    tags: list[str] = Field(default_factory=list)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    output: str
    model: str
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AutomaticMetric(BaseModel):
    id: str
    char_count: int
    rouge_l_f1: float
    char_bleu: float
    constraint_score: float
    constraint_details: dict[str, bool]


class JudgeRecord(BaseModel):
    id: str
    scores: dict[str, float]
    overall: float
    passed: bool
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    attributions: list[str] = Field(default_factory=list)
    judge_model: str
    parse_error: str | None = None
