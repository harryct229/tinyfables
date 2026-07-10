"""Frozen dataclass configs loaded from YAML. Unknown keys are errors so a
typo in an experiment config fails loudly instead of silently using defaults."""

from __future__ import annotations

import math
from numbers import Real
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class SourceSpec:
    jsonl_path: str | None = None
    hf_dataset: str | None = None
    hf_split: str = "train"
    max_rows: int | None = None

    def __post_init__(self) -> None:
        if (self.jsonl_path is None) == (self.hf_dataset is None):
            raise ValueError("set exactly one of source.jsonl_path or source.hf_dataset")


@dataclass(frozen=True)
class TokenizerConfig:
    source: SourceSpec
    vocab_size: int = 8192
    seed: int = 0
    compare_gpt2: bool = False


@dataclass(frozen=True)
class PrepConfig:
    source: SourceSpec
    tokenizer_dir: str
    window: int = 1024
    seed: int = 0
    paraphrase_bank: str | None = None
    paraphrase_coverage: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.paraphrase_coverage <= 1.0:
            raise ValueError("paraphrase_coverage must be in [0, 1]")
        if self.paraphrase_coverage > 0.0 and self.paraphrase_bank is None:
            raise ValueError("paraphrase_coverage > 0 requires paraphrase_bank")


@dataclass(frozen=True)
class PretrainConfig:
    prep_dir: str
    tokenizer_dir: str
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    n_ctx: int = 1024
    batch_size: int = 16
    steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    resume_from: str | None = None
    device: str = "cpu"
    seed: int = 0
    amp: bool = False
    ckpt_dir: str | None = None
    ckpt_every: int = 0
    keep_last_k: int = 2
    log_every: int = 0
    trackio_project: str | None = None
    trackio_space_id: str | None = None
    run_name: str | None = None
    ckpt_hub_repo: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    tokenizer_dir: str
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    n_ctx: int = 1024
    batch_size: int = 16
    window: int = 1024
    warmup_steps: int = 5
    measure_steps: int = 20
    token_budget: int = 250_000_000
    target_tokens_per_sec: float = 0.0
    amp: bool = True
    device: str = "auto"
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: str
    tokenizer_dir: str
    source: SourceSpec
    paraphrase_bank: str | None = None
    n_ctx: int = 1024
    n_perplexity_rows: int = 500
    n_generations: int = 50
    max_new_tokens: int = 320
    min_new_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 50
    moral_threshold: float = 0.3
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class PairgenConfig:
    checkpoint: str
    tokenizer_dir: str
    source: SourceSpec
    n_ctx: int = 1024
    n_pairs: int = 2000
    max_new_tokens: int = 320
    min_new_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 50
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class LabelConfig:
    pairs: str
    rubric: str
    labeler_prompt: str
    model: str
    batch_size: int = 5
    swap_fraction: float = 0.10
    calibration_size: int = 30
    seed: int = 0
    max_batches: int | None = None
    workers: int = 1
    retry_attempts: int = 3
    retry_delay_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.swap_fraction <= 1.0:
            raise ValueError("swap_fraction must be in [0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")


@dataclass(frozen=True)
class DeriveConfig:
    labels: str
    pairs: str
    weight_delta: float = 0.1
    held_out_fraction: float = 0.10
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.held_out_fraction <= 1.0:
            raise ValueError("held_out_fraction must be in [0, 1]")
        if not 0.0 < self.weight_delta < 1.0:
            raise ValueError("weight_delta must be in (0, 1)")


@dataclass(frozen=True)
class AuditConfig:
    labels: str
    self_consistency_gate: float = 0.85
    position_swap_review_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.self_consistency_gate <= 1.0:
            raise ValueError("self_consistency_gate must be in [0, 1]")
        if not 0.0 <= self.position_swap_review_threshold <= 1.0:
            raise ValueError("position_swap_review_threshold must be in [0, 1]")


@dataclass(frozen=True)
class RewardTrainConfig:
    preferences: str
    base_checkpoint: str
    tokenizer_dir: str
    n_ctx: int = 1024
    batch_size: int = 16
    steps: int = 1000
    curve_steps: int = 400
    lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    log_every: int = 20
    curve_sizes: list[int] | None = None
    accuracy_gate: float = 0.65

    def __post_init__(self) -> None:
        if self.curve_sizes is None:
            object.__setattr__(self, "curve_sizes", [100, 500, 1000, 2000])
        if self.n_ctx <= 0:
            raise ValueError("n_ctx must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.curve_steps <= 0:
            raise ValueError("curve_steps must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")
        if not self.curve_sizes or any(size <= 0 for size in self.curve_sizes):
            raise ValueError("curve_sizes must contain positive integers")
        if not 0.0 <= self.accuracy_gate <= 1.0:
            raise ValueError("accuracy_gate must be in [0, 1]")


@dataclass(frozen=True)
class GateConfig:
    audit: str
    reward_summary: str
    rm_accuracy_gate: float = 0.65
    require_labeler_self_consistency: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rm_accuracy_gate, Real)
            or isinstance(self.rm_accuracy_gate, bool)
            or not math.isfinite(self.rm_accuracy_gate)
            or not 0.65 <= self.rm_accuracy_gate <= 1.0
        ):
            raise ValueError("rm_accuracy_gate must be a finite number in [0.65, 1]")
        if type(self.require_labeler_self_consistency) is not bool:
            raise ValueError("require_labeler_self_consistency must be a boolean")
        if not self.require_labeler_self_consistency:
            raise ValueError("labeler self-consistency is required by the production gate policy")


def _build(cls: type[T], data: Any) -> T:
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = dict(data)
    if "source" in kwargs:
        kwargs["source"] = _build(SourceSpec, kwargs["source"])
    return cls(**kwargs)


def load_config(path: str | Path, cls: type[T]) -> T:
    with open(path) as f:
        data = yaml.safe_load(f)
    return _build(cls, data or {})
