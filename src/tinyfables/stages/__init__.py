"""Stage registry: name -> (config class, module path). The CLI resolves the
module (and imports it) only at dispatch time, so heavy stages (torch/trl) only
pay their import cost when actually invoked."""

from tinyfables.config import (
    AuditConfig,
    BenchmarkConfig,
    DeriveConfig,
    EvalConfig,
    GateConfig,
    LabelConfig,
    MarginsConfig,
    PairgenConfig,
    PPOStageConfig,
    PrepConfig,
    PretrainConfig,
    RewardTrainConfig,
    TokenizerConfig,
)

REGISTRY: dict[str, tuple[type, str]] = {
    "tokenizer": (TokenizerConfig, "tinyfables.stages.tokenizer"),
    "prep": (PrepConfig, "tinyfables.stages.prep"),
    "benchmark": (BenchmarkConfig, "tinyfables.stages.benchmark"),
    "pretrain": (PretrainConfig, "tinyfables.stages.pretrain"),
    "evaluate": (EvalConfig, "tinyfables.stages.evaluate"),
    "audit": (AuditConfig, "tinyfables.stages.audit"),
    "pairgen": (PairgenConfig, "tinyfables.stages.pairgen"),
    "label": (LabelConfig, "tinyfables.stages.label"),
    "derive": (DeriveConfig, "tinyfables.stages.derive"),
    "reward": (RewardTrainConfig, "tinyfables.stages.reward"),
    "margins": (MarginsConfig, "tinyfables.stages.margins"),
    "gate": (GateConfig, "tinyfables.stages.gate"),
    "ppo": (PPOStageConfig, "tinyfables.stages.ppo"),
}
