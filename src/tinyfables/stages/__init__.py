"""Stage registry: name -> (config class, module path). The CLI resolves the
module (and imports it) only at dispatch time, so heavy stages (torch/trl) only
pay their import cost when actually invoked."""

from tinyfables.config import BenchmarkConfig, EvalConfig, PrepConfig, PretrainConfig, TokenizerConfig

REGISTRY: dict[str, tuple[type, str]] = {
    "tokenizer": (TokenizerConfig, "tinyfables.stages.tokenizer"),
    "prep": (PrepConfig, "tinyfables.stages.prep"),
    "benchmark": (BenchmarkConfig, "tinyfables.stages.benchmark"),
    "pretrain": (PretrainConfig, "tinyfables.stages.pretrain"),
    "evaluate": (EvalConfig, "tinyfables.stages.evaluate"),
}
