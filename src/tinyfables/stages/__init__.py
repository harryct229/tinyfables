"""Stage registry: name -> (config class, module path). The CLI resolves the
module (and imports it) only at dispatch time, so heavy stages (torch/trl) only
pay their import cost when actually invoked."""

from tinyfables.config import PrepConfig, PretrainConfig, TokenizerConfig

REGISTRY: dict[str, tuple[type, str]] = {
    "tokenizer": (TokenizerConfig, "tinyfables.stages.tokenizer"),
    "prep": (PrepConfig, "tinyfables.stages.prep"),
    "pretrain": (PretrainConfig, "tinyfables.stages.pretrain"),
}
