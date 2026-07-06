"""Stage registry: name -> (config class, run function). The CLI dispatches
through this dict; later slices add stages here (pretrain, label, rm, ppo...)."""

from tinyfables.config import PrepConfig, TokenizerConfig
from tinyfables.stages import prep, tokenizer

REGISTRY = {
    "tokenizer": (TokenizerConfig, tokenizer.run),
    "prep": (PrepConfig, prep.run),
}
