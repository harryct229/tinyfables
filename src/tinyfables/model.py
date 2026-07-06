"""The hand-written decoder-only GPT (design-record config: 6 layers, d_model
384, 6 heads, ctx 1024, pre-norm LayerNorm, GELU, learned absolute positions,
tied embeddings, dropout 0). Attention is written out explicitly (QKV -> scaled
dot-product -> causal-mask softmax -> output projection) rather than delegated to
a fused primitive: this is the pedagogical core of the project.

Wrapped as a transformers PreTrainedModel + GenerationMixin so `.generate()`,
`save_pretrained`/`from_pretrained`, TRL, and push_to_hub work out of the box.

Weight tying (transformers 5.x contract): declare `_tied_weights_keys` as a
{tied -> source} dict, pass `tie_word_embeddings=True` through the config, and let
`post_init()` perform the tie. Do NOT alias `head.weight = tok.weight` by hand —
that aliasing does not survive `from_pretrained` (the param tensor is replaced)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class GPTConfig(PretrainedConfig):
    model_type = "tinyfables_gpt"

    def __init__(
        self,
        vocab_size: int = 8192,
        n_layer: int = 6,
        n_head: int = 6,
        d_model: int = 384,
        n_ctx: int = 1024,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.n_ctx = n_ctx
        # transformers 5.x does NOT default tie_word_embeddings on the base config,
        # so pass it explicitly or tying is silently skipped.
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj.RESIDUAL = True  # scaled-down init (see GPT._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        # Causal mask built inline, NOT a registered buffer: transformers 5.x
        # constructs the model under a meta device in from_pretrained, and a
        # non-persistent buffer (absent from the checkpoint) would be materialized
        # as uninitialized garbage — corrupting the mask after a round-trip.
        future = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        att = att.masked_fill(future, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)
        self.c_proj.RESIDUAL = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # pre-norm
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(PreTrainedModel, GenerationMixin):
    config_class = GPTConfig
    _tied_weights_keys = {"head.weight": "tok.weight"}  # {tied -> source}

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self.tok = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.n_ctx, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lnf = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()  # inits weights AND ties head.weight <- tok.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL", False):  # keep residual stream bounded (fp16)
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self) -> nn.Module:
        return self.tok

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.tok = value

    def get_output_embeddings(self) -> nn.Module:
        return self.head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.head = value

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        B, T = input_ids.shape
        if T > self.config.n_ctx:
            raise ValueError(f"sequence length {T} exceeds n_ctx {self.config.n_ctx}")
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok(input_ids) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.lnf(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)
