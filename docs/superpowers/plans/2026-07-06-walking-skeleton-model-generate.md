# Walking Skeleton — Hand-Written GPT Writes Its First Fable (Issue 02) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the walking skeleton — a hand-written decoder-only GPT (wrapped as a Hugging Face `PreTrainedModel`), a toy pretraining stage over the packed shards, and one generation entrypoint that turns a `FableSpec` **or** free instruction text into fable text — so the system generates (terrible) fables end-to-end on CPU, and every later slice improves an existing path.

**Architecture:** The pedagogical core (`model.py`) is a from-scratch GPT: explicit multi-head causal self-attention (QKV projection → scaled dot-product → causal-mask softmax → output projection), pre-norm LayerNorm + GELU blocks, learned absolute positions, tied input/output embeddings, dropout 0. It subclasses `transformers.PreTrainedModel` + `GenerationMixin` so `.generate()`, `save_pretrained`/`from_pretrained`, TRL and Hub compat come for free. A torch-free `prompts.py` renders a `FableSpec` to the dataset's Canonical Prompt; `generate.py` is the generation contract that encodes the prompt **exactly as the prep stage does** (prompt encoded on its own, no separator) and decodes the fable continuation. A new `pretrain` stage plugs into the existing lazy stage registry: it streams the packed uint16/uint8 shards, trains with completion-only next-token loss (labels `-100` on prompt spans), and checkpoints model + optimizer + step so training is resumable and **exactly** reproducible. Everything runs at toy scale on the committed 24-row fixture, offline, on CPU.

**Tech Stack:** Python ≥3.10 (dev interpreter is 3.14), `torch` (model + training), `transformers` 5.x (`PreTrainedModel`/`GenerationMixin` wrapper), `tokenizers` (encode/decode), `numpy` (memmap shards), `pyyaml` (configs), `pytest`.

## Global Constraints

- Python ≥ 3.10; package lives under `src/tinyfables/`; the dev venv is `.venv` (use `.venv/bin/python` and `.venv/bin/pytest` — shell state does not persist between commands, so never rely on `source activate` carrying over).
- **Model config (design.md, verbatim)** — the full/real architecture: `n_layer=6`, `d_model=384`, `n_head=6`, `n_ctx=1024`, pre-norm LayerNorm, GELU, learned absolute positions, tied embeddings, dropout 0, vocab 8192. Measured unique parameter count at this config: **14,186,496** (≈14.19M = ~13.8M transformer+token-embedding + 0.39M learned positions; the design's "13.7M" headline counts transformer+token-embedding and folds the learned positions into rounding).
- **transformers 5.x weight-tying contract (verified on 5.13):** the model declares `_tied_weights_keys = {"head.weight": "tok.weight"}` (a `{tied → source}` dict, **not** a list), the config passes `tie_word_embeddings=True` into `super().__init__` (5.x does **not** default it), and tying is performed by `self.post_init()` — never alias `self.head.weight = self.tok.weight` by hand (it does not survive `from_pretrained`). Implement `get_input_embeddings`/`set_input_embeddings`/`get_output_embeddings`/`set_output_embeddings`.
- **Prompt/fable separate-encoding contract (from `stages/prep.py`):** training encodes prompt and fable **separately** and concatenates them with no separator token and no merged encoding across the boundary. Generation MUST encode the prompt the same way — `tokenizer.encode(prompt).ids` for the prompt alone — so train and inference token streams match.
- Loss is **completion-only**: labels are the input ids with prompt positions set to `-100` (`ignore_index`). The prep shards' `mask.bin` marks loss tokens (1 = fable or EOT); build labels as `input_ids` with `mask==0 → -100`.
- **Determinism for resume-equality:** batches are a pure function of the training step (`torch.Generator().manual_seed(seed + step)`); dropout is 0; the checkpoint saves model (`save_pretrained`) **and** optimizer `state_dict` **and** the completed step count. This yields bit-exact resume (measured: max param diff 0.0 across train-N vs train-k/resume/train-(N−k)).
- Special tokens are exactly `<|endoftext|>` (id 0) and `<|pad|>` (id 1); `constants.EOT`/`PAD` are frozen. Generation stops on and strips the EOT token from the returned fable.
- **Dependencies:** `torch` and `transformers` are added to core `dependencies`, but they MUST stay out of the import path of the light stages and the CLI dispatch — the stage registry is already lazy (name → module-path string, imported only at dispatch), and the CLI's `generate` branch imports torch lazily. Importing `tinyfables.cli`, `tinyfables.config`, `tinyfables.stages` (the registry), `tinyfables.prompts`, `tinyfables.stages.tokenizer`, or `tinyfables.stages.prep` must not import torch.
- Tests run offline by default (`pytest` deselects `network`-marked tests via `addopts`); no test may touch the network. The full offline suite must finish in under 2 minutes on CPU — keep every torch test on a **tiny** model (2 layers, `d_model` 64, CPU), never the 14M config.
- Manifests are written **last** (marks stage completion); the pretrain stage records its input shards + tokenizer as `inputs` (provenance), exactly as prep does.
- Every commit message ends with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

- `src/tinyfables/model.py` — **create.** `GPTConfig(PretrainedConfig)` + `GPT(PreTrainedModel, GenerationMixin)` and its blocks. The pedagogical core; the only place explicit attention lives. Imports torch + transformers (lazy — only imported by training/generation/model tests, never by the registry or CLI import).
- `src/tinyfables/prompts.py` — **create.** `FableSpec` dataclass + `render_canonical_prompt`. **torch-free** (safe to import anywhere; issue 03 extends it with paraphrases).
- `src/tinyfables/generate.py` — **create.** `encode_prompt` + `generate_fable` — the generation contract. Imports torch.
- `src/tinyfables/stages/pretrain.py` — **create.** The pretrain stage: shard dataloader, deterministic batching, train loop, checkpoint save/resume, summary + manifest. Imports torch.
- `src/tinyfables/config.py` — **modify.** Add the `PretrainConfig` frozen dataclass (torch-free).
- `src/tinyfables/stages/__init__.py` — **modify.** Register `"pretrain"` in `REGISTRY` (module-path string, stays lazy).
- `src/tinyfables/stage.py` — **modify.** Add `torch`, `transformers` to `_VERSIONED_PACKAGES` for manifest provenance.
- `src/tinyfables/cli.py` — **modify.** Add a `generate` subcommand (lazy torch import) alongside the existing `run`.
- `pyproject.toml` — **modify.** Add `torch` + `transformers` to `dependencies`.
- `configs/pretrain_toy.yaml`, `configs/pretrain_full.yaml` — **create.**
- `tests/test_config.py` — **modify.** Add a `PretrainConfig` load test.
- `tests/test_model.py` — **create.** Param count + tying, causality, fp16 stability, loss-mask-zero-loss, save/load round-trip.
- `tests/test_prompts.py` — **create.** Canonical render matches the fixture; partial spec omits missing elements.
- `tests/test_generate.py` — **create.** FableSpec + free-text paths; `encode_prompt` matches the prep separate-encoding contract.
- `tests/test_pretrain_stage.py` — **create.** Overfit-one-batch; resume-equality; stage writes checkpoint + manifest + summary.
- `tests/test_toy_chain.py` — **create.** tokenizer → prep → pretrain → generate as a single CLI test that produces a fable.
- `docs/design.md`, `docs/issues/README.md` — **modify** (final task): record decisions, tick issue 02.

---

### Task 1: Dependencies, `PretrainConfig`, and manifest version provenance

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/tinyfables/config.py`
- Modify: `src/tinyfables/stage.py:18`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: the existing `load_config(path, cls)` loader (rejects unknown keys) and `_build` (only special-cases a nested `source`; `PretrainConfig` has no `source`, so it loads by plain kwargs).
- Produces: `tinyfables.config.PretrainConfig` — a frozen dataclass with fields `prep_dir: str`, `tokenizer_dir: str`, `n_layer: int = 6`, `n_head: int = 6`, `d_model: int = 384`, `n_ctx: int = 1024`, `batch_size: int = 16`, `steps: int = 1000`, `lr: float = 3e-4`, `weight_decay: float = 0.1`, `warmup_steps: int = 100`, `grad_clip: float = 1.0`, `save_every: int = 0`, `resume_from: str | None = None`, `device: str = "cpu"`, `seed: int = 0`. Later tasks build the model and training loop from it.

- [ ] **Step 1: Declare the new dependencies**

In `pyproject.toml`, replace the `dependencies` list so it reads:

```toml
dependencies = [
    "numpy>=1.26",
    "tokenizers>=0.15",
    "pyyaml>=6.0",
    "datasets>=2.19",
    "torch>=2.4",
    "transformers>=5.0",
]
```

(The `.venv` already has `torch==2.12.1` and `transformers==5.13.0` installed from environment setup — this step only records the requirement; no reinstall is needed.)

- [ ] **Step 2: Add `torch`/`transformers` to manifest provenance**

In `src/tinyfables/stage.py`, change line 18 from:

```python
_VERSIONED_PACKAGES = ["tinyfables", "tokenizers", "numpy"]
```

to:

```python
_VERSIONED_PACKAGES = ["tinyfables", "tokenizers", "numpy", "torch", "transformers"]
```

(`_versions()` already swallows `PackageNotFoundError`, so this is safe even if a package is absent.)

- [ ] **Step 3: Write the failing `PretrainConfig` test**

Append to `tests/test_config.py`:

```python
def test_load_pretrain_config(tmp_path):
    p = write_yaml(
        tmp_path,
        "prep_dir: runs/prep\ntokenizer_dir: runs/tok\n"
        "n_layer: 2\nd_model: 64\nn_head: 2\nn_ctx: 256\n"
        "batch_size: 4\nsteps: 50\nlr: 0.001\nseed: 0\n",
    )
    from tinyfables.config import PretrainConfig

    cfg = load_config(p, PretrainConfig)
    assert cfg.prep_dir == "runs/prep"
    assert cfg.tokenizer_dir == "runs/tok"
    assert cfg.n_layer == 2 and cfg.d_model == 64 and cfg.n_head == 2
    assert cfg.n_ctx == 256 and cfg.batch_size == 4 and cfg.steps == 50
    assert cfg.resume_from is None  # default


def test_pretrain_config_rejects_unknown_key(tmp_path):
    from tinyfables.config import PretrainConfig

    p = write_yaml(tmp_path, "prep_dir: a\ntokenizer_dir: b\nlayers: 6\n")
    with pytest.raises(KeyError):
        load_config(p, PretrainConfig)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: the two new tests FAIL/ERROR with `ImportError: cannot import name 'PretrainConfig'`; existing config tests still pass.

- [ ] **Step 5: Add the `PretrainConfig` dataclass**

In `src/tinyfables/config.py`, add after the `PrepConfig` class (before `_build`):

```python
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
    save_every: int = 0
    resume_from: str | None = None
    device: str = "cpu"
    seed: int = 0
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: all config tests pass (including the two new ones).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/tinyfables/config.py src/tinyfables/stage.py tests/test_config.py
git commit -m "feat: torch/transformers deps + PretrainConfig + manifest provenance

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: The hand-written GPT (`model.py`)

**Files:**
- Create: `src/tinyfables/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `torch`, `transformers` (`PreTrainedModel`, `PretrainedConfig`, `GenerationMixin`, `modeling_outputs.CausalLMOutput`).
- Produces:
  - `tinyfables.model.GPTConfig(vocab_size=8192, n_layer=6, n_head=6, d_model=384, n_ctx=1024, tie_word_embeddings=True, **kwargs)` — a `PretrainedConfig` subclass with `model_type = "tinyfables_gpt"`.
  - `tinyfables.model.GPT(config)` — a `PreTrainedModel` + `GenerationMixin` subclass. `forward(input_ids, attention_mask=None, labels=None, **kwargs) -> CausalLMOutput` with `.logits` shape `(B, T, vocab)` and `.loss` (mean CE over shifted, non-`-100` targets) when `labels` is given. Input/output embeddings are tied (shared storage). Supports `.generate(...)`, `save_pretrained`, `from_pretrained`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:

```python
import math

import pytest
import torch
import torch.nn.functional as F

from tinyfables.model import GPT, GPTConfig


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, n_ctx=32)).eval()


def test_real_config_param_count_and_tying():
    m = GPT(GPTConfig())  # design defaults: 6/384/6/1024, vocab 8192
    unique = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    assert 13_500_000 < unique < 14_500_000  # measured 14,186,496
    assert m.head.weight.data_ptr() == m.tok.weight.data_ptr()  # shared storage


def test_forward_shapes():
    m = tiny()
    ids = torch.randint(0, 64, (3, 10))
    out = m(input_ids=ids)
    assert out.logits.shape == (3, 10, 64)
    assert out.loss is None


def test_future_tokens_do_not_change_current_logits():
    m = tiny()
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        base = m(input_ids=ids).logits
        perturbed = ids.clone()
        perturbed[0, -1] = (perturbed[0, -1] + 1) % 64
        after = m(input_ids=perturbed).logits
    assert torch.allclose(base[0, :9], after[0, :9], atol=1e-6)   # prefix unchanged
    assert not torch.allclose(base[0, 9], after[0, 9], atol=1e-6)  # perturbation is real


def test_fp16_forward_is_shape_and_dtype_stable():
    m = tiny().half()
    ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert out.dtype == torch.float16
    assert out.shape == (2, 16, 64)
    assert torch.isfinite(out).all()


def test_masked_prompt_tokens_contribute_zero_loss():
    m = tiny()
    ids = torch.randint(0, 64, (2, 8))
    labels = ids.clone()
    labels[:, :4] = -100  # mask the "prompt" half
    with torch.no_grad():
        out = m(input_ids=ids, labels=labels)
        shift_logits = out.logits[:, :-1].reshape(-1, 64)
        shift_labels = labels[:, 1:].reshape(-1)
        keep = shift_labels != -100
        manual = F.cross_entropy(shift_logits[keep], shift_labels[keep])
    assert torch.allclose(out.loss, manual, atol=1e-6)  # masked targets add exactly nothing


def test_save_load_round_trip_preserves_logits_and_tie(tmp_path):
    m = tiny()
    ids = torch.randint(0, 64, (1, 8))
    m.save_pretrained(tmp_path / "ckpt")
    m2 = GPT.from_pretrained(tmp_path / "ckpt").eval()
    assert m2.head.weight.data_ptr() == m2.tok.weight.data_ptr()
    with torch.no_grad():
        assert torch.allclose(m(input_ids=ids).logits, m2(input_ids=ids).logits, atol=1e-5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.model'`.

- [ ] **Step 3: Implement the model**

Create `src/tinyfables/model.py`:

```python
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
        mask = torch.tril(torch.ones(config.n_ctx, config.n_ctx)).view(
            1, 1, config.n_ctx, config.n_ctx
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_model.py -v`
Expected: all 6 tests pass (transformers may print a load-report line during the round-trip test — that is informational, not a failure).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/model.py tests/test_model.py
git commit -m "feat: hand-written GPT wrapped as HF PreTrainedModel

Explicit causal self-attention (pedagogical core), pre-norm/GELU blocks,
learned positions, tied embeddings (transformers 5.x _tied_weights_keys dict
+ post_init tying), dropout 0. Causality, fp16 stability, completion-only
loss masking, and save/load round-trip all covered by tests.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Canonical prompt rendering (`prompts.py`)

**Files:**
- Create: `src/tinyfables/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: nothing (torch-free, stdlib only).
- Produces:
  - `tinyfables.prompts.FableSpec` — a frozen dataclass: `character: str | None = None`, `setting: str | None = None`, `challenge: str | None = None`, `outcome: str | None = None`, `moral: str | None = None`, `age_range: str = "4-7"`, `word_count: int = 60`.
  - `tinyfables.prompts.render_canonical_prompt(spec: FableSpec) -> str` — renders the dataset's Canonical Prompt. For a fully-populated spec it reproduces the fixture corpus's prompt byte-for-byte (so a toy-trained model generates in-distribution). Elements whose value is `None` are omitted. Issue 03 will extend this module with paraphrased renderings.

**Note on the moral label:** the dataset (and the fixture) labels the moral element `Teaching:` — that is the dataset's field name, kept verbatim in the rendered text even though the project's domain term is "Moral" (see CONTEXT.md).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prompts.py`:

```python
import json
from pathlib import Path

from tinyfables.prompts import FableSpec, render_canonical_prompt

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl"


def test_full_spec_matches_fixture_canonical_prompt():
    # tiny_corpus.jsonl is written in itertools.product order; row 0 is
    # (a shy octopus, a quiet tide pool, courage grows by small steps).
    row0 = json.loads(FIXTURE.read_text().splitlines()[0])
    spec = FableSpec(
        character="a shy octopus",
        setting="a quiet tide pool",
        challenge="doubting oneself",
        outcome="a friend helps just in time",
        moral="courage grows by small steps",
        age_range="4-7",
        word_count=60,
    )
    assert render_canonical_prompt(spec) == row0["prompt"]


def test_partial_spec_omits_missing_elements():
    text = render_canonical_prompt(FableSpec(character="a brave mouse"))
    assert text.startswith("Create a fable based on the following elements")
    assert "- Main Character: a brave mouse" in text
    assert "- Setting:" not in text
    assert "- Teaching:" not in text
    assert text.endswith("and about 60 words.")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.prompts'`.

- [ ] **Step 3: Implement**

Create `src/tinyfables/prompts.py`:

```python
"""Rendering a FableSpec to the dataset's Canonical Prompt. Kept torch-free so it
can be imported anywhere. For a fully-populated spec, `render_canonical_prompt`
reproduces the dataset/fixture prompt byte-for-byte, so a model trained on that
distribution generates in-distribution. Issue 03 extends this with Paraphrased
Prompts; element values are always preserved verbatim so spec adherence stays
mechanically measurable."""

from __future__ import annotations

from dataclasses import dataclass

_HEADER = "Create a fable based on the following elements. Weave them naturally into a story:"

# (FableSpec field, rendered label). The moral is labelled "Teaching" — the
# dataset's own field name, kept verbatim in the prompt text.
_ELEMENTS = [
    ("character", "Main Character"),
    ("setting", "Setting"),
    ("challenge", "Challenge"),
    ("outcome", "Outcome"),
    ("moral", "Teaching"),
]


@dataclass(frozen=True)
class FableSpec:
    character: str | None = None
    setting: str | None = None
    challenge: str | None = None
    outcome: str | None = None
    moral: str | None = None
    age_range: str = "4-7"
    word_count: int = 60


def render_canonical_prompt(spec: FableSpec) -> str:
    lines = [_HEADER]
    for field, label in _ELEMENTS:
        value = getattr(spec, field)
        if value is not None:
            lines.append(f"- {label}: {value}")
    lines.append(
        f"Keep it age-appropriate for ages {spec.age_range} and about {spec.word_count} words."
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prompts.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/prompts.py tests/test_prompts.py
git commit -m "feat: FableSpec + canonical prompt rendering (matches dataset template)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: The generation contract (`generate.py`)

**Files:**
- Create: `src/tinyfables/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `tinyfables.model.GPT`, `tinyfables.prompts.FableSpec`/`render_canonical_prompt`, `tinyfables.constants.EOT`, a `tokenizers.Tokenizer`, `torch`.
- Produces:
  - `tinyfables.generate.encode_prompt(tokenizer, request) -> list[int]` — encodes a `FableSpec` (via `render_canonical_prompt`) or a raw `str` to prompt ids using `tokenizer.encode(text).ids` (the **same separate encoding** the prep stage uses; no separator, no specials prepended).
  - `tinyfables.generate.generate_fable(model, tokenizer, request, *, max_new_tokens=256, min_new_tokens=0, temperature=1.0, top_k=None, do_sample=False, seed=None, device=None) -> str` — encodes the prompt, runs `model.generate` (KV cache disabled — the walking-skeleton model recomputes the full window each step), decodes **only the continuation**, and strips at the first EOT. Returns the fable text.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate.py`:

```python
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.generate import encode_prompt, generate_fable
from tinyfables.model import GPT, GPTConfig
from tinyfables.prompts import FableSpec, render_canonical_prompt
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")

SPEC = FableSpec(
    character="a shy octopus",
    setting="a quiet tide pool",
    challenge="doubting oneself",
    outcome="a friend helps just in time",
    moral="courage grows by small steps",
)


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    out = tmp_path_factory.mktemp("tok")
    tokenizer_stage.run(TokenizerConfig(source=SourceSpec(jsonl_path=FIXTURE), vocab_size=512, seed=0), out)
    return Tokenizer.from_file(str(out / "tokenizer.json"))


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2, d_model=64, n_ctx=256)).eval()


def test_encode_prompt_matches_prep_separate_encoding(tok):
    # prep.py does tok.encode(prompt).ids for the prompt alone; the contract is
    # that generation encodes the prompt exactly the same way.
    prompt = render_canonical_prompt(SPEC)
    assert encode_prompt(tok, SPEC) == tok.encode(prompt).ids
    assert encode_prompt(tok, "Write a fable about a brave mouse.") == tok.encode(
        "Write a fable about a brave mouse."
    ).ids


def test_generate_from_fablespec_returns_text(model, tok):
    out = generate_fable(model, tok, SPEC, max_new_tokens=24, min_new_tokens=8, seed=0)
    assert isinstance(out, str) and len(out) > 0


def test_generate_from_free_text_returns_text(model, tok):
    out = generate_fable(model, tok, "Write a fable about a brave mouse.", max_new_tokens=24, min_new_tokens=8, seed=0)
    assert isinstance(out, str) and len(out) > 0


def test_generation_strips_at_endoftext(model, tok):
    from tinyfables.constants import EOT

    out = generate_fable(model, tok, SPEC, max_new_tokens=24, min_new_tokens=8, seed=0)
    assert EOT not in out  # the endoftext marker is never part of the returned fable
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.generate'`.

- [ ] **Step 3: Implement**

Create `src/tinyfables/generate.py`:

```python
"""The generation contract: one entrypoint that accepts either a FableSpec
(rendered to the Canonical Prompt) or raw instruction text and returns fable
text. The prompt is encoded exactly as the prep stage encodes it (prompt alone,
via `tokenizer.encode(prompt).ids`, no separator) so train and inference token
streams match. The walking-skeleton model has no KV cache, so generation runs
with use_cache=False (the full window is recomputed each step — fine at toy and
ctx-1024 demo scale)."""

from __future__ import annotations

import torch

from tinyfables.constants import EOT
from tinyfables.prompts import FableSpec, render_canonical_prompt


def encode_prompt(tokenizer, request) -> list[int]:
    text = render_canonical_prompt(request) if isinstance(request, FableSpec) else request
    return tokenizer.encode(text).ids


def generate_fable(
    model,
    tokenizer,
    request,
    *,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = False,
    seed: int | None = None,
    device=None,
) -> str:
    device = device or next(model.parameters()).device
    prompt_ids = encode_prompt(tokenizer, request)
    eot_id = tokenizer.token_to_id(EOT)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    if seed is not None:
        torch.manual_seed(seed)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=False,
        pad_token_id=eot_id,
        eos_token_id=eot_id,
    )
    if min_new_tokens:
        gen_kwargs["min_new_tokens"] = min_new_tokens
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)

    new_ids = out[0, len(prompt_ids):].tolist()
    if eot_id in new_ids:
        new_ids = new_ids[: new_ids.index(eot_id)]
    return tokenizer.decode(new_ids)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_generate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/generate.py tests/test_generate.py
git commit -m "feat: generation contract (FableSpec or free text -> fable)

Encodes the prompt exactly as prep does (separate encoding, no separator),
decodes only the continuation, strips at endoftext.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: The pretrain stage with checkpoint-resume (`stages/pretrain.py`)

**Files:**
- Create: `src/tinyfables/stages/pretrain.py`
- Modify: `src/tinyfables/stages/__init__.py`
- Test: `tests/test_pretrain_stage.py`

**Interfaces:**
- Consumes: `PretrainConfig`; `tinyfables.model.GPT`/`GPTConfig`; the prep stage's `tokens.bin`/`mask.bin`/`prep_summary.json`; a tokenizer dir (for `vocab_size`); `write_manifest`; `torch`, `numpy`, `tokenizers.Tokenizer`.
- Produces (in `stages/pretrain.py`):
  - `build_model(cfg: PretrainConfig, vocab_size: int) -> GPT` — constructs `GPT(GPTConfig(vocab_size=..., n_layer=cfg.n_layer, n_head=cfg.n_head, d_model=cfg.d_model, n_ctx=cfg.n_ctx))`.
  - `lr_at(step: int, cfg: PretrainConfig) -> float` — linear warmup for `warmup_steps`, then constant `cfg.lr`.
  - `get_batch(tokens, mask, n_windows, window, cfg, step, device) -> tuple[Tensor, Tensor]` — deterministic per step (`torch.Generator().manual_seed(cfg.seed + step)`); returns `(input_ids, labels)` where `labels = input_ids` with `mask==0 → -100`.
  - `run(cfg: PretrainConfig, out_dir: Path) -> None` — the stage entrypoint. Writes a checkpoint (`save_pretrained` → `config.json` + `model.safetensors` + `generation_config.json`), `optimizer.pt` (`{"optimizer": state_dict, "step": cfg.steps}`), `pretrain_summary.json`, and `manifest.json` (last). Resumes from `cfg.resume_from` when set (loads model + optimizer + start step), training up to `cfg.steps` total.
- Registry: `stages/__init__.py` maps `"pretrain" -> (PretrainConfig, "tinyfables.stages.pretrain")` (lazy module path — torch is imported only at dispatch).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pretrain_stage.py`:

```python
import json
from pathlib import Path

import pytest
import torch

from tinyfables.config import PrepConfig, PretrainConfig, SourceSpec, TokenizerConfig
from tinyfables.model import GPT
from tinyfables.stages import pretrain as pretrain_stage
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
SRC = SourceSpec(jsonl_path=FIXTURE)


@pytest.fixture(scope="module")
def toy_shards(tmp_path_factory):
    root = tmp_path_factory.mktemp("shards")
    tok_dir = root / "tok"
    prep_dir = root / "prep"
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), tok_dir)
    prep_stage.run(PrepConfig(source=SRC, tokenizer_dir=str(tok_dir), window=256, seed=0), prep_dir)
    return tok_dir, prep_dir


def toy_cfg(toy_shards, **overrides):
    tok_dir, prep_dir = toy_shards
    base = dict(
        prep_dir=str(prep_dir),
        tokenizer_dir=str(tok_dir),
        n_layer=2, n_head=2, d_model=64, n_ctx=256,
        batch_size=4, steps=6, lr=1e-3, warmup_steps=0, seed=0, device="cpu",
    )
    base.update(overrides)
    return PretrainConfig(**base)


def test_overfit_one_batch_drives_loss_down():
    # A tiny model must memorize a single fixed batch quickly.
    torch.manual_seed(0)
    cfg = PretrainConfig(prep_dir="unused", tokenizer_dir="unused",
                         n_layer=2, n_head=2, d_model=64, n_ctx=64, batch_size=4, steps=0, seed=0)
    model = pretrain_stage.build_model(cfg, vocab_size=48)
    ids = torch.randint(0, 48, (4, 32))
    labels = ids.clone()
    labels[:, :16] = -100  # mask the "prompt" half; loss only on the rest
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    assert losses[-1] < 0.5
    assert losses[-1] < losses[0] * 0.2


def test_stage_writes_checkpoint_and_manifest(toy_shards, tmp_path):
    out = tmp_path / "pretrain"
    pretrain_stage.run(toy_cfg(toy_shards, steps=4), out)
    names = {p.name for p in out.iterdir()}
    assert {"config.json", "model.safetensors", "generation_config.json",
            "optimizer.pt", "pretrain_summary.json", "manifest.json"} <= names
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "pretrain"
    assert manifest["artifacts"]
    assert "tokens.bin" in manifest["inputs"] and "tokenizer.json" in manifest["inputs"]
    summary = json.loads((out / "pretrain_summary.json").read_text())
    assert summary["steps"] == 4
    assert 13_000 < summary["n_params"] < 10_000_000  # tiny model, sanity band


def test_resume_equality_within_tolerance(toy_shards, tmp_path):
    full = tmp_path / "full"
    part = tmp_path / "part"
    resumed = tmp_path / "resumed"
    pretrain_stage.run(toy_cfg(toy_shards, steps=6), full)
    pretrain_stage.run(toy_cfg(toy_shards, steps=3), part)
    pretrain_stage.run(toy_cfg(toy_shards, steps=6, resume_from=str(part)), resumed)

    a = GPT.from_pretrained(full).state_dict()
    b = GPT.from_pretrained(resumed).state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), f"param {k} diverged on resume"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pretrain_stage.py -v`
Expected: FAIL with `ImportError: cannot import name 'pretrain' from 'tinyfables.stages'`.

- [ ] **Step 3: Implement the stage**

Create `src/tinyfables/stages/pretrain.py`:

```python
"""Pretrain stage: train the hand-written GPT on the packed shards with
completion-only next-token loss (labels -100 on prompt spans), then checkpoint
model + optimizer + step so a dead session costs at most a few steps.

Determinism: batches are a pure function of the step (seed + step), dropout is 0,
and the checkpoint round-trips model (safetensors, bit-exact) + optimizer state +
step. So train-N equals train-k then resume then train-(N-k) exactly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from tinyfables.config import PretrainConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.stage import write_manifest


def build_model(cfg: PretrainConfig, vocab_size: int) -> GPT:
    return GPT(
        GPTConfig(
            vocab_size=vocab_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            d_model=cfg.d_model,
            n_ctx=cfg.n_ctx,
        )
    )


def lr_at(step: int, cfg: PretrainConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(tokens, mask, n_windows, window, cfg, step, device):
    g = torch.Generator().manual_seed(cfg.seed + step)
    idx = torch.randint(0, n_windows, (cfg.batch_size,), generator=g).numpy()
    tok_rows = tokens.reshape(n_windows, window)[idx].astype(np.int64)
    mask_rows = mask.reshape(n_windows, window)[idx].astype(np.int64)
    input_ids = torch.from_numpy(tok_rows).to(device)
    labels = input_ids.clone()
    labels[torch.from_numpy(mask_rows).to(device) == 0] = -100
    return input_ids, labels


def run(cfg: PretrainConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)

    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    vocab_size = tok.get_vocab_size()

    prep_dir = Path(cfg.prep_dir)
    summary = json.loads((prep_dir / "prep_summary.json").read_text())
    window = summary["window"]
    if window > cfg.n_ctx:
        raise ValueError(f"prep window {window} exceeds model n_ctx {cfg.n_ctx}")
    tokens = np.memmap(prep_dir / "tokens.bin", dtype="<u2", mode="r")
    mask = np.memmap(prep_dir / "mask.bin", dtype=np.uint8, mode="r")
    n_windows = tokens.shape[0] // window
    if n_windows < cfg.batch_size:
        raise ValueError(f"only {n_windows} windows for batch_size {cfg.batch_size}")

    if cfg.resume_from:
        model = GPT.from_pretrained(cfg.resume_from).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        state = torch.load(Path(cfg.resume_from) / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
    else:
        torch.manual_seed(cfg.seed)
        model = build_model(cfg, vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        start_step = 0

    model.train()
    last_loss = float("nan")
    for step in range(start_step, cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        input_ids, labels = get_batch(tokens, mask, n_windows, window, cfg, step, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last_loss = out.loss.item()

    model.save_pretrained(out_dir)
    torch.save({"optimizer": optimizer.state_dict(), "step": cfg.steps}, out_dir / "optimizer.pt")

    n_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    summary_out = {
        "steps": cfg.steps,
        "start_step": start_step,
        "final_loss": round(last_loss, 4),
        "n_params": n_params,
        "vocab_size": vocab_size,
        "window": window,
        "n_windows": n_windows,
        "device": device,
    }
    (out_dir / "pretrain_summary.json").write_text(
        json.dumps(summary_out, indent=2, sort_keys=True) + "\n"
    )

    write_manifest(
        out_dir,
        "pretrain",
        cfg,
        [
            out_dir / "config.json",
            out_dir / "model.safetensors",
            out_dir / "optimizer.pt",
            out_dir / "pretrain_summary.json",
        ],
        inputs={
            "tokens.bin": prep_dir / "tokens.bin",
            "mask.bin": prep_dir / "mask.bin",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
```

- [ ] **Step 4: Register the stage**

In `src/tinyfables/stages/__init__.py`, update the imports and `REGISTRY`:

```python
"""Stage registry: name -> (config class, module path). The CLI resolves the
module (and imports it) only at dispatch time, so heavy stages (torch/trl) only
pay their import cost when actually invoked."""

from tinyfables.config import PrepConfig, PretrainConfig, TokenizerConfig

REGISTRY: dict[str, tuple[type, str]] = {
    "tokenizer": (TokenizerConfig, "tinyfables.stages.tokenizer"),
    "prep": (PrepConfig, "tinyfables.stages.prep"),
    "pretrain": (PretrainConfig, "tinyfables.stages.pretrain"),
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pretrain_stage.py -v`
Expected: 3 passed (`test_resume_equality_within_tolerance` is the key acceptance criterion; if it ever fails, the cause is nondeterminism — check that dropout is 0, batching is seeded by `seed + step`, and the optimizer state is saved/restored).

- [ ] **Step 6: Confirm the registry import stays torch-free**

Run:

```bash
.venv/bin/python -c "import sys, tinyfables.stages, tinyfables.cli; assert 'torch' not in sys.modules, 'torch leaked into the light import path'; print('registry/CLI import is torch-free')"
```

Expected: prints `registry/CLI import is torch-free`.

- [ ] **Step 7: Commit**

```bash
git add src/tinyfables/stages/pretrain.py src/tinyfables/stages/__init__.py tests/test_pretrain_stage.py
git commit -m "feat: pretrain stage with exact checkpoint-resume

Completion-only loss over packed shards, step-seeded deterministic batching,
model+optimizer+step checkpoint. Overfit-one-batch and resume-equality covered.
Stage registered lazily so torch stays out of the light import path.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `generate` CLI subcommand, configs, end-to-end toy chain, and docs

**Files:**
- Modify: `src/tinyfables/cli.py`
- Create: `configs/pretrain_toy.yaml`, `configs/pretrain_full.yaml`
- Test: `tests/test_toy_chain.py`
- Modify: `docs/design.md`, `docs/issues/README.md`

**Interfaces:**
- Consumes: everything above; the existing `load_config` + `REGISTRY` dispatch.
- Produces: `python -m tinyfables generate --checkpoint DIR --tokenizer DIR [--text ... | --character ... --setting ... --challenge ... --outcome ... --moral ... --age-range ... --word-count N] [--max-new-tokens N] [--min-new-tokens N] [--temperature T] [--top-k K] [--sample] [--seed S]` — prints a fable to stdout. `python -m tinyfables run pretrain --config <yaml> --out <dir>` now also works (registry). Issue 04 consumes `configs/pretrain_full.yaml`.

- [ ] **Step 1: Write the failing end-to-end toy-chain test**

Create `tests/test_toy_chain.py`:

```python
from pathlib import Path

from tinyfables.cli import main

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def test_toy_chain_tokenizer_prep_pretrain_generate(tmp_path, capsys):
    runs = tmp_path / "runs"
    tok_dir = runs / "tok"
    prep_dir = runs / "prep"
    ckpt = runs / "pretrain"

    (tmp_path / "tok.yaml").write_text(
        f"source:\n  jsonl_path: {FIXTURE}\nvocab_size: 512\nseed: 0\ncompare_gpt2: false\n"
    )
    (tmp_path / "prep.yaml").write_text(
        f"source:\n  jsonl_path: {FIXTURE}\ntokenizer_dir: {tok_dir}\nwindow: 256\nseed: 0\n"
    )
    (tmp_path / "pre.yaml").write_text(
        f"prep_dir: {prep_dir}\ntokenizer_dir: {tok_dir}\n"
        "n_layer: 2\nn_head: 2\nd_model: 64\nn_ctx: 256\n"
        "batch_size: 4\nsteps: 100\nwarmup_steps: 10\nlr: 0.001\nseed: 0\ndevice: cpu\n"
    )

    assert main(["run", "tokenizer", "--config", str(tmp_path / "tok.yaml"), "--out", str(tok_dir)]) == 0
    assert main(["run", "prep", "--config", str(tmp_path / "prep.yaml"), "--out", str(prep_dir)]) == 0
    assert main(["run", "pretrain", "--config", str(tmp_path / "pre.yaml"), "--out", str(ckpt)]) == 0

    capsys.readouterr()  # clear the stage-completion prints
    rc = main([
        "generate",
        "--checkpoint", str(ckpt),
        "--tokenizer", str(tok_dir),
        "--character", "a shy octopus",
        "--setting", "a quiet tide pool",
        "--challenge", "doubting oneself",
        "--outcome", "a friend helps just in time",
        "--moral", "courage grows by small steps",
        "--max-new-tokens", "60",
        "--min-new-tokens", "8",
        "--seed", "0",
    ])
    assert rc == 0
    fable = capsys.readouterr().out.strip()
    assert len(fable) > 0
    assert (ckpt / "model.safetensors").exists()
    assert (ckpt / "manifest.json").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_toy_chain.py -v`
Expected: FAIL — argparse errors on the unknown `generate` subcommand (`invalid choice: 'generate'`).

- [ ] **Step 3: Add the `generate` subcommand to the CLI**

Replace `src/tinyfables/cli.py` with:

```python
"""One CLI for every stage plus the generation demo:
    python -m tinyfables run <stage> --config c.yaml --out dir
    python -m tinyfables generate --checkpoint dir --tokenizer dir [--text ... | --character ... ]
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from tinyfables.config import load_config
from tinyfables.stages import REGISTRY


def _add_run_parser(sub) -> None:
    run_p = sub.add_parser("run", help="run a pipeline stage")
    run_p.add_argument("stage", choices=sorted(REGISTRY))
    run_p.add_argument("--config", required=True, help="path to the stage's YAML config")
    run_p.add_argument("--out", required=True, help="output directory for artifacts")


def _add_generate_parser(sub) -> None:
    gen_p = sub.add_parser("generate", help="write a fable from a trained checkpoint")
    gen_p.add_argument("--checkpoint", required=True, help="model checkpoint dir (save_pretrained)")
    gen_p.add_argument("--tokenizer", required=True, help="dir containing tokenizer.json")
    gen_p.add_argument("--text", help="raw instruction text (mutually exclusive with the element flags)")
    gen_p.add_argument("--character")
    gen_p.add_argument("--setting")
    gen_p.add_argument("--challenge")
    gen_p.add_argument("--outcome")
    gen_p.add_argument("--moral")
    gen_p.add_argument("--age-range", default="4-7")
    gen_p.add_argument("--word-count", type=int, default=250)
    gen_p.add_argument("--max-new-tokens", type=int, default=256)
    gen_p.add_argument("--min-new-tokens", type=int, default=0)
    gen_p.add_argument("--temperature", type=float, default=1.0)
    gen_p.add_argument("--top-k", type=int)
    gen_p.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    gen_p.add_argument("--seed", type=int)


def _run_stage(args) -> int:
    config_cls, module_path = REGISTRY[args.stage]
    cfg = load_config(args.config, config_cls)
    run_fn = importlib.import_module(module_path).run
    run_fn(cfg, Path(args.out))
    print(f"[tinyfables] stage '{args.stage}' complete -> {args.out}")
    return 0


def _generate(args) -> int:
    # torch/transformers imported lazily so `import tinyfables.cli` stays light.
    from tokenizers import Tokenizer

    from tinyfables.generate import generate_fable
    from tinyfables.model import GPT
    from tinyfables.prompts import FableSpec

    tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    model = GPT.from_pretrained(args.checkpoint).eval()
    if args.text is not None:
        request: object = args.text
    else:
        request = FableSpec(
            character=args.character,
            setting=args.setting,
            challenge=args.challenge,
            outcome=args.outcome,
            moral=args.moral,
            age_range=args.age_range,
            word_count=args.word_count,
        )
    fable = generate_fable(
        model,
        tok,
        request,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=args.sample,
        seed=args.seed,
    )
    print(fable)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinyfables")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_run_parser(sub)
    _add_generate_parser(sub)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _run_stage(args)
    return _generate(args)
```

- [ ] **Step 4: Run the toy-chain test to verify it passes**

Run: `.venv/bin/pytest tests/test_toy_chain.py -v`
Expected: 1 passed. (If the printed fable is empty, the tiny model emitted EOT immediately — the `--min-new-tokens 8` floor prevents this; if it ever recurs, raise the toy `steps`. The corpus is 24 near-identical fables, so ~100 steps memorizes the structure.)

- [ ] **Step 5: Add the checked-in configs**

Create `configs/pretrain_toy.yaml`:

```yaml
# Toy pretrain over the fixture shards (CPU, seconds) — the walking-skeleton demo.
prep_dir: runs/prep_toy
tokenizer_dir: runs/tokenizer_toy
n_layer: 2
n_head: 2
d_model: 64
n_ctx: 256
batch_size: 4
steps: 100
warmup_steps: 10
lr: 0.001
seed: 0
device: cpu
```

Create `configs/pretrain_full.yaml`:

```yaml
# Real architecture (design.md: 6/384/6, ctx 1024 -> ~14.2M params). The real
# training run — mixed precision, the 250M-token data budget, and Drive/HF
# checkpointing — lands in issue 04; this file fixes the model geometry now.
prep_dir: runs/prep_full
tokenizer_dir: runs/tokenizer_full
n_layer: 6
n_head: 6
d_model: 384
n_ctx: 1024
batch_size: 16
steps: 20000
warmup_steps: 200
lr: 0.0003
weight_decay: 0.1
grad_clip: 1.0
save_every: 1000
device: auto
seed: 0
```

Verify the documented demo command works exactly as a fresh engineer would run it from the repo root:

```bash
.venv/bin/python -m tinyfables run tokenizer --config configs/tokenizer_toy.yaml --out runs/tokenizer_toy \
  && .venv/bin/python -m tinyfables run prep --config configs/prep_toy.yaml --out runs/prep_toy \
  && .venv/bin/python -m tinyfables run pretrain --config configs/pretrain_toy.yaml --out runs/pretrain_toy \
  && .venv/bin/python -m tinyfables generate --checkpoint runs/pretrain_toy --tokenizer runs/tokenizer_toy \
       --character "a shy octopus" --setting "a quiet tide pool" --challenge "doubting oneself" \
       --outcome "a friend helps just in time" --moral "courage grows by small steps" \
       --max-new-tokens 80 --min-new-tokens 8 --seed 0
```

Expected: the three stages print `complete`, then a fable (likely rough — it's a 2-layer toy model) prints to stdout. (`runs/` is gitignored, so these artifacts are not committed.)

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/pytest`
Expected: all tests pass (1 `network`-marked test deselected), total well under 2 minutes.

- [ ] **Step 7: Record decisions in `docs/design.md`**

In `docs/design.md`, append a subsection to the **Model** section (after the "Sizing argument" line, before "## Build vs buy"):

```markdown
### Implementation (issue 02)

- **HF wrapper (transformers 5.x).** `GPT` subclasses `PreTrainedModel` + `GenerationMixin`
  so `.generate()`, `save_pretrained`/`from_pretrained`, TRL (issue 08) and `push_to_hub`
  (issue 11) come for free. Weight tying uses the 5.x contract: `_tied_weights_keys =
  {"head.weight": "tok.weight"}` (a dict, not a list), `tie_word_embeddings=True` passed
  explicitly through the config (5.x does not default it), and tying performed by
  `post_init()` — hand-aliasing the tensors does not survive `from_pretrained`.
- **Attention written out explicitly** (QKV projection → scaled dot-product → causal-mask
  softmax → output projection), not a fused SDPA/`MultiheadAttention` call — the model is
  the pedagogical core, so the causal mask and softmax are visible and directly tested.
- **Generation** uses HF `.generate()` with `use_cache=False` (no KV cache in the walking
  skeleton — the full window is recomputed each step; acceptable at ctx 1024 demo scale, a
  later optimization if needed). The generation *contract* (FableSpec/free-text → fable) is
  our own code and encodes the prompt exactly as prep does.
- **Measured parameter count** at the real config: **14,186,496** (≈14.19M) — 13.8M
  transformer+token-embedding + 0.39M learned positions. The 13.7M headline counts
  transformer+token-embedding.
- **Walking-skeleton training is fp32** on CPU/MPS/CUDA (device-selectable). Batches are a
  pure function of the step and dropout is 0, so checkpoint-resume is bit-exact (train-N ==
  train-k → resume → train-(N−k)). Mixed precision (fp16 + GradScaler) and the real
  data/checkpoint budget are deferred to issue 04.
- `torch` and `transformers` are core dependencies but kept out of the light import path
  (stage registry is lazy by module-path string; the CLI's `generate` branch imports torch
  lazily). The Canonical Prompt template lives in `prompts.py` (torch-free); Paraphrased
  Prompts extend it in issue 03.
```

- [ ] **Step 8: Tick issue 02 in the queue and commit**

In `docs/issues/README.md`, change the issue 02 row's `☐` to `☑` (the row: `| 02 | [Walking skeleton: hand-written GPT writes its first fable](./02-walking-skeleton-model-generate.md) | 01 | ☐ |`).

```bash
git add src/tinyfables/cli.py configs/pretrain_toy.yaml configs/pretrain_full.yaml \
        tests/test_toy_chain.py docs/design.md docs/issues/README.md
git commit -m "feat: generate CLI + pretrain configs + e2e toy chain

Closes issue 02 (walking skeleton): hand-written GPT trains on toy shards and
writes a fable end-to-end from either a FableSpec or free text. All acceptance
criteria covered by tests; full offline suite runs in seconds on CPU.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed)

**1. Spec coverage (issue 02 acceptance criteria):**
- Causality (perturbing future tokens does not change current logits) → Task 2 `test_future_tokens_do_not_change_current_logits`.
- fp16 stability (forward shape/dtype-stable in half precision) → Task 2 `test_fp16_forward_is_shape_and_dtype_stable` (verified: torch 2.12 CPU supports half matmul).
- Loss-mask (masked prompt tokens contribute exactly zero loss) → Task 2 `test_masked_prompt_tokens_contribute_zero_loss`.
- Overfit-one-batch drives loss near zero within minutes → Task 5 `test_overfit_one_batch_drives_loss_down`.
- Resume-equality within tolerance → Task 5 `test_resume_equality_within_tolerance` (verified: exact, max diff 0.0).
- Generation contract accepts both a FableSpec and free instruction text → Task 4 `test_generate_from_fablespec_returns_text` + `test_generate_from_free_text_returns_text`.
- Toy chain (prep → tokenizer → pretrain → generate) as a single test → Task 6 `test_toy_chain_tokenizer_prep_pretrain_generate`.
- Documented command produces a fable from a toy checkpoint → Task 6 `generate` subcommand + the documented command block in Step 5 + design.md.
- "Model-forward contract (seam 2)" → the `forward(input_ids, attention_mask, labels) -> CausalLMOutput` signature (Task 2) is the model-forward seam; "prompt/fable separate-encoding contract" honored by `encode_prompt` (Task 4) matching `prep.py`.

**2. Placeholder scan:** No TBDs. Every code step shows complete code; every run step gives an exact command and expected result. The only conditional guidance (empty-fable fallback) names a concrete, bounded fix (raise `steps`) — not a placeholder.

**3. Type consistency:** `PretrainConfig` fields are identical across Tasks 1/5/6 (config, stage, YAML). `GPT`/`GPTConfig` constructor and `forward(...) -> CausalLMOutput` are used identically in Tasks 2/4/5. `build_model(cfg, vocab_size)`, `lr_at(step, cfg)`, `get_batch(...)`, `run(cfg, out_dir)` signatures match between the pretrain implementation and its tests. `encode_prompt(tokenizer, request)`/`generate_fable(model, tokenizer, request, *, ...)` match between `generate.py`, its tests, and the CLI. `REGISTRY` value type stays `(type, module_path_str)` — consistent with the existing lazy-dispatch CLI. `_tied_weights_keys` is a dict everywhere. Manifest `inputs`/`artifacts` usage matches `write_manifest`'s existing signature (Task 5 passes `inputs=` exactly as `prep.py` does).
