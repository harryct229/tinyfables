# Issue 08 — Aligned Model (PPO with DPO fallback) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the NO-GO alignment gate honestly, land both `ppo` and `dpo` stages as tested code, and ship `congthanh991/tinyfables-13m-aligned` to the Hub via the fork the maintainer picks (recorded as ADR-0005).

**Architecture:** Track A (this plan's Tasks 1–10) is local, offline-tested code on a feature branch: a HF-tokenizer wrapper, TRL-compat extensions to the hand-written GPT, a margin diagnostic stage, TRL-backed `ppo` and `dpo` stages with a shared length-drift alarm, and a `samples` (before/after) stage. Between Track A and Track B sit three main-session-only checkpoints: STEP 0 (mirror canonical `runs/` artifacts to the Hub — actually done FIRST, before Track A), the margin diagnostic run, and the fork decision (AskUserQuestion → ADR-0005). Track B runs the chosen path on Colab and pushes everything to the Hub before the session ends.

**Tech Stack:** Python 3.10+, torch 2.12, transformers 5.13, **trl 1.8.0 (new dependency, lazy)**, tokenizers, datasets, huggingface_hub. Tests: pytest, offline, `.venv/bin/pytest`.

## Global Constraints

- All compute runs (Track B) happen on **Colab**; every artifact must be pushed to the Hub **before the session ends** (unpushed runs didn't happen). Inputs are pulled from the Hub, not local `runs/`.
- Exception: the AI Labeler (`claude -p`) runs on the maintainer's Mac (subscription auth); its cache is then pushed to the Hub like any artifact.
- The gate record stays **honest**: the real `gate.json` verdict is NO-GO and must not be edited. PPO refuses to start without a gate-pass record (`tinyfables.gates.assert_gate_passed`); toy tests use toy gate records.
- TRL is a **lazy** dependency: never imported at module top-level of anything on the light import path (the stage registry already defers imports to dispatch time). trl lives only inside `stages/ppo.py`, `stages/dpo.py` (and their tests via `pytest.importorskip`).
- **TRL 1.8.0 facts (verified against the installed package on 2026-07-10 — do not "fix" these from memory):**
  - `PPOTrainer`/`PPOConfig` import ONLY from `trl.experimental.ppo`. Set env `TRL_EXPERIMENTAL_SILENCE=1` before import to silence the experimental warning.
  - `PPOTrainer(args, processing_class, model, ref_model, reward_model, train_dataset, value_model, ...)`. `train_dataset` is a `datasets.Dataset` with an `input_ids` column (tokenized prompts, unpadded), collated by `DataCollatorWithPadding(processing_class)` — so the tokenizer's `padding_side` must be `"left"` for PPO.
  - TRL calls policy/value/reward models as `model(input_ids=…, attention_mask=…, position_ids=…, return_dict=True, output_hidden_states=True)` where `position_ids = attention_mask.cumsum(1) - attention_mask.long()` and pads are `masked_fill`-ed to token id 0. **The policy must honor attention_mask and position_ids.**
  - reward_model/value_model contract: `getattr(model, model.base_model_prefix)` must be a backbone returning `.hidden_states[-1]` under `output_hidden_states=True`; `model.score(hidden)` → `(B, T, 1)`; TRL pools the last non-pad position itself.
  - Generation inside PPO: `lm_backbone.generate(input_ids=…, attention_mask=…, generation_config=…, return_dict_in_generate=True, output_scores=True)` with **left-padded** queries; position adjustment from the mask is the model's job. TRL's `GenerationConfig` does not set `use_cache`, so it defaults True — the GPT must be cache-proof (it has no KV cache).
  - Per-iteration metrics logged (visible in `TrainerCallback.on_log` `logs` dict): `objective/kl`, `objective/scores`, `objective/rlhf_reward`, `objective/non_score_reward`, `objective/entropy`, `loss/policy_avg`, `loss/value_avg`, `policy/approxkl_avg`, `policy/clipfrac_avg`, `val/num_eos_tokens`, `episode`, `lr`. **No response-length metric** → the length alarm is our own probe callback.
  - `DPOTrainer`/`DPOConfig` import from top-level `trl` (stable). `DPOTrainer(model=<PreTrainedModel>, ref_model=<PreTrainedModel>, args=DPOConfig, train_dataset=Dataset({"prompt","chosen","rejected"} strings), processing_class=<tokenizer>)`. It appends `eos_token_id` to completions itself (matches the RM's `prompt+fable+<|endoftext|>` convention) and skips bos (our tokenizer has none).
- Manifest contract: the Aligned Model's manifest records the **preferences.jsonl hash**, **Base checkpoint sha**, and the **ADR-0005 decision id** (config field `adr_decision`).
- Env: local dev in `.venv` (`.venv/bin/pytest`, `.venv/bin/python`); shell state does NOT persist between bash calls — always use absolute `.venv/bin/...` paths. Tests offline (fake gate records, replay fixtures). trl 1.8.0 + accelerate 1.14.0 are ALREADY installed in `.venv`.
- Specials: `<|endoftext|>` (id 0), `<|pad|>` (id 1) — from `tinyfables.constants` (`EOT`, `PAD`).
- Hub repos: `congthanh991/tinyfables-13m-base`, `-tokenizer`, `-13m-rm` (exist); `-preferences` (dataset repo, created in STEP 0); `-13m-aligned` (created in Track B). New repos default **private** (public flip is issue 11's carding job).
- Feature branch for Track A: `issue-08-aligned-model` off `main`.

---

## Main-session-only steps (not subagent tasks)

STEP 0, the diagnostic run, the fork decision, ADR-0005, and all of Track B are executed by the **main session** (they need Hub credentials, `runs/` artifacts on this Mac, AskUserQuestion, or colab-mcp). Everything under "Track A tasks" is subagent work on the feature branch.

## STEP 0 — Mirror canonical artifacts to the Hub (do FIRST, before any Track A task)

The canonical preference/label/audit artifacts live only in gitignored `runs/` on this Mac. **Discovered during planning:** `runs/derive_base/preferences.jsonl`, `runs/derive_base/derive_summary.json`, and `runs/audit_base/audit.json` are MISSING from disk, but their SHA-256 hashes are recorded in the surviving `manifest.json` files, and both stages are deterministic given the surviving inputs (`runs/labels_base/labels.jsonl`, `runs/pairgen_base/pairs.jsonl`). Regenerate, hash-verify, then mirror.

- [ ] **Step 0.1: Verify the inputs are the recorded ones**

```bash
cd /Users/thanh/code/tinystories && .venv/bin/python - <<'EOF'
import hashlib, pathlib
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha("runs/labels_base/labels.jsonl") == "c4939c2ab9fa7db9d9796211d416783de78e582c7b7650bd2cc2cc97912c99ad", "labels.jsonl drifted"
assert sha("runs/pairgen_base/pairs.jsonl") == "f28de9752b8a1af6902a3b6a12ddb99f467467916210ce26e686a15cc613f78b", "pairs.jsonl drifted"
print("inputs OK")
EOF
```

Expected: `inputs OK`. **If this fails, STOP — do not regenerate; surface the mismatch to the maintainer.**

- [ ] **Step 0.2: Regenerate derive + audit into scratch dirs and hash-verify against the recorded manifests**

```bash
cd /Users/thanh/code/tinystories && \
.venv/bin/python -m tinyfables run derive --config configs/derive_full.yaml --out /tmp/regen_derive && \
.venv/bin/python -m tinyfables run audit  --config configs/audit_full.yaml  --out /tmp/regen_audit && \
.venv/bin/python - <<'EOF'
import hashlib, pathlib
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha("/tmp/regen_derive/preferences.jsonl") == "6b2b8abbeed610e880bbd6478ecf54938b1adc2034958e71958b08f11e1f5387", "preferences.jsonl regeneration mismatch"
assert sha("/tmp/regen_derive/derive_summary.json") == "3031aa6d12c56d7fec33bbffb1a6f76dfeeed6c48dd61fc92e35f958fc425a1e", "derive_summary.json regeneration mismatch"
assert sha("/tmp/regen_derive/sensitivity.json") == "89c8761c5cc2659f6ae0288600574de7910e64e1cad25a1be4aefdce3b149dfa", "sensitivity.json regeneration mismatch"
assert sha("/tmp/regen_audit/audit.json") == "5c5282340335c9ba1d1457dd3013999f4036605fc486f1c91dcdd9960d592d9a", "audit.json regeneration mismatch"
print("regeneration byte-identical")
EOF
```

(Check `configs/derive_full.yaml` / `configs/audit_full.yaml` still point at `runs/labels_base/labels.jsonl` + `runs/pairgen_base/pairs.jsonl` with `weight_delta: 0.1`, `held_out_fraction: 0.10`, `seed: 0` — they matched the recorded manifest configs at plan time.)

Expected: `regeneration byte-identical`. If a hash mismatches, STOP and investigate before overwriting anything.

- [ ] **Step 0.3: Move the regenerated files into the canonical dirs**

```bash
cp /tmp/regen_derive/preferences.jsonl /tmp/regen_derive/derive_summary.json /Users/thanh/code/tinystories/runs/derive_base/ && \
cp /tmp/regen_audit/audit.json /Users/thanh/code/tinystories/runs/audit_base/
```

- [ ] **Step 0.4: Regenerate the missing gate manifest** (`runs/gate_base/` has `gate.json` + `gate_report.md` but no `manifest.json`; the gate stage is deterministic over `audit.json` + `reward_summary.json`)

```bash
cd /Users/thanh/code/tinystories && \
.venv/bin/python -m tinyfables run gate --config configs/gate_full.yaml --out /tmp/regen_gate && \
diff /tmp/regen_gate/gate.json runs/gate_base/gate.json && \
cp /tmp/regen_gate/manifest.json runs/gate_base/
```

Expected: `diff` silent (gate.json identical — the honest NO-GO record), manifest copied. If gate.json differs, STOP.

- [ ] **Step 0.5: Create the private dataset repo and upload**

```bash
cd /Users/thanh/code/tinystories && \
.venv/bin/hf repo create congthanh991/tinyfables-preferences --repo-type dataset --private ; \
.venv/bin/hf upload congthanh991/tinyfables-preferences runs/pairgen_base pairgen --repo-type dataset && \
.venv/bin/hf upload congthanh991/tinyfables-preferences runs/labels_base labels --repo-type dataset --exclude "invalid-v1/*" && \
.venv/bin/hf upload congthanh991/tinyfables-preferences runs/derive_base derive --repo-type dataset && \
.venv/bin/hf upload congthanh991/tinyfables-preferences runs/audit_base audit --repo-type dataset && \
.venv/bin/hf upload congthanh991/tinyfables-preferences runs/gate_base gate --repo-type dataset && \
.venv/bin/hf upload congthanh991/tinyfables-preferences runs/reward_base reward --repo-type dataset
```

(`labels_base/invalid-v1/` holds superseded caches — excluded. `reward_base/` is summaries + data curve only; the RM weights are already at `congthanh991/tinyfables-13m-rm`. If the `hf` CLI syntax differs in the installed version, `.venv/bin/hf upload --help` — or fall back to `huggingface_hub.HfApi().upload_folder(folder_path=…, path_in_repo=…, repo_id="congthanh991/tinyfables-preferences", repo_type="dataset")`.)

- [ ] **Step 0.6: Verify the round-trip** (the Operations rule: Colab pulls from the Hub, so prove a pull returns byte-identical files)

```bash
cd /Users/thanh/code/tinystories && .venv/bin/python - <<'EOF'
import hashlib, pathlib
from huggingface_hub import snapshot_download
d = snapshot_download("congthanh991/tinyfables-preferences", repo_type="dataset")
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
checks = {
    "derive/preferences.jsonl": "6b2b8abbeed610e880bbd6478ecf54938b1adc2034958e71958b08f11e1f5387",
    "labels/labels.jsonl": "c4939c2ab9fa7db9d9796211d416783de78e582c7b7650bd2cc2cc97912c99ad",
    "pairgen/pairs.jsonl": "f28de9752b8a1af6902a3b6a12ddb99f467467916210ce26e686a15cc613f78b",
    "audit/audit.json": "5c5282340335c9ba1d1457dd3013999f4036605fc486f1c91dcdd9960d592d9a",
}
for rel, want in checks.items():
    got = sha(pathlib.Path(d) / rel)
    assert got == want, f"{rel}: {got} != {want}"
print("round-trip verified")
EOF
```

Expected: `round-trip verified`.

---

## Track A — code on branch `issue-08-aligned-model` (subagent-driven, TDD, offline)

File map (what Track A creates/modifies):

| File | Responsibility |
|---|---|
| `src/tinyfables/hf_tokenizer.py` (new) | wrap `tokenizer.json` as a `PreTrainedTokenizerFast` with pad/eos set |
| `src/tinyfables/model.py` (modify) | GPT honors attention_mask + position_ids, supports `output_hidden_states`, cache-proof generation |
| `src/tinyfables/trl_compat.py` (new) | `ScoredModelAdapter` (base_model_prefix + `.score`) for TRL's reward/value contract |
| `src/tinyfables/length_alarm.py` (new) | probe-generation length stats + drift alarm (shared by ppo/dpo) |
| `src/tinyfables/stages/margins.py` (new) | RM accuracy stratified by aggregate-score margin (the STEP 1 diagnostic) |
| `src/tinyfables/stages/ppo.py` (new) | PPO stage: gate-guarded TRL PPOTrainer + curves + length alarm |
| `src/tinyfables/stages/dpo.py` (new) | DPO stage: TRL DPOTrainer on preferences + length alarm |
| `src/tinyfables/stages/samples.py` (new) | before/after generations for fixed FableSpecs |
| `src/tinyfables/config.py` (modify) | `MarginsConfig`, `PPOStageConfig`, `DPOStageConfig`, `SamplesConfig` |
| `src/tinyfables/stages/__init__.py` (modify) | register `margins`, `ppo`, `dpo`, `samples` |
| `pyproject.toml` (modify) | `align = ["trl>=1.8"]` optional extra |
| `configs/{margins_full,ppo_toy,ppo_full,dpo_toy,dpo_full,samples_full}.yaml` (new) | stage configs |
| `tests/fixtures/gate_pass.json`, `tests/fixtures/gate_nogo.json` (new) | toy gate records |
| `tests/test_hf_tokenizer.py`, `tests/test_model_trl_compat.py`, `tests/test_trl_compat.py`, `tests/test_length_alarm.py`, `tests/test_margins_stage.py`, `tests/test_ppo_stage.py`, `tests/test_dpo_stage.py`, `tests/test_samples_stage.py` (new) | per-unit tests |
| `tests/test_toy_chain.py` (modify) | extend toy chain: prep → … → ppo → evaluate |

### Task 1: `align` extra + HF tokenizer wrapper

**Files:**
- Modify: `pyproject.toml`
- Create: `src/tinyfables/hf_tokenizer.py`
- Test: `tests/test_hf_tokenizer.py`

**Interfaces:**
- Produces: `load_hf_tokenizer(tokenizer_dir: str | Path, padding_side: str = "right") -> PreTrainedTokenizerFast` — pad_token `<|pad|>`, eos_token `<|endoftext|>`, ids matching the raw `tokenizers.Tokenizer`. Used by Tasks 5 and 6.

- [ ] **Step 1: Add the optional extra to `pyproject.toml`** — in `[project.optional-dependencies]` add:

```toml
align = ["trl>=1.8"]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_hf_tokenizer.py
from pathlib import Path

from tokenizers import Tokenizer

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.constants import EOT, PAD
from tinyfables.hf_tokenizer import load_hf_tokenizer
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def _tok_dir(tmp_path):
    out = tmp_path / "tok"
    tokenizer_stage.run(TokenizerConfig(source=SourceSpec(jsonl_path=FIXTURE), vocab_size=512, seed=0), out)
    return out


def test_wrapper_matches_raw_tokenizer_and_sets_specials(tmp_path):
    tok_dir = _tok_dir(tmp_path)
    raw = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    hf = load_hf_tokenizer(tok_dir)
    text = "Create a fable about a shy octopus."
    assert hf(text, add_special_tokens=False)["input_ids"] == raw.encode(text).ids
    assert hf.pad_token == PAD and hf.pad_token_id == raw.token_to_id(PAD)
    assert hf.eos_token == EOT and hf.eos_token_id == raw.token_to_id(EOT)
    assert hf.bos_token_id is None
    assert hf.padding_side == "right"


def test_wrapper_left_padding_side(tmp_path):
    hf = load_hf_tokenizer(_tok_dir(tmp_path), padding_side="left")
    assert hf.padding_side == "left"
    batch = hf(["a b", "a b c d e"], padding=True, return_tensors="pt", add_special_tokens=False)
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["attention_mask"][0, 0].item() == 0  # short row left-padded
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_hf_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.hf_tokenizer'`

- [ ] **Step 4: Implement**

```python
# src/tinyfables/hf_tokenizer.py
"""Wrap the project tokenizer.json as a transformers PreTrainedTokenizerFast.

TRL trainers (issue 08) require a PreTrainedTokenizerBase `processing_class`;
the raw `tokenizers.Tokenizer` is not one. The wrapper preserves ids exactly
(no post-processor is added) and exposes the reserved specials. transformers
is imported lazily to keep the light import path light."""

from __future__ import annotations

from pathlib import Path

from tinyfables.constants import EOT, PAD


def load_hf_tokenizer(tokenizer_dir: str | Path, padding_side: str = "right"):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token=PAD,
        eos_token=EOT,
        padding_side=padding_side,
    )
    return hf
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_hf_tokenizer.py -v`
Expected: 2 PASS

- [ ] **Step 6: Verify the light import path stayed light**

Run: `.venv/bin/python -c "import sys, tinyfables.cli, tinyfables.hf_tokenizer; assert 'torch' not in sys.modules and 'transformers' not in sys.modules; print('light')"`
Expected: `light`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/tinyfables/hf_tokenizer.py tests/test_hf_tokenizer.py
git commit -m "feat(align): trl extra + PreTrainedTokenizerFast wrapper"
```

### Task 2: GPT TRL-compat — attention_mask, position_ids, hidden states, cache-proof generate

**Files:**
- Modify: `src/tinyfables/model.py`
- Test: `tests/test_model_trl_compat.py`

**Interfaces:**
- Consumes: existing `GPT`, `GPTConfig`, `CausalSelfAttention`, `Block`.
- Produces: `GPT.forward(input_ids, attention_mask=None, position_ids=None, labels=None, output_hidden_states=False, **kwargs) -> CausalLMOutput` where (a) attention over pad keys is masked, (b) `position_ids` defaults to arange (mask absent) or `(mask.cumsum(-1)-1).clamp(min=0)` (mask present), (c) `output_hidden_states=True` fills `CausalLMOutput.hidden_states` (tuple, last element = post-`lnf` states), and `GPT.prepare_inputs_for_generation` always feeds the full sequence (no KV-cache path can corrupt generation). **Behavior with `attention_mask=None` is bit-identical to today** — existing tests must pass unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_trl_compat.py
import torch

from tinyfables.model import GPT, GPTConfig


def _model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, n_ctx=32)).eval()


def test_no_mask_path_bit_identical_to_all_ones_mask():
    m = _model()
    ids = torch.randint(2, 64, (2, 7))
    with torch.no_grad():
        plain = m(ids).logits
        masked = m(ids, attention_mask=torch.ones_like(ids)).logits
    assert torch.equal(plain, masked)


def test_left_padded_forward_matches_unpadded_on_real_tokens():
    m = _model()
    ids = torch.randint(2, 64, (1, 6))
    pad = torch.zeros((1, 3), dtype=torch.long)  # TRL masked_fills pads to token id 0
    padded = torch.cat([pad, ids], dim=1)
    mask = torch.cat([torch.zeros_like(pad), torch.ones_like(ids)], dim=1)
    position_ids = mask.cumsum(1) - mask.long()
    with torch.no_grad():
        want = m(ids).logits
        got = m(padded, attention_mask=mask, position_ids=position_ids).logits[:, 3:, :]
    assert torch.allclose(want, got, atol=1e-5)


def test_output_hidden_states():
    m = _model()
    ids = torch.randint(2, 64, (2, 5))
    with torch.no_grad():
        out = m(ids, output_hidden_states=True)
    assert out.hidden_states is not None
    assert out.hidden_states[-1].shape == (2, 5, 32)
    # default stays None (CausalLMOutput contract unchanged for existing callers)
    assert m(ids).hidden_states is None


def test_generate_is_cache_proof():
    m = _model()
    ids = torch.randint(2, 64, (1, 5))
    with torch.no_grad():
        no_cache = m.generate(ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=0, eos_token_id=0)
        cached = m.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True, pad_token_id=0, eos_token_id=0)
    assert torch.equal(no_cache, cached)


def test_generate_handles_left_padded_batch():
    m = _model()
    a = torch.randint(2, 64, (1, 4))
    b = torch.randint(2, 64, (1, 7))
    pad_id = 1
    width = 7
    queries = torch.full((2, width), pad_id, dtype=torch.long)
    queries[0, width - 4:] = a
    queries[1, :] = b
    mask = queries != pad_id
    ids = torch.masked_fill(queries, ~mask, 0)
    with torch.no_grad():
        batch_out = m.generate(ids, attention_mask=mask, max_new_tokens=6, do_sample=False, pad_token_id=pad_id, eos_token_id=None)
        solo_out = m.generate(a, max_new_tokens=6, do_sample=False, pad_token_id=pad_id, eos_token_id=None)
    assert torch.equal(batch_out[0, width:], solo_out[0, 4:])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_model_trl_compat.py -v`
Expected: `test_no_mask_path_bit_identical_to_all_ones_mask` may already pass (mask currently ignored); the padded/hidden-states/left-pad tests FAIL (`hidden_states` is None; padded logits differ because mask and position_ids are ignored).

- [ ] **Step 3: Implement.** Three edits in `src/tinyfables/model.py`:

(a) `CausalSelfAttention.forward` gains an optional key-padding mask (use `torch.finfo(att.dtype).min`, NOT `-inf`, so fully-masked pad-query rows softmax to uniform instead of NaN):

```python
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
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
        if attention_mask is not None:
            # Key-padding mask (TRL batches are padded). finfo.min, not -inf: a
            # pad-query row has every key masked, and all--inf would softmax to NaN.
            pad_keys = ~attention_mask[:, None, None, :].to(torch.bool)
            att = att.masked_fill(pad_keys, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```

(b) `Block.forward` threads the mask: `def forward(self, x, attention_mask=None)` and `x = x + self.attn(self.ln1(x), attention_mask)`.

(c) `GPT.forward` + a `prepare_inputs_for_generation` override:

```python
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_hidden_states=False,
        **kwargs,
    ):
        B, T = input_ids.shape
        if T > self.config.n_ctx:
            raise ValueError(f"sequence length {T} exceeds n_ctx {self.config.n_ctx}")
        if position_ids is None:
            if attention_mask is not None:
                # Left-padded batches (TRL): real tokens get positions 0..n-1.
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
            else:
                position_ids = torch.arange(T, device=input_ids.device)[None, :]
        x = self.tok(input_ids) + self.pos(position_ids)
        hidden = [x] if output_hidden_states else None
        for block in self.blocks:
            x = block(x, attention_mask)
            if hidden is not None:
                hidden.append(x)
        x = self.lnf(x)
        if hidden is not None:
            hidden[-1] = x  # convention: last entry is the final (post-norm) states
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=tuple(hidden) if hidden is not None else None,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # No KV cache exists in this model: always recompute the full window,
        # whatever use_cache the caller's GenerationConfig requests (TRL's PPO
        # generation defaults use_cache=True). Dropping cache kwargs here keeps
        # GenerationMixin from cropping input_ids to cached positions.
        return {"input_ids": input_ids, "attention_mask": attention_mask}
```

Note the existing `x = self.tok(input_ids) + self.pos(pos)[None, :, :]` line is replaced: with `position_ids` shaped `(1, T)` or `(B, T)`, `self.pos(position_ids)` already broadcasts correctly — no `[None]` indexing.

- [ ] **Step 4: Run the new tests AND the full suite (bit-exactness regression: pretrain resume, evaluate, reward tests all exercise the old path)**

Run: `.venv/bin/pytest tests/test_model_trl_compat.py tests/test_model.py tests/test_generate.py tests/test_pretrain_stage.py tests/test_reward_model.py -v && .venv/bin/pytest -q`
Expected: all PASS. If `test_generate_is_cache_proof` fails because GenerationMixin injects a cache object anyway, the `prepare_inputs_for_generation` override above is the fix — make sure it actually overrides (method on `GPT`, not on the config).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/model.py tests/test_model_trl_compat.py
git commit -m "feat(model): attention_mask/position_ids/hidden-states support + cache-proof generate for TRL"
```

### Task 3: TRL score adapter (reward/value contract)

**Files:**
- Create: `src/tinyfables/trl_compat.py`
- Test: `tests/test_trl_compat.py`

**Interfaces:**
- Consumes: `GPT` (Task 2's `output_hidden_states`), `tinyfables.reward_model.RewardModel` / RM directory layout (`backbone/`, `reward_head.pt`).
- Produces: `ScoredModelAdapter(nn.Module)` with `base_model_prefix = "backbone"`, `.backbone` (a `GPT`), `.score` (`nn.Linear(d_model, 1)`), and classmethod `from_reward_model_dir(rm_dir, device="cpu") -> ScoredModelAdapter`. Satisfies exactly what TRL's `get_reward`/`PolicyAndValueWrapper` touch. Used twice by Task 5 (reward_model and value_model are independent instances).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trl_compat.py
import torch

from tinyfables.model import GPT, GPTConfig
from tinyfables.reward_model import RewardModel, save_reward_model
from tinyfables.trl_compat import ScoredModelAdapter


def _rm_dir(tmp_path):
    torch.manual_seed(0)
    backbone = GPT(GPTConfig(vocab_size=64, n_layer=1, n_head=2, d_model=32, n_ctx=32))
    rm = RewardModel(backbone)
    save_reward_model(rm, tmp_path / "rm")
    return tmp_path / "rm", rm


def test_adapter_satisfies_trl_reward_contract(tmp_path):
    rm_dir, rm = _rm_dir(tmp_path)
    adapter = ScoredModelAdapter.from_reward_model_dir(rm_dir)
    assert adapter.base_model_prefix == "backbone"
    backbone = getattr(adapter, adapter.base_model_prefix)

    ids = torch.randint(2, 64, (2, 6))
    mask = torch.ones_like(ids)
    mask[1, 4:] = 0  # right-padded row
    position_ids = mask.cumsum(1) - mask.long()
    with torch.no_grad():
        out = backbone(
            input_ids=torch.masked_fill(ids, ~mask.bool(), 0),
            attention_mask=mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = adapter.score(out.hidden_states[-1])
    assert logits.shape == (2, 6, 1)

    # pooled score at last non-pad == the original RewardModel's score
    with torch.no_grad():
        want = rm(torch.masked_fill(ids, ~mask.bool(), 0), mask)
    last = mask.sum(1) - 1
    got = logits[torch.arange(2), last].squeeze(-1)
    assert torch.allclose(want, got, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_trl_compat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.trl_compat'`

- [ ] **Step 3: Implement**

```python
# src/tinyfables/trl_compat.py
"""Adapters satisfying TRL's reward/value-model calling convention.

TRL's experimental PPOTrainer computes rewards as:
    lm_backbone = getattr(model, model.base_model_prefix)
    out = lm_backbone(input_ids=…, attention_mask=…, position_ids=…,
                      return_dict=True, output_hidden_states=True, use_cache=False)
    model.score(out.hidden_states[-1])   # then pools last non-pad itself
so an adapter needs exactly: `base_model_prefix`, the backbone attribute, and a
`.score` head. The TinyFables RewardModel (backbone + reward_head, last-non-pad
pooling) maps onto this 1:1."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from tinyfables.model import GPT


class ScoredModelAdapter(nn.Module):
    base_model_prefix = "backbone"

    def __init__(self, backbone: GPT) -> None:
        super().__init__()
        self.backbone = backbone
        self.score = nn.Linear(backbone.config.d_model, 1)

    @classmethod
    def from_reward_model_dir(
        cls, rm_dir: str | Path, device: str | torch.device = "cpu"
    ) -> "ScoredModelAdapter":
        root = Path(rm_dir)
        backbone = GPT.from_pretrained(root / "backbone").to(device)
        adapter = cls(backbone).to(device)
        state = torch.load(root / "reward_head.pt", map_location=device)
        adapter.score.load_state_dict(state)
        return adapter

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        out = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        return self.score(out.hidden_states[-1])
```

(`reward_head.pt` is an `nn.Linear(d_model, 1)` state dict — `weight` + `bias` keys load into `score` directly.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_trl_compat.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/trl_compat.py tests/test_trl_compat.py
git commit -m "feat(align): ScoredModelAdapter for TRL reward/value contract"
```

### Task 4: length-drift alarm helper

**Files:**
- Create: `src/tinyfables/length_alarm.py`
- Test: `tests/test_length_alarm.py`

**Interfaces:**
- Consumes: `tinyfables.generate.generate_fable(model, tokenizer, request, …)` (raw `tokenizers.Tokenizer`, existing).
- Produces (used by Tasks 5 and 6):
  - `probe_lengths(model, tokenizer, prompts: list[str], *, max_new_tokens: int, temperature: float, seed: int, device) -> dict` → `{"mean_words": float, "mean_new_tokens": float, "n_prompts": int}` (samples one fable per prompt, seeded `seed + i`).
  - `length_drift(baseline_mean_words: float, current_mean_words: float, threshold: float) -> dict` → `{"baseline_mean_words", "current_mean_words", "drift_pct", "alarm": bool}` where `drift_pct = abs(current-baseline)/baseline` and `alarm = drift_pct > threshold`. Baseline 0 → `drift_pct = 0.0`, no alarm (degenerate toy models).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_length_alarm.py
import torch

from tinyfables.length_alarm import length_drift, probe_lengths
from tinyfables.model import GPT, GPTConfig


def test_length_drift_flags_beyond_threshold():
    calm = length_drift(250.0, 260.0, threshold=0.25)
    assert calm["alarm"] is False and abs(calm["drift_pct"] - 0.04) < 1e-9
    loud = length_drift(250.0, 100.0, threshold=0.25)
    assert loud["alarm"] is True and abs(loud["drift_pct"] - 0.6) < 1e-9
    assert length_drift(0.0, 10.0, threshold=0.25) == {
        "baseline_mean_words": 0.0,
        "current_mean_words": 10.0,
        "drift_pct": 0.0,
        "alarm": False,
    }


def test_probe_lengths_is_deterministic_given_seed(tmp_path):
    from tinyfables.config import SourceSpec, TokenizerConfig
    from tinyfables.stages import tokenizer as tokenizer_stage
    from pathlib import Path
    from tokenizers import Tokenizer

    fixture = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
    tokenizer_stage.run(TokenizerConfig(source=SourceSpec(jsonl_path=fixture), vocab_size=512, seed=0), tmp_path / "tok")
    tok = Tokenizer.from_file(str(tmp_path / "tok" / "tokenizer.json"))
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=1, n_head=2, d_model=32, n_ctx=128)).eval()

    prompts = ["Create a fable about a fox.", "Create a fable about a crow."]
    a = probe_lengths(model, tok, prompts, max_new_tokens=12, temperature=0.9, seed=7, device="cpu")
    b = probe_lengths(model, tok, prompts, max_new_tokens=12, temperature=0.9, seed=7, device="cpu")
    assert a == b
    assert a["n_prompts"] == 2
    assert a["mean_new_tokens"] <= 12
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_length_alarm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.length_alarm'`

- [ ] **Step 3: Implement**

```python
# src/tinyfables/length_alarm.py
"""Length-drift alarm (issue 08): length bias is the classic reward hack, and
the ~250-word fable format makes it detectable. Both alignment stages probe a
fixed prompt set with the current policy and compare mean generated length to
the frozen Base Model's baseline."""

from __future__ import annotations

from statistics import mean


def probe_lengths(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    device=None,
) -> dict:
    from tinyfables.generate import generate_fable  # torch imported lazily via caller

    words: list[int] = []
    new_tokens: list[int] = []
    for i, prompt in enumerate(prompts):
        fable = generate_fable(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            seed=seed + i,
            device=device,
        )
        words.append(len(fable.split()))
        new_tokens.append(len(tokenizer.encode(fable).ids))
    return {
        "mean_words": mean(words) if words else 0.0,
        "mean_new_tokens": mean(new_tokens) if new_tokens else 0.0,
        "n_prompts": len(prompts),
    }


def length_drift(baseline_mean_words: float, current_mean_words: float, threshold: float) -> dict:
    drift = (
        abs(current_mean_words - baseline_mean_words) / baseline_mean_words
        if baseline_mean_words > 0
        else 0.0
    )
    return {
        "baseline_mean_words": baseline_mean_words,
        "current_mean_words": current_mean_words,
        "drift_pct": drift,
        "alarm": drift > threshold,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_length_alarm.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/length_alarm.py tests/test_length_alarm.py
git commit -m "feat(align): shared length-drift alarm helper"
```

### Task 5: `margins` stage — the STEP 1 diagnostic

**Files:**
- Modify: `src/tinyfables/config.py` (add `MarginsConfig`), `src/tinyfables/stages/__init__.py` (register)
- Create: `src/tinyfables/stages/margins.py`, `configs/margins_full.yaml`
- Test: `tests/test_margins_stage.py`

**Interfaces:**
- Consumes: `load_preferences(path, split=…)`, `collate_preference_batch`, `load_reward_model`, `preference_accuracy` (all existing), preferences rows carry `aggregate_chosen`/`aggregate_rejected`.
- Produces: stage `margins` writing `margins.json` (`{"overall_accuracy": float, "n_held_out": int, "buckets": [{"lo": float, "hi": float | None, "n": int, "accuracy": float | None}, …]}`), `margins_report.md`, `manifest.json`. Margin = `aggregate_chosen - aggregate_rejected` (> 0 by construction; ties were skipped at derive).

```python
@dataclass(frozen=True)
class MarginsConfig:
    preferences: str
    reward_model_dir: str
    tokenizer_dir: str
    n_ctx: int = 1024
    batch_size: int = 16
    bucket_edges: list[float] | None = None  # defaults to [0.1, 0.3, 0.6, 1.0]
    device: str = "auto"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.bucket_edges is None:
            object.__setattr__(self, "bucket_edges", [0.1, 0.3, 0.6, 1.0])
        if self.n_ctx <= 0:
            raise ValueError("n_ctx must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        edges = self.bucket_edges
        if not edges or any(e <= 0 for e in edges) or sorted(edges) != list(edges):
            raise ValueError("bucket_edges must be positive and ascending")
```

Buckets: `[edges[0], edges[1])`, …, `[edges[-1], ∞)` — rows with margin `< edges[0]` land in the first bucket too (edges[0] is the minimum possible margin 0.1, so nothing is dropped; assert every held-out row lands in a bucket).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_margins_stage.py
import json
from pathlib import Path

import torch

from tinyfables.config import MarginsConfig, SourceSpec, TokenizerConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.reward_model import RewardModel, save_reward_model
from tinyfables.stages import margins as margins_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURES = Path(__file__).parent / "fixtures"


def test_margins_stage_buckets_held_out_accuracy(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(FIXTURES / "tiny_corpus.jsonl")), vocab_size=512, seed=0),
        tok_dir,
    )
    torch.manual_seed(0)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    rm = RewardModel(GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=1, n_head=2, d_model=32, n_ctx=128)))
    save_reward_model(rm, tmp_path / "rm")

    out = tmp_path / "margins"
    margins_stage.run(
        MarginsConfig(
            preferences=str(FIXTURES / "preferences_replay.jsonl"),
            reward_model_dir=str(tmp_path / "rm"),
            tokenizer_dir=str(tok_dir),
            n_ctx=128,
            batch_size=2,
            device="cpu",
        ),
        out,
    )

    report = json.loads((out / "margins.json").read_text())
    n_held_out = report["n_held_out"]
    assert n_held_out >= 1
    assert 0.0 <= report["overall_accuracy"] <= 1.0
    assert sum(b["n"] for b in report["buckets"]) == n_held_out
    for b in report["buckets"]:
        assert b["accuracy"] is None if b["n"] == 0 else 0.0 <= b["accuracy"] <= 1.0
    assert (out / "margins_report.md").exists()
    assert (out / "manifest.json").exists()


def test_margins_config_rejects_bad_edges():
    import pytest

    with pytest.raises(ValueError):
        MarginsConfig(preferences="p", reward_model_dir="r", tokenizer_dir="t", bucket_edges=[0.5, 0.3])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_margins_stage.py -v`
Expected: FAIL (`MarginsConfig` import error)

- [ ] **Step 3: Implement.** Add `MarginsConfig` (code above) to `config.py`; register `"margins": (MarginsConfig, "tinyfables.stages.margins")` in `stages/__init__.py` (import `MarginsConfig` there); create the stage:

```python
# src/tinyfables/stages/margins.py
"""RM-accuracy-by-margin diagnostic (issue 08, STEP 1).

Stratifies held-out preferences by aggregate-score margin and reports RM
accuracy per bucket. High accuracy on wide margins + ~chance on near-ties
means the label signal is real but tie-diluted (margin-filtering/relabeling
helps); flat across margins means the labels carry little usable signal (PPO
on this RM would optimize noise). Feeds the ADR-0005 fork decision."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import MarginsConfig
from tinyfables.reward_data import collate_preference_batch, load_preferences
from tinyfables.reward_model import load_reward_model
from tinyfables.stage import write_manifest


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _margin(rec: dict) -> float:
    return rec["aggregate_chosen"] - rec["aggregate_rejected"]


def _bucket_index(margin: float, edges: list[float]) -> int:
    idx = 0
    for i, edge in enumerate(edges):
        if margin >= edge:
            idx = i
    return idx


@torch.no_grad()
def run(cfg: MarginsConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    tokenizer_path = Path(cfg.tokenizer_dir) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = load_reward_model(cfg.reward_model_dir, device=device).eval()

    held_out = load_preferences(cfg.preferences, split="held_out")
    if not held_out:
        raise ValueError("margins diagnostic requires at least one held_out preference")
    raw = {
        rec["pair_id"]: rec
        for rec in (json.loads(line) for line in Path(cfg.preferences).read_text().splitlines() if line.strip())
        if rec["split"] == "held_out"
    }

    correct_by_pair: dict[str, bool] = {}
    for start in range(0, len(held_out), cfg.batch_size):
        rows = held_out[start : start + cfg.batch_size]
        batch = collate_preference_batch(rows, tokenizer, cfg.n_ctx, device)
        chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        for row, c, r in zip(rows, chosen.tolist(), rejected.tolist()):
            correct_by_pair[row.pair_id] = c > r

    edges = list(cfg.bucket_edges)
    buckets = [{"lo": edge, "hi": (edges[i + 1] if i + 1 < len(edges) else None), "n": 0, "n_correct": 0} for i, edge in enumerate(edges)]
    for pair_id, correct in sorted(correct_by_pair.items()):
        b = buckets[_bucket_index(_margin(raw[pair_id]), edges)]
        b["n"] += 1
        b["n_correct"] += int(correct)
    for b in buckets:
        b["accuracy"] = round(b["n_correct"] / b["n"], 6) if b["n"] else None

    overall = sum(correct_by_pair.values()) / len(correct_by_pair)
    report = {
        "overall_accuracy": round(overall, 6),
        "n_held_out": len(correct_by_pair),
        "bucket_edges": edges,
        "buckets": buckets,
    }
    (out_dir / "margins.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# RM accuracy by aggregate-score margin",
        "",
        f"Held-out preferences: {report['n_held_out']}; overall accuracy {report['overall_accuracy']:.3f}",
        "",
        "| margin | n | accuracy |",
        "|---|---|---|",
    ]
    for b in buckets:
        hi = f"{b['hi']:.1f}" if b["hi"] is not None else "inf"
        acc = f"{b['accuracy']:.3f}" if b["accuracy"] is not None else "-"
        lines.append(f"| [{b['lo']:.1f}, {hi}) | {b['n']} | {acc} |")
    (out_dir / "margins_report.md").write_text("\n".join(lines) + "\n")

    write_manifest(
        out_dir,
        "margins",
        cfg,
        [out_dir / "margins.json", out_dir / "margins_report.md"],
        inputs={
            "preferences.jsonl": Path(cfg.preferences),
            "tokenizer.json": tokenizer_path,
            "reward_head.pt": Path(cfg.reward_model_dir) / "reward_head.pt",
            "backbone_model.safetensors": Path(cfg.reward_model_dir) / "backbone" / "model.safetensors",
        },
    )
```

And `configs/margins_full.yaml`:

```yaml
preferences: runs/derive_base/preferences.jsonl
reward_model_dir: runs/rm_hub          # snapshot of congthanh991/tinyfables-13m-rm
tokenizer_dir: runs/tokenizer_full
n_ctx: 1024
batch_size: 16
bucket_edges: [0.1, 0.3, 0.6, 1.0]
device: auto
seed: 0
```

- [ ] **Step 4: Run to verify pass, plus config-registry regression**

Run: `.venv/bin/pytest tests/test_margins_stage.py tests/test_config.py tests/test_cli_dispatch.py -v`
Expected: PASS (if `test_full_configs.py` enumerates `configs/*_full.yaml`, check it accepts `margins_full.yaml` or add it to its table — read that test before finishing).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/config.py src/tinyfables/stages/__init__.py src/tinyfables/stages/margins.py configs/margins_full.yaml tests/test_margins_stage.py
git commit -m "feat(margins): RM-accuracy-by-margin diagnostic stage"
```

### Task 6: toy gate fixtures

**Files:**
- Create: `tests/fixtures/gate_pass.json`, `tests/fixtures/gate_nogo.json`

**Interfaces:**
- Produces: gate records that pass/fail `tinyfables.gates.assert_gate_passed` exactly (its structure checks are strict — see `src/tinyfables/gates.py`). Used by Tasks 7 and 9.

- [ ] **Step 1: Write the fixtures**

`tests/fixtures/gate_pass.json`:

```json
{
  "pass": true,
  "reasons": [],
  "warnings": [],
  "thresholds": {
    "rm_accuracy_gate": 0.65,
    "labeler_self_consistency_required": true,
    "labeler_self_consistency_gate": 0.85,
    "position_swap_review_threshold": 0.5
  },
  "inputs": {
    "audit": {
      "self_consistency_pass": true,
      "self_consistency": {"mean_agreement": 0.9, "n_pairs": 30, "n_unanimous": 27},
      "position_swap": {"flip_rate": 0.2, "n_flipped": 40, "n_pairs": 200},
      "position_swap_review_flag": false
    },
    "reward": {
      "held_out_accuracy": 0.71,
      "n_train": 4,
      "n_held_out": 2,
      "rm_gate_pass": true
    }
  }
}
```

`tests/fixtures/gate_nogo.json` (mirrors the real NO-GO record):

```json
{
  "pass": false,
  "reasons": [
    "labeler self-consistency below gate",
    "reward model held-out accuracy below gate"
  ],
  "warnings": [],
  "thresholds": {
    "rm_accuracy_gate": 0.65,
    "labeler_self_consistency_required": true,
    "labeler_self_consistency_gate": 0.85,
    "position_swap_review_threshold": 0.5
  },
  "inputs": {
    "audit": {
      "self_consistency_pass": false,
      "self_consistency": {"mean_agreement": 0.778, "n_pairs": 30, "n_unanimous": 21},
      "position_swap": {"flip_rate": 0.295, "n_flipped": 59, "n_pairs": 200},
      "position_swap_review_flag": false
    },
    "reward": {
      "held_out_accuracy": 0.590426,
      "n_train": 1761,
      "n_held_out": 188,
      "rm_gate_pass": false
    }
  }
}
```

- [ ] **Step 2: Verify both against the real gate checker**

```bash
.venv/bin/python - <<'EOF'
from tinyfables.gates import GateError, assert_gate_passed
assert_gate_passed("tests/fixtures/gate_pass.json")
try:
    assert_gate_passed("tests/fixtures/gate_nogo.json")
    raise SystemExit("nogo fixture unexpectedly passed")
except GateError as e:
    print("fixtures OK:", e)
EOF
```

Expected: `fixtures OK: alignment gate failed: labeler self-consistency below gate, reward model held-out accuracy below gate`

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/gate_pass.json tests/fixtures/gate_nogo.json
git commit -m "test(align): toy gate-pass/no-go fixtures"
```

### Task 7: `ppo` stage

**Files:**
- Modify: `src/tinyfables/config.py` (add `PPOStageConfig`), `src/tinyfables/stages/__init__.py` (register `ppo`)
- Create: `src/tinyfables/stages/ppo.py`, `configs/ppo_toy.yaml`, `configs/ppo_full.yaml`
- Test: `tests/test_ppo_stage.py`

**Interfaces:**
- Consumes: `assert_gate_passed(path)` (raises `GateError`), `load_hf_tokenizer(dir, padding_side)` (Task 1), `ScoredModelAdapter.from_reward_model_dir` (Task 3), `probe_lengths`/`length_drift` (Task 4), `load_preferences` (prompts), `GPT.from_pretrained`, `write_manifest`.
- Produces: stage `ppo`: given a **passing** gate record, trains the policy with TRL's experimental PPOTrainer and writes to `out_dir`: `config.json`+`model.safetensors`+`generation_config.json` (aligned policy, `save_pretrained` at the out_dir root, matching the base-model run layout so `push_model_to_hub` works on the folder), `ppo_curves.csv` (one row per logged iteration: `episode, objective_kl, objective_scores, objective_rlhf_reward, loss_policy_avg, loss_value_avg, probe_mean_words, probe_mean_new_tokens, length_drift_pct, length_alarm`), `ppo_summary.json`, `manifest.json` last. Raises `GateError` before touching torch/trl when the gate record is missing or failing.

```python
@dataclass(frozen=True)
class PPOStageConfig:
    gate: str                     # path to gate.json — MUST pass assert_gate_passed
    preferences: str              # prompts come from the train split; probes from held_out
    base_checkpoint: str
    reward_model_dir: str
    tokenizer_dir: str
    adr_decision: str             # e.g. "ADR-0005a" — echoed into summary + manifest config
    n_ctx: int = 1024
    response_length: int = 320
    total_episodes: int = 2000
    batch_size: int = 8           # per-device
    gradient_accumulation_steps: int = 2
    local_rollout_forward_batch_size: int = 8
    num_ppo_epochs: int = 4
    num_mini_batches: int = 1
    kl_coef: float = 0.2          # strong KL anchor (design: Feedback stage)
    lr: float = 3e-6
    temperature: float = 0.9      # matches pairgen sampling
    missing_eos_penalty: float | None = 1.0
    whiten_rewards: bool = False
    n_probe_prompts: int = 8
    length_alarm_threshold: float = 0.25
    seed: int = 0
    device: str = "auto"          # "cpu" forces use_cpu=True for toy tests
    fp16: bool = False

    def __post_init__(self) -> None:
        for name in ("n_ctx", "response_length", "total_episodes", "batch_size",
                     "gradient_accumulation_steps", "local_rollout_forward_batch_size",
                     "num_ppo_epochs", "num_mini_batches", "n_probe_prompts"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lr <= 0 or self.temperature <= 0:
            raise ValueError("lr and temperature must be positive")
        if not 0.0 < self.length_alarm_threshold:
            raise ValueError("length_alarm_threshold must be positive")
        if not self.adr_decision:
            raise ValueError("adr_decision is required on the Aligned Model manifest")
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ppo_stage.py
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("trl")

import torch

from tinyfables.config import PPOStageConfig, SourceSpec, TokenizerConfig
from tinyfables.gates import GateError
from tinyfables.model import GPT, GPTConfig
from tinyfables.reward_model import RewardModel, save_reward_model
from tinyfables.stages import ppo as ppo_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURES = Path(__file__).parent / "fixtures"


def _toy_world(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(FIXTURES / "tiny_corpus.jsonl")), vocab_size=512, seed=0),
        tok_dir,
    )
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    torch.manual_seed(0)
    base = tmp_path / "base"
    GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=1, n_head=2, d_model=32, n_ctx=128)).save_pretrained(base)
    torch.manual_seed(1)
    rm = RewardModel(GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=1, n_head=2, d_model=32, n_ctx=128)))
    save_reward_model(rm, tmp_path / "rm")
    return tok_dir, base, tmp_path / "rm"


def _cfg(tmp_path, tok_dir, base, rm_dir, gate):
    return PPOStageConfig(
        gate=str(gate),
        preferences=str(FIXTURES / "preferences_replay.jsonl"),
        base_checkpoint=str(base),
        reward_model_dir=str(rm_dir),
        tokenizer_dir=str(tok_dir),
        adr_decision="ADR-0005-test",
        n_ctx=128,
        response_length=12,
        total_episodes=4,
        batch_size=2,
        gradient_accumulation_steps=1,
        local_rollout_forward_batch_size=2,
        num_ppo_epochs=1,
        num_mini_batches=1,
        lr=1e-4,
        n_probe_prompts=2,
        seed=0,
        device="cpu",
    )


def test_ppo_refuses_without_gate_pass(tmp_path):
    tok_dir, base, rm_dir = _toy_world(tmp_path)
    with pytest.raises(GateError):
        ppo_stage.run(_cfg(tmp_path, tok_dir, base, rm_dir, FIXTURES / "gate_nogo.json"), tmp_path / "out")
    with pytest.raises((GateError, FileNotFoundError)):
        ppo_stage.run(_cfg(tmp_path, tok_dir, base, rm_dir, tmp_path / "missing.json"), tmp_path / "out2")
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_ppo_toy_smoke_trains_and_writes_artifacts(tmp_path):
    tok_dir, base, rm_dir = _toy_world(tmp_path)
    out = tmp_path / "aligned"
    ppo_stage.run(_cfg(tmp_path, tok_dir, base, rm_dir, FIXTURES / "gate_pass.json"), out)

    assert (out / "model.safetensors").exists()
    summary = json.loads((out / "ppo_summary.json").read_text())
    assert summary["adr_decision"] == "ADR-0005-test"
    assert summary["gate"] == "pass"
    assert "preferences_sha" in summary and "base_checkpoint_sha" in summary
    assert isinstance(summary["length_alarm_triggered"], bool)

    rows = list(csv.DictReader((out / "ppo_curves.csv").open()))
    assert rows, "at least one logged iteration"
    for col in ("episode", "objective_kl", "objective_scores", "probe_mean_words", "length_alarm"):
        assert col in rows[0]

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "ppo"
    assert manifest["config"]["adr_decision"] == "ADR-0005-test"
    for key in ("preferences.jsonl", "gate.json", "base_model.safetensors", "tokenizer.json", "reward_head.pt"):
        assert key in manifest["inputs"]

    # the aligned model reloads and generates
    policy = GPT.from_pretrained(out)
    assert policy.config.n_ctx == 128
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_ppo_stage.py -v`
Expected: FAIL (`PPOStageConfig` import error)

- [ ] **Step 3: Implement.** Add `PPOStageConfig` to `config.py`, register `"ppo": (PPOStageConfig, "tinyfables.stages.ppo")`, then the stage:

```python
# src/tinyfables/stages/ppo.py
"""PPO alignment stage (issue 08): TRL experimental PPOTrainer, KL-anchored to
the frozen Base Model, gate-guarded (refuses to start without a gate-pass
record — tinyfables.gates keeps "do not optimize a noisy signal" executable),
with a per-iteration length-drift alarm and reward/KL curves.

TRL 1.8.0: PPOTrainer lives ONLY in trl.experimental.ppo. It calls every model
as model(input_ids, attention_mask, position_ids, return_dict=True,
output_hidden_states=True) over LEFT-padded batches, and generates through
lm_backbone.generate(...) — the GPT's mask/position/cache-proof support
(issue-08 model change) exists precisely for this."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from tinyfables.config import PPOStageConfig
from tinyfables.gates import assert_gate_passed
from tinyfables.stage import sha256_file, write_manifest


def _resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(cfg: PPOStageConfig, out_dir: Path) -> None:
    # Gate FIRST: no torch/trl import, no out_dir writes, before a failing gate.
    gate = assert_gate_passed(cfg.gate)

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from transformers import TrainerCallback
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    from tinyfables.hf_tokenizer import load_hf_tokenizer
    from tinyfables.length_alarm import length_drift, probe_lengths
    from tinyfables.model import GPT
    from tinyfables.reward_data import load_preferences
    from tinyfables.trl_compat import ScoredModelAdapter

    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    raw_tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    hf_tok = load_hf_tokenizer(cfg.tokenizer_dir, padding_side="left")

    policy = GPT.from_pretrained(cfg.base_checkpoint).to(device)
    ref = GPT.from_pretrained(cfg.base_checkpoint).to(device)
    reward_model = ScoredModelAdapter.from_reward_model_dir(cfg.reward_model_dir, device=device)
    value_model = ScoredModelAdapter.from_reward_model_dir(cfg.reward_model_dir, device=device)

    train_prompts = sorted({ex.prompt for ex in load_preferences(cfg.preferences, split="train")})
    probe_prompts = sorted({ex.prompt for ex in load_preferences(cfg.preferences, split="held_out")})[: cfg.n_probe_prompts]
    if not train_prompts:
        raise ValueError("no train-split prompts in preferences")
    if not probe_prompts:
        probe_prompts = train_prompts[: cfg.n_probe_prompts]

    max_prompt = cfg.n_ctx - cfg.response_length
    def _tokenize(row):
        return {"input_ids": hf_tok(row["prompt"], add_special_tokens=False)["input_ids"][:max_prompt]}

    dataset = Dataset.from_list([{"prompt": p} for p in train_prompts]).map(_tokenize, remove_columns=["prompt"])

    baseline = probe_lengths(
        policy.eval(), raw_tok, probe_prompts,
        max_new_tokens=cfg.response_length, temperature=cfg.temperature, seed=cfg.seed, device=device,
    )

    curves_path = out_dir / "ppo_curves.csv"
    fieldnames = [
        "episode", "objective_kl", "objective_scores", "objective_rlhf_reward",
        "objective_non_score_reward", "loss_policy_avg", "loss_value_avg",
        "probe_mean_words", "probe_mean_new_tokens", "length_drift_pct", "length_alarm",
    ]
    curves_fh = curves_path.open("w", newline="")
    writer = csv.DictWriter(curves_fh, fieldnames=fieldnames)
    writer.writeheader()
    alarm_state = {"triggered": False, "final": None}

    class LengthAlarmCallback(TrainerCallback):
        """Per-iteration probe: PPO logs once per rollout batch (logging_steps=1),
        so on_log is the every-iteration hook; TRL logs no length metric itself."""

        def __init__(self, trainer_ref):
            self.trainer_ref = trainer_ref

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None or "objective/kl" not in logs:
                return
            pol = self.trainer_ref["trainer"].policy_model  # PPOTrainer keeps the raw policy here
            was_training = pol.training
            pol.eval()
            probe = probe_lengths(
                pol, raw_tok, probe_prompts,
                max_new_tokens=cfg.response_length, temperature=cfg.temperature,
                seed=cfg.seed, device=next(pol.parameters()).device,
            )
            if was_training:
                pol.train()
            drift = length_drift(baseline["mean_words"], probe["mean_words"], cfg.length_alarm_threshold)
            alarm_state["triggered"] = alarm_state["triggered"] or drift["alarm"]
            alarm_state["final"] = drift
            writer.writerow({
                "episode": logs.get("episode", ""),
                "objective_kl": logs.get("objective/kl", ""),
                "objective_scores": logs.get("objective/scores", ""),
                "objective_rlhf_reward": logs.get("objective/rlhf_reward", ""),
                "objective_non_score_reward": logs.get("objective/non_score_reward", ""),
                "loss_policy_avg": logs.get("loss/policy_avg", ""),
                "loss_value_avg": logs.get("loss/value_avg", ""),
                "probe_mean_words": probe["mean_words"],
                "probe_mean_new_tokens": probe["mean_new_tokens"],
                "length_drift_pct": drift["drift_pct"],
                "length_alarm": drift["alarm"],
            })
            curves_fh.flush()

    args = PPOConfig(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        total_episodes=cfg.total_episodes,
        num_ppo_epochs=cfg.num_ppo_epochs,
        num_mini_batches=cfg.num_mini_batches,
        local_rollout_forward_batch_size=cfg.local_rollout_forward_batch_size,
        response_length=cfg.response_length,
        stop_token="eos",
        temperature=cfg.temperature,
        missing_eos_penalty=cfg.missing_eos_penalty,
        kl_coef=cfg.kl_coef,
        whiten_rewards=cfg.whiten_rewards,
        num_sample_generations=0,
        logging_steps=1,
        report_to=[],
        seed=cfg.seed,
        use_cpu=(device == "cpu"),
        fp16=cfg.fp16 and device == "cuda",
        save_strategy="no",
    )
    trainer_ref: dict = {}
    trainer = PPOTrainer(
        args=args,
        processing_class=hf_tok,
        model=policy,
        ref_model=ref,
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=value_model,
        callbacks=[LengthAlarmCallback(trainer_ref)],
    )
    trainer_ref["trainer"] = trainer
    trainer.train()
    curves_fh.close()

    aligned = trainer.policy_model
    aligned.save_pretrained(out_dir)

    summary = {
        "adr_decision": cfg.adr_decision,
        "gate": "pass",
        "gate_sha": sha256_file(Path(cfg.gate)),
        "preferences_sha": sha256_file(Path(cfg.preferences)),
        "base_checkpoint_sha": sha256_file(Path(cfg.base_checkpoint) / "model.safetensors"),
        "reward_model_sha": sha256_file(Path(cfg.reward_model_dir) / "reward_head.pt"),
        "total_episodes": cfg.total_episodes,
        "kl_coef": cfg.kl_coef,
        "baseline_probe": baseline,
        "final_length_drift": alarm_state["final"],
        "length_alarm_triggered": alarm_state["triggered"],
        "device": device,
    }
    (out_dir / "ppo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "ppo",
        cfg,
        [
            out_dir / "config.json",
            out_dir / "generation_config.json",
            out_dir / "model.safetensors",
            out_dir / "ppo_curves.csv",
            out_dir / "ppo_summary.json",
        ],
        inputs={
            "gate.json": Path(cfg.gate),
            "preferences.jsonl": Path(cfg.preferences),
            "base_model.safetensors": Path(cfg.base_checkpoint) / "model.safetensors",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
            "reward_head.pt": Path(cfg.reward_model_dir) / "reward_head.pt",
        },
    )
```

`configs/ppo_toy.yaml` (mirrors the test `_cfg`, pointing at `tests/fixtures/gate_pass.json` and `tests/fixtures/preferences_replay.jsonl`); `configs/ppo_full.yaml`:

```yaml
gate: runs/gate_base/gate.json          # honest record: NO-GO today — this config only runs if a future gate passes
preferences: runs/derive_base/preferences.jsonl
base_checkpoint: runs/base_model
reward_model_dir: runs/rm_hub
tokenizer_dir: runs/tokenizer_full
adr_decision: SET-ME-AT-FORK            # e.g. ADR-0005a
n_ctx: 1024
response_length: 320
total_episodes: 2000
batch_size: 8
gradient_accumulation_steps: 2
local_rollout_forward_batch_size: 8
num_ppo_epochs: 4
kl_coef: 0.2
lr: 3.0e-6
temperature: 0.9
missing_eos_penalty: 1.0
n_probe_prompts: 8
length_alarm_threshold: 0.25
seed: 0
device: auto
fp16: true
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_ppo_stage.py -v`
Expected: 2 PASS (the smoke test takes tens of seconds on CPU — TRL warns about experimental API; that's expected). Debugging notes if the smoke test fails inside TRL: (a) `generation_config` errors → the Task 2 `prepare_inputs_for_generation` override isn't being hit; (b) NaNs in rewards → the finfo.min pad-mask detail in Task 2; (c) dataloader/`input_ids` errors → dataset must contain ONLY the `input_ids` column; (d) `PPOConfig` rejects a kwarg → check `dataclasses.fields(PPOConfig)` against the installed 1.8.0 rather than guessing.

- [ ] **Step 5: Full-suite regression**

Run: `.venv/bin/pytest -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tinyfables/config.py src/tinyfables/stages/__init__.py src/tinyfables/stages/ppo.py configs/ppo_toy.yaml configs/ppo_full.yaml tests/test_ppo_stage.py
git commit -m "feat(ppo): gate-guarded TRL PPO stage with KL curves + length alarm"
```

### Task 8: `dpo` stage

**Files:**
- Modify: `src/tinyfables/config.py` (add `DPOStageConfig`), `src/tinyfables/stages/__init__.py` (register `dpo`)
- Create: `src/tinyfables/stages/dpo.py`, `configs/dpo_toy.yaml`, `configs/dpo_full.yaml`
- Test: `tests/test_dpo_stage.py`

**Interfaces:**
- Consumes: `load_hf_tokenizer` (Task 1), `probe_lengths`/`length_drift` (Task 4), `load_preferences`, `GPT.from_pretrained`, `write_manifest`. NO gate requirement — DPO is the pre-declared fallback and the gate record stays honestly NO-GO for PPO (ADR-0004; the ADR-0005 decision id is recorded instead).
- Produces: stage `dpo`: preferences in, Aligned Model out (`save_pretrained` at out_dir root), `dpo_curves.csv` (per logged step: `step, loss, rewards_margins if present`), `dpo_summary.json` (adr_decision, preferences_sha, base_checkpoint_sha, n_train_pairs, n_filtered_out, before/after probe + `length_alarm_triggered`), `manifest.json` last. `min_margin` filters training pairs by `aggregate_chosen - aggregate_rejected >= min_margin` (the cheap margin-filtered variant).

```python
@dataclass(frozen=True)
class DPOStageConfig:
    preferences: str
    base_checkpoint: str
    tokenizer_dir: str
    adr_decision: str
    n_ctx: int = 1024             # -> DPOConfig max_length
    beta: float = 0.1
    lr: float = 5e-6
    num_train_epochs: float = 1.0
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    min_margin: float = 0.0       # keep pairs with aggregate margin >= this
    logging_steps: int = 10
    n_probe_prompts: int = 8
    probe_max_new_tokens: int = 320
    temperature: float = 0.9
    length_alarm_threshold: float = 0.25
    seed: int = 0
    device: str = "auto"
    fp16: bool = False

    def __post_init__(self) -> None:
        for name in ("n_ctx", "batch_size", "gradient_accumulation_steps",
                     "logging_steps", "n_probe_prompts", "probe_max_new_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.beta <= 0 or self.lr <= 0 or self.num_train_epochs <= 0 or self.temperature <= 0:
            raise ValueError("beta, lr, num_train_epochs, temperature must be positive")
        if self.min_margin < 0:
            raise ValueError("min_margin must be non-negative")
        if not self.adr_decision:
            raise ValueError("adr_decision is required on the Aligned Model manifest")
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dpo_stage.py
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("trl")

import torch

from tinyfables.config import DPOStageConfig, SourceSpec, TokenizerConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.stages import dpo as dpo_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURES = Path(__file__).parent / "fixtures"


def _toy_world(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(FIXTURES / "tiny_corpus.jsonl")), vocab_size=512, seed=0),
        tok_dir,
    )
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    torch.manual_seed(0)
    base = tmp_path / "base"
    GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=1, n_head=2, d_model=32, n_ctx=128)).save_pretrained(base)
    return tok_dir, base


def _cfg(tok_dir, base, **overrides):
    kwargs = dict(
        preferences=str(FIXTURES / "preferences_replay.jsonl"),
        base_checkpoint=str(base),
        tokenizer_dir=str(tok_dir),
        adr_decision="ADR-0005-test",
        n_ctx=128,
        lr=1e-4,
        num_train_epochs=1.0,
        batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        n_probe_prompts=2,
        probe_max_new_tokens=12,
        seed=0,
        device="cpu",
    )
    kwargs.update(overrides)
    return DPOStageConfig(**kwargs)


def test_dpo_toy_smoke_trains_and_writes_artifacts(tmp_path):
    tok_dir, base = _toy_world(tmp_path)
    out = tmp_path / "aligned"
    dpo_stage.run(_cfg(tok_dir, base), out)

    assert (out / "model.safetensors").exists()
    summary = json.loads((out / "dpo_summary.json").read_text())
    assert summary["adr_decision"] == "ADR-0005-test"
    assert summary["n_train_pairs"] >= 1
    assert isinstance(summary["length_alarm_triggered"], bool)
    assert "preferences_sha" in summary and "base_checkpoint_sha" in summary
    assert list(csv.DictReader((out / "dpo_curves.csv").open()))

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "dpo"
    assert manifest["config"]["adr_decision"] == "ADR-0005-test"
    for key in ("preferences.jsonl", "base_model.safetensors", "tokenizer.json"):
        assert key in manifest["inputs"]

    # aligned weights differ from the base (training happened)
    aligned = GPT.from_pretrained(out)
    base_model = GPT.from_pretrained(base)
    assert not torch.equal(aligned.tok.weight, base_model.tok.weight)


def test_dpo_min_margin_filters_pairs(tmp_path):
    tok_dir, base = _toy_world(tmp_path)
    out = tmp_path / "aligned_filtered"
    dpo_stage.run(_cfg(tok_dir, base, min_margin=1.5), out)
    summary = json.loads((out / "dpo_summary.json").read_text())
    assert summary["min_margin"] == 1.5
    assert summary["n_filtered_out"] >= 1
    assert summary["n_train_pairs"] + summary["n_filtered_out"] == summary["n_train_pairs_available"]


def test_dpo_all_pairs_filtered_is_loud(tmp_path):
    tok_dir, base = _toy_world(tmp_path)
    with pytest.raises(ValueError, match="min_margin"):
        dpo_stage.run(_cfg(tok_dir, base, min_margin=99.0), tmp_path / "out")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_dpo_stage.py -v`
Expected: FAIL (`DPOStageConfig` import error)

- [ ] **Step 3: Implement.** Add `DPOStageConfig` to `config.py`, register `"dpo": (DPOStageConfig, "tinyfables.stages.dpo")`, then:

```python
# src/tinyfables/stages/dpo.py
"""DPO alignment stage (issue 08): the pre-declared fallback (ADR-0004 /
design.md Feedback stage). Consumes the same derived-preferences artifact as
the reward model, optimizes the policy directly (no reward model at
optimization time — nothing to reward-hack), and applies the same
length-drift alarm via before/after probes. `min_margin` implements the cheap
margin-filtered variant."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tinyfables.config import DPOStageConfig
from tinyfables.stage import sha256_file, write_manifest


def _resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(cfg: DPOStageConfig, out_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from trl import DPOConfig, DPOTrainer

    from tinyfables.hf_tokenizer import load_hf_tokenizer
    from tinyfables.length_alarm import length_drift, probe_lengths
    from tinyfables.model import GPT

    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    rows = [
        json.loads(line)
        for line in Path(cfg.preferences).read_text().splitlines()
        if line.strip()
    ]
    train_rows = [r for r in rows if r["split"] == "train"]
    kept = [
        r for r in train_rows
        if (r["aggregate_chosen"] - r["aggregate_rejected"]) >= cfg.min_margin
    ]
    if not kept:
        raise ValueError(
            f"min_margin={cfg.min_margin} filtered out all {len(train_rows)} train preferences"
        )
    probe_prompts = sorted({r["prompt"] for r in rows if r["split"] == "held_out"})[: cfg.n_probe_prompts]
    if not probe_prompts:
        probe_prompts = sorted({r["prompt"] for r in kept})[: cfg.n_probe_prompts]

    dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in kept]
    )

    raw_tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    hf_tok = load_hf_tokenizer(cfg.tokenizer_dir)
    policy = GPT.from_pretrained(cfg.base_checkpoint).to(device)
    ref = GPT.from_pretrained(cfg.base_checkpoint).to(device)

    before = probe_lengths(
        policy.eval(), raw_tok, probe_prompts,
        max_new_tokens=cfg.probe_max_new_tokens, temperature=cfg.temperature,
        seed=cfg.seed, device=device,
    )

    args = DPOConfig(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.num_train_epochs,
        beta=cfg.beta,
        max_length=cfg.n_ctx,
        logging_steps=cfg.logging_steps,
        report_to=[],
        seed=cfg.seed,
        use_cpu=(device == "cpu"),
        fp16=cfg.fp16 and device == "cuda",
        save_strategy="no",
    )
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=args,
        train_dataset=dataset,
        processing_class=hf_tok,
    )
    trainer.train()

    curves_path = out_dir / "dpo_curves.csv"
    logged = [h for h in trainer.state.log_history if "loss" in h]
    with curves_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "loss", "rewards_margins"])
        writer.writeheader()
        for h in logged:
            writer.writerow({
                "step": h.get("step", ""),
                "loss": h.get("loss", ""),
                "rewards_margins": h.get("rewards/margins", ""),
            })

    aligned = trainer.model
    aligned.save_pretrained(out_dir)

    after = probe_lengths(
        aligned.eval(), raw_tok, probe_prompts,
        max_new_tokens=cfg.probe_max_new_tokens, temperature=cfg.temperature,
        seed=cfg.seed, device=device,
    )
    drift = length_drift(before["mean_words"], after["mean_words"], cfg.length_alarm_threshold)

    summary = {
        "adr_decision": cfg.adr_decision,
        "preferences_sha": sha256_file(Path(cfg.preferences)),
        "base_checkpoint_sha": sha256_file(Path(cfg.base_checkpoint) / "model.safetensors"),
        "min_margin": cfg.min_margin,
        "n_train_pairs_available": len(train_rows),
        "n_train_pairs": len(kept),
        "n_filtered_out": len(train_rows) - len(kept),
        "beta": cfg.beta,
        "final_loss": logged[-1]["loss"] if logged else None,
        "probe_before": before,
        "probe_after": after,
        "length_drift": drift,
        "length_alarm_triggered": drift["alarm"],
        "device": device,
    }
    (out_dir / "dpo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "dpo",
        cfg,
        [
            out_dir / "config.json",
            out_dir / "generation_config.json",
            out_dir / "model.safetensors",
            out_dir / "dpo_curves.csv",
            out_dir / "dpo_summary.json",
        ],
        inputs={
            "preferences.jsonl": Path(cfg.preferences),
            "base_model.safetensors": Path(cfg.base_checkpoint) / "model.safetensors",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
```

`configs/dpo_toy.yaml` mirrors the test `_cfg`; `configs/dpo_full.yaml`:

```yaml
preferences: runs/derive_base/preferences.jsonl
base_checkpoint: runs/base_model
tokenizer_dir: runs/tokenizer_full
adr_decision: SET-ME-AT-FORK        # e.g. ADR-0005b
n_ctx: 1024
beta: 0.1
lr: 5.0e-6
num_train_epochs: 3.0
batch_size: 8
gradient_accumulation_steps: 2
min_margin: 0.0                     # margin-filtered variant: bump and rerun
logging_steps: 10
n_probe_prompts: 8
probe_max_new_tokens: 320
temperature: 0.9
length_alarm_threshold: 0.25
seed: 0
device: auto
fp16: true
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_dpo_stage.py -v`
Expected: 3 PASS. If DPOTrainer errors on tokenization, remember it calls `processing_class(text)` per-row and appends `eos_token_id` itself; if it errors on a missing config attr, our `GPTConfig` inherits `PretrainedConfig` defaults (`is_encoder_decoder=False` etc.) — inspect the actual attr before patching.

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/config.py src/tinyfables/stages/__init__.py src/tinyfables/stages/dpo.py configs/dpo_toy.yaml configs/dpo_full.yaml tests/test_dpo_stage.py
git commit -m "feat(dpo): fallback DPO stage on derived preferences with margin filter + length alarm"
```

### Task 9: `samples` stage (before/after generations) + toy-chain extension

**Files:**
- Modify: `src/tinyfables/config.py` (add `SamplesConfig`), `src/tinyfables/stages/__init__.py` (register `samples`), `tests/test_toy_chain.py`
- Create: `src/tinyfables/stages/samples.py`, `configs/samples_full.yaml`
- Test: `tests/test_samples_stage.py`

**Interfaces:**
- Consumes: `iter_rows(source, seed)` + `parse_canonical_prompt`/`render_canonical_prompt`/`FableSpec` (the pairgen pattern — see `stages/pairgen.py:34-58`), `generate_fable`, `write_manifest`.
- Produces: stage `samples`: fixed FableSpecs → one greedy-free sampled generation per model per spec; `samples.jsonl` (`{"spec_id", "spec", "prompt", "base", "aligned", "base_words", "aligned_words"}`), `samples_report.md` (side-by-side), `samples_summary.json` (mean lengths both sides), manifest inputs = both `model.safetensors` + tokenizer.

```python
@dataclass(frozen=True)
class SamplesConfig:
    base_checkpoint: str
    aligned_checkpoint: str
    tokenizer_dir: str
    source: SourceSpec
    n_specs: int = 20
    max_new_tokens: int = 320
    min_new_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 50
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.n_specs <= 0:
            raise ValueError("n_specs must be positive")
```

Full implementation (follows `stages/pairgen.py`'s spec-collection pattern):

```python
# src/tinyfables/stages/samples.py
"""Before/after sample set (issue 08): the same fixed FableSpecs generated by
the Base Model and the Aligned Model, side by side — the RLHF effect made
visible, and the acceptance-criterion artifact for issue 08. Same seed per
spec on both sides: same sampling noise, different weights."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch
from tokenizers import Tokenizer

from tinyfables.config import SamplesConfig
from tinyfables.data import iter_rows
from tinyfables.generate import generate_fable
from tinyfables.model import GPT
from tinyfables.paraphrases import parse_canonical_prompt
from tinyfables.prompts import FableSpec, render_canonical_prompt
from tinyfables.stage import write_manifest

_ELEMENTS = ("character", "setting", "challenge", "outcome", "moral")


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _spec_from_prompt(prompt: str) -> FableSpec | None:
    parsed = parse_canonical_prompt(prompt)
    if parsed is None:
        return None
    return FableSpec(
        **{field: getattr(parsed, field) for field in _ELEMENTS},
        age_range=parsed.age_range,
        word_count=parsed.word_count,
    )


def run(cfg: SamplesConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs: list[FableSpec] = []
    for row in itertools.islice(iter_rows(cfg.source, cfg.seed), cfg.n_specs * 4):
        spec = _spec_from_prompt(row["prompt"])
        if spec is not None:
            specs.append(spec)
        if len(specs) >= cfg.n_specs:
            break
    if len(specs) < cfg.n_specs:
        raise ValueError(
            f"samples requested {cfg.n_specs} specs but only found {len(specs)} canonical prompts in {cfg.source}"
        )

    device = _resolve_device(cfg.device)
    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    base = GPT.from_pretrained(cfg.base_checkpoint).to(device).eval()
    aligned = GPT.from_pretrained(cfg.aligned_checkpoint).to(device).eval()

    def _gen(model, prompt_text: str, seed: int) -> str:
        return generate_fable(
            model,
            tok,
            prompt_text,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            seed=seed,
            device=device,
        )

    rows = []
    for i, spec in enumerate(specs):
        prompt = render_canonical_prompt(spec)
        base_fable = _gen(base, prompt, cfg.seed + i)
        aligned_fable = _gen(aligned, prompt, cfg.seed + i)
        rows.append(
            {
                "spec_id": f"spec-{i:03d}",
                "spec": asdict(spec),
                "prompt": prompt,
                "base": base_fable,
                "aligned": aligned_fable,
                "base_words": len(base_fable.split()),
                "aligned_words": len(aligned_fable.split()),
            }
        )

    samples_path = out_dir / "samples.jsonl"
    samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report_lines = ["# Before/after generations (Base vs Aligned)", ""]
    for row in rows:
        report_lines += [
            f"## {row['spec_id']}",
            "",
            f"- character: {row['spec']['character']} · setting: {row['spec']['setting']}",
            f"- challenge: {row['spec']['challenge']} · outcome: {row['spec']['outcome']}",
            f"- moral: {row['spec']['moral']}",
            "",
            f"**Base** ({row['base_words']} words):",
            "",
            "```",
            row["base"],
            "```",
            "",
            f"**Aligned** ({row['aligned_words']} words):",
            "",
            "```",
            row["aligned"],
            "```",
            "",
        ]
    report_path = out_dir / "samples_report.md"
    report_path.write_text("\n".join(report_lines))

    summary = {
        "n_specs": len(rows),
        "base_mean_words": mean(r["base_words"] for r in rows),
        "aligned_mean_words": mean(r["aligned_words"] for r in rows),
        "temperature": cfg.temperature,
        "seed": cfg.seed,
    }
    summary_path = out_dir / "samples_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "samples",
        cfg,
        [samples_path, report_path, summary_path],
        inputs={
            "base_model.safetensors": Path(cfg.base_checkpoint) / "model.safetensors",
            "aligned_model.safetensors": Path(cfg.aligned_checkpoint) / "model.safetensors",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_samples_stage.py
import json
from pathlib import Path

import torch

from tinyfables.config import SamplesConfig, SourceSpec, TokenizerConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.stages import samples as samples_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def test_samples_stage_writes_side_by_side(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(TokenizerConfig(source=SourceSpec(jsonl_path=FIXTURE), vocab_size=512, seed=0), tok_dir)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    for seed, name in ((0, "base"), (1, "aligned")):
        torch.manual_seed(seed)
        GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=1, n_head=2, d_model=32, n_ctx=512)).save_pretrained(tmp_path / name)

    out = tmp_path / "samples"
    samples_stage.run(
        SamplesConfig(
            base_checkpoint=str(tmp_path / "base"),
            aligned_checkpoint=str(tmp_path / "aligned"),
            tokenizer_dir=str(tok_dir),
            source=SourceSpec(jsonl_path=FIXTURE),
            n_specs=3,
            max_new_tokens=16,
            min_new_tokens=4,
            seed=0,
            device="cpu",
        ),
        out,
    )
    rows = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    for row in rows:
        assert row["base"] and row["aligned"]
        assert row["prompt"].startswith("Create a fable")
    assert "Base" in (out / "samples_report.md").read_text()
    assert (out / "manifest.json").exists()
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_samples_stage.py -v` → FAIL (import error).

- [ ] **Step 3: Implement** `SamplesConfig` + `stages/samples.py` per the sketch, register `"samples": (SamplesConfig, "tinyfables.stages.samples")`, and `configs/samples_full.yaml`:

```yaml
base_checkpoint: runs/base_model
aligned_checkpoint: runs/aligned_model
tokenizer_dir: runs/tokenizer_full
source:
  hf_dataset: klusai/ds-tf1-en-3m
  hf_split: validation
n_specs: 20
max_new_tokens: 320
min_new_tokens: 80
temperature: 0.9
top_k: 50
seed: 0
device: auto
```

- [ ] **Step 4: Extend the toy chain** (AC-3: prep → … → PPO → eval). Append to `tests/test_toy_chain.py`:

```python
def _fake_runner(prompt, model):
    # copy of tests/test_feedback_chain.py's deterministic fake labeler
    # (tests/ is not a package, so no cross-test import)
    import hashlib
    import json

    from tinyfables.feedback import AXES

    ids = [ln.split("pair_id: ")[1].strip() for ln in prompt.splitlines() if "pair_id: " in ln]

    def rate(seedtext):
        h = int.from_bytes(hashlib.sha256(seedtext.encode("utf-8")).digest()[:8], "big")
        return {a: 1 + (h >> (3 * j)) % 5 for j, a in enumerate(AXES)}

    return json.dumps(
        {
            "labels": [
                {"pair_id": pid, "fable_a": rate(pid + "a"), "fable_b": rate(pid + "b")}
                for pid in ids
            ]
        }
    )


def test_toy_chain_extends_through_ppo_and_eval(tmp_path, capsys):
    import json

    import pytest

    pytest.importorskip("trl")

    from tinyfables.config import (
        EvalConfig, PairgenConfig, PPOStageConfig, DeriveConfig, LabelConfig, SourceSpec,
    )
    from tinyfables.stages import derive as derive_stage
    from tinyfables.stages import evaluate as evaluate_stage
    from tinyfables.stages import label as label_stage
    from tinyfables.stages import pairgen as pairgen_stage
    from tinyfables.stages import ppo as ppo_stage

    runs = tmp_path / "runs"
    tok_dir, prep_dir, ckpt = runs / "tok", runs / "prep", runs / "pretrain"
    # reuse the toy front half exactly as test_toy_chain_tokenizer_prep_pretrain_generate builds it
    (tmp_path / "tok.yaml").write_text(
        f"source:\n  jsonl_path: {FIXTURE}\nvocab_size: 512\nseed: 0\ncompare_gpt2: false\n"
    )
    (tmp_path / "prep.yaml").write_text(
        f"source:\n  jsonl_path: {FIXTURE}\ntokenizer_dir: {tok_dir}\nwindow: 512\nseed: 0\n"
    )
    (tmp_path / "pre.yaml").write_text(
        f"prep_dir: {prep_dir}\ntokenizer_dir: {tok_dir}\n"
        "n_layer: 2\nn_head: 2\nd_model: 64\nn_ctx: 512\n"
        "batch_size: 4\nsteps: 20\nwarmup_steps: 2\nlr: 0.001\nseed: 0\ndevice: cpu\n"
    )
    for stage, cfg_name, out in (("tokenizer", "tok.yaml", tok_dir), ("prep", "prep.yaml", prep_dir), ("pretrain", "pre.yaml", ckpt)):
        assert main(["run", stage, "--config", str(tmp_path / cfg_name), "--out", str(out)]) == 0

    src = SourceSpec(jsonl_path=FIXTURE)
    pairgen_stage.run(
        PairgenConfig(checkpoint=str(ckpt), tokenizer_dir=str(tok_dir), source=src, n_ctx=512,
                      n_pairs=4, max_new_tokens=16, min_new_tokens=4, top_k=10, seed=0, device="cpu"),
        runs / "pairs",
    )
    label_stage.run(
        LabelConfig(pairs=str(runs / "pairs" / "pairs.jsonl"), rubric=str(Path(__file__).parents[1] / "RUBRIC.md"),
                    labeler_prompt=str(Path(__file__).parents[1] / "configs" / "labeler_prompt.yaml"),
                    model="fake-model", batch_size=2, swap_fraction=0.5, calibration_size=2, seed=0),
        runs / "labels", runner=_fake_runner,
    )
    derive_stage.run(
        DeriveConfig(labels=str(runs / "labels" / "labels.jsonl"), pairs=str(runs / "pairs" / "pairs.jsonl"),
                     held_out_fraction=0.25),
        runs / "derive",
    )

    # toy RM trained on the derived toy preferences
    from tinyfables.config import RewardTrainConfig
    from tinyfables.stages import reward as reward_stage
    reward_stage.run(
        RewardTrainConfig(preferences=str(runs / "derive" / "preferences.jsonl"), base_checkpoint=str(ckpt),
                          tokenizer_dir=str(tok_dir), n_ctx=512, batch_size=2, steps=2, curve_steps=1,
                          lr=1e-3, warmup_steps=0, seed=0, device="cpu", amp=False, log_every=1, curve_sizes=[2]),
        runs / "rm",
    )

    ppo_stage.run(
        PPOStageConfig(
            gate=str(Path(__file__).parent / "fixtures" / "gate_pass.json"),
            preferences=str(runs / "derive" / "preferences.jsonl"),
            base_checkpoint=str(ckpt), reward_model_dir=str(runs / "rm"), tokenizer_dir=str(tok_dir),
            adr_decision="ADR-0005-toy", n_ctx=512, response_length=12, total_episodes=2, batch_size=2,
            gradient_accumulation_steps=1, local_rollout_forward_batch_size=2, num_ppo_epochs=1,
            num_mini_batches=1, lr=1e-4, n_probe_prompts=1, seed=0, device="cpu",
        ),
        runs / "aligned",
    )

    evaluate_stage.run(
        EvalConfig(checkpoint=str(runs / "aligned"), tokenizer_dir=str(tok_dir), source=src,
                   n_ctx=512, n_perplexity_rows=4, n_generations=2, max_new_tokens=16,
                   min_new_tokens=4, seed=0, device="cpu"),
        runs / "eval_aligned",
    )
    assert json.loads((runs / "eval_aligned" / "eval_metrics.json").read_text())
    assert (runs / "aligned" / "manifest.json").exists()
```

(Note: `derive` with a toy 4-pair set may yield 0 held-out rows — `held_out_fraction=0.25` plus the sha-bucket split makes it probabilistic; the ppo stage falls back to train prompts for probes, and the RM requires ≥1 held-out row. If the toy derive produces no held-out row, bump `n_pairs` to 6 and/or `held_out_fraction` to 0.5 until it does — the fixture is deterministic, so this is a one-time tuning, not flakiness. Check the actual split before finalizing; the reward stage will fail loudly if it's wrong.)

- [ ] **Step 5: Run everything**

Run: `.venv/bin/pytest tests/test_samples_stage.py tests/test_toy_chain.py -v && .venv/bin/pytest -q`
Expected: all PASS (the extended chain is the slowest test in the suite — CPU-tiny geometry keeps it well under a couple of minutes).

- [ ] **Step 6: Commit**

```bash
git add src/tinyfables/config.py src/tinyfables/stages/__init__.py src/tinyfables/stages/samples.py configs/samples_full.yaml tests/test_samples_stage.py tests/test_toy_chain.py
git commit -m "feat(samples): before/after generation stage + toy chain through PPO and eval"
```

### Task 10: design.md Track A notes + merge

**Files:**
- Modify: `docs/design.md` (new "Implementation (issue 08, Track A)" subsection after the issue-07 one)

- [ ] **Step 1: Write the design note.** Add under the issue-07 implementation section:

```markdown
### Implementation (issue 08, Track A)

- **Both alignment stages land as code** (the fork decision — ADR-0005 — picks which one
  ships the Aligned Model; the other stays report material). `ppo` wraps TRL 1.8.0's
  *experimental* PPOTrainer (`trl.experimental.ppo` — PPO left TRL's stable namespace in
  1.0) and **refuses to start** without a gate-pass record via
  `gates.assert_gate_passed`, before importing torch/trl or writing anything. `dpo`
  consumes the same `preferences.jsonl`, needs no reward model at optimization time, and
  supports a `min_margin` filter (the cheap margin-filtered variant).
- **GPT grew TRL-compat**: attention_mask-aware attention (finfo.min key-padding mask, so
  fully-padded query rows don't NaN), position_ids derived from the mask (left-padded
  batches), `output_hidden_states`, and a cache-proof `prepare_inputs_for_generation`
  (the model has no KV cache; TRL's GenerationConfig defaults `use_cache=True`). The
  no-mask path is bit-identical to before — resume guarantees hold.
- **TRL's reward/value contract** (`base_model_prefix` + `.score` + backbone hidden
  states) is satisfied by `trl_compat.ScoredModelAdapter`, loaded from the issue-07 RM
  dir; value model = a second adapter instance (standard PPO value init from the RM).
- **Length-drift alarm** (`length_alarm.py`): fixed held-out probe prompts, mean
  generated words vs the frozen Base baseline, alarm at >25% drift. PPO probes every
  logged iteration (TRL logs no length metric); DPO probes before/after training.
  Reward/KL curves land in `ppo_curves.csv`; the alarm state in each stage summary.
- **`margins` stage** (the fork diagnostic): RM accuracy over held-out preferences
  stratified by aggregate-score margin. **`samples` stage**: before/after generations
  for fixed FableSpecs (the AC artifact). Manifest contract: aligned-model manifests
  record preferences hash, Base checkpoint sha, and the `adr_decision` id.
- TRL is the lazy `align` extra (`pip install -e '.[dev,align]'`); tests
  `importorskip("trl")` so the base suite stays green without it.
```

- [ ] **Step 2: Full suite + light-import check one last time**

Run: `.venv/bin/pytest -q && .venv/bin/python -c "import sys, tinyfables.cli; assert 'torch' not in sys.modules and 'trl' not in sys.modules; print('light')"`
Expected: all PASS, `light`.

- [ ] **Step 3: Commit, then merge to main (main session does the merge)**

```bash
git add docs/design.md
git commit -m "docs(design): issue-08 Track A implementation notes"
```

Merge: `git checkout main && git merge --no-ff issue-08-aligned-model && .venv/bin/pytest -q` — use the superpowers:finishing-a-development-branch skill.

---

## CHECKPOINT A — run the real margin diagnostic (main session, minutes, local)

- [ ] **Step A.1: Pull the RM from the Hub and run `margins`**

```bash
cd /Users/thanh/code/tinystories && .venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("congthanh991/tinyfables-13m-rm", local_dir="runs/rm_hub"))
EOF
.venv/bin/python -m tinyfables run margins --config configs/margins_full.yaml --out runs/margins_base
cat runs/margins_base/margins_report.md
```

Expected: `overall_accuracy` ≈ 0.590 (sanity: matches the issue-07 record 0.590426 — if it doesn't, the RM/preferences pairing is wrong; STOP). The bucket table is the fork input.

- [ ] **Step A.2: Mirror the diagnostic to the Hub**

```bash
cd /Users/thanh/code/tinystories && .venv/bin/hf upload congthanh991/tinyfables-preferences runs/margins_base margins --repo-type dataset
```

## CHECKPOINT B — the fork (main session, AskUserQuestion) → ADR-0005

- [ ] **Step B.1: Present the fork with the measured bucket table** via AskUserQuestion. Options exactly as the issue framing:
  - **(a) Improve the signal**: majority-vote relabel (3 samples/pair, prompt v3) and/or margin-filter, retrain RM; if ≥0.65 → PPO as designed. Recommend only if the diagnostic shows high wide-margin accuracy + ~chance near-ties.
  - **(b) DPO-primary**: DPO directly on the 1,949 preferences (margin-filtered variant cheap to try); no reward model to hack; the gate record stays honestly NO-GO for PPO. Recommend if accuracy is flat across margins.
  - **(c) ADR-gated PPO anyway**: consciously accept noisy-reward risk. Weakest option (0.59 RM ≈ coin-flip); included for completeness. Requires a follow-up code task: an explicit `adr_override` field on `PPOStageConfig` that permits a failing gate record while stamping the override into summary + manifest (Track A deliberately did NOT build this — the gate stays strict unless an ADR says otherwise).

- [ ] **Step B.2: Write `docs/adr/0005-<slug>.md`** following the ADR-0004 format (decision statement, Consequences, Considered Options). It MUST include: the diagnostic bucket table verbatim, the chosen option + why the numbers support it, the rejected options, and the decision id used in configs (`ADR-0005a`/`ADR-0005b`/`ADR-0005c`). Set `adr_decision` in `configs/dpo_full.yaml` and/or `configs/ppo_full.yaml` accordingly.

- [ ] **Step B.3: Commit** `git add docs/adr/0005-*.md configs/ && git commit -m "docs(adr): ADR-0005 issue-08 fork decision"` and push `main`.

### Contingent task group (ONLY if fork = (a) Improve the signal)

1. **Prompt v3**: edit `configs/labeler_prompt.yaml` — bump `version: 3`, tighten the instruction against near-tie inflation (e.g. "If the two fables genuinely differ in quality, your ratings must reflect it; reserve identical ratings for genuinely indistinguishable pairs."). The version bump forces cache misses by design.
2. **Majority-vote derive**: add `DeriveConfig.extra_labels: list[str] | None = None`; in `stages/derive.py`, when set, compute `derive_preference` per cache and keep pairs where ≥2 of the 3 caches agree on the same non-None winner (else skip, counted as `n_no_majority`). TDD with a 3-cache fixture built from `labels_replay.jsonl` variants.
3. **Relabel on THIS Mac** (`claude -p`, subscription auth): run the `label` stage 3× with prompt v3 into `runs/labels_v3_s{0,1,2}` (fresh caches each — sampling variance across runs is the "3 samples/pair"), push all three caches to the Hub dataset repo under `labels_v3/`.
4. **Re-derive + re-audit + RM retrain + gate** (RM retrain on Colab per Operations): if new gate PASSES → Track B runs `ppo` with `gate: <new gate.json>`; if it still fails → fall back to (b) and note it in ADR-0005.

---

## Track B — Colab (main session via colab-mcp), after ADR-0005

Follow the colab-mcp quirks from memory: clone to `/content/repo` (not `/content/tinyfables`), open the browser connection with a live kernel, and re-open before every `run_code_cell`.

- [ ] **Step T.1: Session setup** (one cell):

```bash
cd /content && git clone https://github.com/harryct229/tinyfables.git repo && cd repo && pip install -e '.[align]' -q
```

Then `hf auth login` with the maintainer's token (or `HF_TOKEN` env), and restart-safe: everything below re-runs idempotently.

- [ ] **Step T.2: Pull inputs from the Hub** (one cell):

```python
from huggingface_hub import snapshot_download
snapshot_download("congthanh991/tinyfables-13m-base", local_dir="/content/repo/runs/base_model")
snapshot_download("congthanh991/tinyfables-tokenizer", local_dir="/content/repo/runs/tokenizer_full")
snapshot_download("congthanh991/tinyfables-13m-rm", local_dir="/content/repo/runs/rm_hub")
snapshot_download("congthanh991/tinyfables-preferences", repo_type="dataset", local_dir="/content/repo/runs/prefs")
```

Note the config paths: on Colab, `preferences: runs/prefs/derive/preferences.jsonl` and `gate: runs/prefs/gate/gate.json` — write a small Colab-local YAML override rather than editing the committed full configs (e.g. `configs/dpo_colab.yaml` generated in the notebook from `dpo_full.yaml` with the three paths swapped).

- [ ] **Step T.3: Run the chosen stage** (fork-dependent):
  - **(b) DPO-primary**: `python -m tinyfables run dpo --config configs/dpo_colab.yaml --out runs/aligned_model` (minutes-to-hours on T4 at 13M; ~1,761 train pairs × 3 epochs). If ADR-0005 chose the margin-filtered variant, set `min_margin` per the ADR. Watch `dpo_summary.json`'s `length_alarm_triggered`.
  - **(a) after gate PASS**: `python -m tinyfables run ppo --config configs/ppo_colab.yaml --out runs/aligned_model` — watch `ppo_curves.csv` (reward up, KL bounded, no length alarm).
  - **(c)**: only after the `adr_override` follow-up task exists and ADR-0005c is committed.

- [ ] **Step T.4: Before/after samples**: `python -m tinyfables run samples --config configs/samples_colab.yaml --out runs/samples_aligned` (aligned_checkpoint = `runs/aligned_model`, source = the HF val split as in `samples_full.yaml`).

- [ ] **Step T.5: Push EVERYTHING before the session ends** (the session-death rule):

```bash
cd /content/repo && \
python -m tinyfables push --kind model --path runs/aligned_model --repo congthanh991/tinyfables-13m-aligned && \
hf upload congthanh991/tinyfables-preferences runs/samples_aligned samples --repo-type dataset && \
hf upload congthanh991/tinyfables-13m-aligned runs/samples_aligned/samples_report.md samples_report.md
```

(`push_model_to_hub` uploads the whole stage folder — model + curves + summary + manifest — mirroring how the base model repo carries its run artifacts.)

- [ ] **Step T.6: Round-trip check** (new Colab cell or locally): `snapshot_download("congthanh991/tinyfables-13m-aligned")` and confirm `model.safetensors`, the stage summary, curves, and manifest are all present; `GPT.from_pretrained(<snapshot>)` loads.

## Closeout (main session)

- [ ] **Step Z.1: design.md Track B record** — append to the issue-08 implementation section: the margin-diagnostic bucket table, the ADR-0005 decision + id, the chosen run's measured numbers (train pairs used, epochs/episodes, final loss / reward + KL trajectory, probe lengths before/after + alarm state), the Hub repo `congthanh991/tinyfables-13m-aligned`, and the honest sentence about the gate (e.g. "the pipeline gated itself off PPO; the pre-declared fallback shipped the Aligned Model" if (b)).
- [ ] **Step Z.2: Tick issue 08** — check every satisfied acceptance-criterion box in `docs/issues/08-aligned-model-ppo.md` and flip 06/07/08's `Done` boxes in `docs/issues/README.md` to ☑ (06 and 07 shipped but were never ticked — verify their ACs first; tick only what's true).
- [ ] **Step Z.3: Commit + push** `git add docs/ && git commit -m "docs: issue-08 Track B record + ADR-0005 outcome" && git push`.

## Done means

ADR-0005 committed · `congthanh991/tinyfables-13m-aligned` on the Hub · before/after samples saved as an artifact · gate record honest (NO-GO untouched unless a real re-gate passed) · issue 08 ticked · design.md updated with all measured numbers.
