# Reward Model + Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the issue 07 reward-model stage, data-curve ablation, and explicit alignment gate that controls whether issue 08 PPO may run.

**Architecture:** Add a small reward-model wrapper around the existing `GPT` backbone: the Base Model produces hidden states, a scalar head scores the last non-pad token, and Bradley-Terry loss trains chosen > rejected from `preferences.jsonl`. The `reward` stage trains the full RM and fresh data-curve points; the `gate` stage combines issue 06 audit numbers with RM held-out accuracy into a machine-readable go/no-go artifact.

**Tech Stack:** Python 3.10+, PyTorch, transformers 5.x, tokenizers, pyyaml, stdlib JSON/CSV/pathlib; no new dependencies. Tests use pytest and toy CPU runs only. Real Track B run is intended for Colab T4 or local MPS/CUDA via `device: auto`.

## Global Constraints

- **Input preference artifact:** `runs/derive_base/preferences.jsonl` from issue 06 is the stable downstream input; it contains `chosen`, `rejected`, `prompt`, `split`, ratings, and aggregate scores.
- **Input audit artifact:** `runs/audit_base/audit.json` from issue 06 is required by the gate stage.
- **Known issue 06 audit result:** Calibration self-consistency was `0.778` (`21/30`) against a `0.85` gate; the issue 07 gate must preserve that failure instead of silently passing.
- **Reward model architecture:** Base Model plus scalar head, trained with Bradley-Terry loss over derived preferences.
- **Reward score pooling:** score the final non-pad token of `prompt + fable + <|endoftext|>`; pad with `<|pad|>`.
- **RM gate:** held-out reward-model accuracy must be `>= 0.65`.
- **Labeler gate:** issue 06 `self_consistency_pass` must be true for PPO go/no-go to pass; position-swap flip rate is recorded and warning-flagged but not a hard numeric fail unless the audit stage says it is.
- **Data curve:** emit report-ready accuracy points for requested train sizes `[100, 500, 1000, 2000]`; if a requested size exceeds available train preferences, train on all available preferences and keep both `requested_train_size` and `train_size` in the artifact.
- **Lazy stage registry:** `reward` imports torch; keep it behind `REGISTRY` module-path strings. Importing `tinyfables.stages` must not import torch-heavy reward code.
- **Tests never require network or Claude.** Hub push is tested with an injected fake API or the existing CLI dispatch pattern.
- **Manifest last:** `reward` and `gate` must write `manifest.json` after all other artifacts, using `stage.write_manifest`.
- **Existing command convention:** shell commands in this repo are run through `rtk`.

---

## Scope Check

Issue 07 is one coherent subsystem: reward-model training, data-curve measurement, gate emission, and Hub publication of the resulting RM artifact. It should stay in one plan because the gate cannot be meaningfully tested without the reward summary, and issue 08 depends on the gate schema.

---

## File Structure

**New modules:**
- `src/tinyfables/reward_data.py` — load derived preferences, tokenize `prompt + fable + EOT`, pad batches, and provide deterministic train-size subsets.
- `src/tinyfables/reward_model.py` — `RewardModel` wrapper, scalar head, Bradley-Terry loss, accuracy, save/load helpers.
- `src/tinyfables/stages/reward.py` — train full RM, train data-curve runs, write `reward_summary.json`, `data_curve.json`, `loss_log.csv`, model files, manifest.
- `src/tinyfables/stages/gate.py` — combine audit + reward summary into `gate.json` and `gate_report.md`.
- `src/tinyfables/gates.py` — future PPO-facing `assert_gate_passed(path)` helper.

**New configs:**
- `configs/reward_toy.yaml` — documented toy config for local smoke runs.
- `configs/reward_full.yaml` — real issue 07 run over `runs/derive_base/preferences.jsonl`.
- `configs/gate_full.yaml` — gate over `runs/audit_base/audit.json` + `runs/reward_base/reward_summary.json`.

**New tests/fixtures:**
- `tests/fixtures/preferences_replay.jsonl` — small derived-preference replay artifact.
- `tests/test_reward_data.py`
- `tests/test_reward_model.py`
- `tests/test_reward_stage.py`
- `tests/test_gate_stage.py`

**Modified:**
- `src/tinyfables/config.py` — add `RewardTrainConfig`, `GateConfig`.
- `src/tinyfables/stages/__init__.py` — register `reward`, `gate`.
- `tests/test_config.py`, `tests/test_full_configs.py`, `tests/test_cli_dispatch.py` — config and registry coverage.
- `docs/design.md` — record Track A implementation and Track B run numbers.
- `docs/issues/07-reward-model-gates.md` — tick acceptance criteria after Track B.

---

### Task 1: Reward Preference Data Loader

**Files:**
- Create: `src/tinyfables/reward_data.py`
- Create: `tests/test_reward_data.py`
- Create: `tests/fixtures/preferences_replay.jsonl`

**Interfaces:**
- Consumes: issue 06 preference rows with fields `pair_id`, `prompt`, `chosen`, `rejected`, and `split`.
- Produces:
  - `PreferenceExample`
  - `load_preferences(path: str | Path, *, split: str | None = None) -> list[PreferenceExample]`
  - `select_train_subset(examples: list[PreferenceExample], requested_size: int, seed: int) -> list[PreferenceExample]`
  - `encode_reward_text(tokenizer, prompt: str, fable: str, n_ctx: int) -> list[int]`
  - `collate_preference_batch(examples, tokenizer, n_ctx: int, device: str | torch.device) -> dict[str, torch.Tensor]`
  - Consumed by Tasks 2 and 4.

- [ ] **Step 1: Write the replay fixture**

Create `tests/fixtures/preferences_replay.jsonl`:

```jsonl
{"pair_id": "pair-000000", "prompt": "Create a fable about a shy octopus.", "chosen": "The octopus asked for help and learned courage.", "rejected": "The octopus hid and the story stopped.", "ratings_chosen": {"moral": 5, "adherence": 4, "coherence": 4, "prose": 4}, "ratings_rejected": {"moral": 2, "adherence": 2, "coherence": 2, "prose": 2}, "aggregate_chosen": 4.3, "aggregate_rejected": 2.0, "split": "train"}
{"pair_id": "pair-000001", "prompt": "Create a fable about a bold sparrow.", "chosen": "The sparrow shared seeds and made peace.", "rejected": "The sparrow shouted and forgot the lesson.", "ratings_chosen": {"moral": 4, "adherence": 4, "coherence": 4, "prose": 3}, "ratings_rejected": {"moral": 2, "adherence": 3, "coherence": 2, "prose": 2}, "aggregate_chosen": 3.9, "aggregate_rejected": 2.3, "split": "train"}
{"pair_id": "pair-000002", "prompt": "Create a fable about a patient mole.", "chosen": "The mole listened carefully and solved the tunnel problem.", "rejected": "The mole found a tunnel and then the words became unclear.", "ratings_chosen": {"moral": 4, "adherence": 5, "coherence": 4, "prose": 4}, "ratings_rejected": {"moral": 3, "adherence": 3, "coherence": 2, "prose": 2}, "aggregate_chosen": 4.3, "aggregate_rejected": 2.7, "split": "train"}
{"pair_id": "pair-000003", "prompt": "Create a fable about a kind bear.", "chosen": "The bear gave up a prize to help a friend.", "rejected": "The bear kept the prize and the moral contradicted the plot.", "ratings_chosen": {"moral": 5, "adherence": 4, "coherence": 5, "prose": 4}, "ratings_rejected": {"moral": 2, "adherence": 2, "coherence": 3, "prose": 3}, "aggregate_chosen": 4.6, "aggregate_rejected": 2.3, "split": "train"}
{"pair_id": "pair-000004", "prompt": "Create a fable about a quick hare.", "chosen": "The hare slowed down and helped the group finish together.", "rejected": "The hare raced away and the ending cut off.", "ratings_chosen": {"moral": 4, "adherence": 4, "coherence": 4, "prose": 4}, "ratings_rejected": {"moral": 2, "adherence": 2, "coherence": 2, "prose": 2}, "aggregate_chosen": 4.0, "aggregate_rejected": 2.0, "split": "held_out"}
{"pair_id": "pair-000005", "prompt": "Create a fable about a careful fox.", "chosen": "The fox told the truth before the lie grew larger.", "rejected": "The fox changed names twice and never resolved the conflict.", "ratings_chosen": {"moral": 5, "adherence": 5, "coherence": 4, "prose": 4}, "ratings_rejected": {"moral": 2, "adherence": 3, "coherence": 1, "prose": 2}, "aggregate_chosen": 4.7, "aggregate_rejected": 2.1, "split": "held_out"}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_reward_data.py`:

```python
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.constants import EOT, PAD
from tinyfables.reward_data import (
    collate_preference_batch,
    encode_reward_text,
    load_preferences,
    select_train_subset,
)
from tinyfables.stages import tokenizer as tokenizer_stage

FIX = Path(__file__).parent / "fixtures"
PREFS = FIX / "preferences_replay.jsonl"
CORPUS = FIX / "tiny_corpus.jsonl"


def test_load_preferences_filters_split_and_keeps_order():
    train = load_preferences(PREFS, split="train")
    held = load_preferences(PREFS, split="held_out")

    assert [row.pair_id for row in train] == [
        "pair-000000",
        "pair-000001",
        "pair-000002",
        "pair-000003",
    ]
    assert [row.pair_id for row in held] == ["pair-000004", "pair-000005"]
    assert train[0].chosen.startswith("The octopus")
    assert train[0].rejected.startswith("The octopus hid")


def test_select_train_subset_is_seeded_and_clamps_to_available():
    train = load_preferences(PREFS, split="train")
    a = select_train_subset(train, requested_size=3, seed=123)
    b = select_train_subset(train, requested_size=3, seed=123)
    c = select_train_subset(train, requested_size=99, seed=123)

    assert [x.pair_id for x in a] == [x.pair_id for x in b]
    assert len(a) == 3
    assert len(c) == 4
    assert {x.pair_id for x in c} == {x.pair_id for x in train}


def test_encode_reward_text_appends_eot_and_respects_n_ctx(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(CORPUS)), vocab_size=512, seed=0),
        tok_dir,
    )
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))

    ids = encode_reward_text(tok, "Prompt text. ", "Fable text.", n_ctx=64)

    assert ids[-1] == tok.token_to_id(EOT)
    assert len(ids) <= 64
    assert ids == tok.encode("Prompt text. ").ids + tok.encode("Fable text.").ids + [tok.token_to_id(EOT)]


def test_collate_preference_batch_pads_and_builds_attention_masks(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(CORPUS)), vocab_size=512, seed=0),
        tok_dir,
    )
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    examples = load_preferences(PREFS, split="train")[:2]

    batch = collate_preference_batch(examples, tok, n_ctx=64, device="cpu")

    assert set(batch) == {
        "chosen_input_ids",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
    }
    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["rejected_input_ids"].shape == batch["chosen_input_ids"].shape
    assert batch["chosen_attention_mask"].dtype == torch.long
    pad_id = tok.token_to_id(PAD)
    padded_positions = batch["chosen_attention_mask"] == 0
    assert torch.all(batch["chosen_input_ids"][padded_positions] == pad_id)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
rtk .venv/bin/pytest tests/test_reward_data.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.reward_data'`.

- [ ] **Step 4: Implement the data module**

Create `src/tinyfables/reward_data.py`:

```python
"""Reward-model preference data utilities.

The reward model scores prompt+fable sequences. Derived preference rows from issue
06 already contain chosen/rejected text and a deterministic train/held_out split.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path

import torch

from tinyfables.constants import EOT, PAD


@dataclass(frozen=True)
class PreferenceExample:
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    split: str


def load_preferences(path: str | Path, *, split: str | None = None) -> list[PreferenceExample]:
    rows: list[PreferenceExample] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if split is not None and rec["split"] != split:
            continue
        rows.append(
            PreferenceExample(
                pair_id=rec["pair_id"],
                prompt=rec["prompt"],
                chosen=rec["chosen"],
                rejected=rec["rejected"],
                split=rec["split"],
            )
        )
    return rows


def select_train_subset(
    examples: list[PreferenceExample], requested_size: int, seed: int
) -> list[PreferenceExample]:
    if requested_size <= 0:
        raise ValueError("requested_size must be positive")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[: min(requested_size, len(shuffled))]


def encode_reward_text(tokenizer, prompt: str, fable: str, n_ctx: int) -> list[int]:
    if n_ctx <= 0:
        raise ValueError("n_ctx must be positive")
    eot_id = tokenizer.token_to_id(EOT)
    if eot_id is None:
        raise ValueError("tokenizer lacks <|endoftext|>")
    ids = tokenizer.encode(prompt).ids + tokenizer.encode(fable).ids + [eot_id]
    return ids[:n_ctx]


def _pad(rows: list[list[int]], pad_id: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        ids = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[i, : len(row)] = ids
        attention_mask[i, : len(row)] = 1
    return input_ids, attention_mask


def collate_preference_batch(
    examples: list[PreferenceExample],
    tokenizer,
    n_ctx: int,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty preference batch")
    pad_id = tokenizer.token_to_id(PAD)
    if pad_id is None:
        raise ValueError("tokenizer lacks <|pad|>")
    chosen = [encode_reward_text(tokenizer, ex.prompt, ex.chosen, n_ctx) for ex in examples]
    rejected = [encode_reward_text(tokenizer, ex.prompt, ex.rejected, n_ctx) for ex in examples]
    chosen_ids, chosen_mask = _pad(chosen, pad_id, device)
    rejected_ids, rejected_mask = _pad(rejected, pad_id, device)
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_mask,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
rtk .venv/bin/pytest tests/test_reward_data.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add src/tinyfables/reward_data.py tests/test_reward_data.py tests/fixtures/preferences_replay.jsonl
rtk git commit -m "feat(reward): load derived preference batches"
```

---

### Task 2: Reward Model Wrapper + Bradley-Terry Math

**Files:**
- Create: `src/tinyfables/reward_model.py`
- Create: `tests/test_reward_model.py`

**Interfaces:**
- Consumes:
  - `tinyfables.model.GPT`
  - padded batches from `reward_data.collate_preference_batch`
- Produces:
  - `RewardModel(backbone: GPT)`
  - `RewardModel.from_base_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> RewardModel`
  - `RewardModel.forward(input_ids, attention_mask=None) -> torch.Tensor` returning shape `(batch,)`
  - `bradley_terry_loss(chosen_rewards, rejected_rewards) -> torch.Tensor`
  - `preference_accuracy(chosen_rewards, rejected_rewards) -> float`
  - `save_reward_model(model, out_dir)`
  - `load_reward_model(path, device="cpu") -> RewardModel`
  - Consumed by Task 4 and future issue 08 PPO integration.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reward_model.py`:

```python
from pathlib import Path

import torch

from tinyfables.model import GPT, GPTConfig
from tinyfables.reward_model import (
    RewardModel,
    bradley_terry_loss,
    load_reward_model,
    preference_accuracy,
    save_reward_model,
)


def tiny_backbone(vocab_size=32):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab_size, n_layer=1, n_head=2, d_model=16, n_ctx=32))


def test_reward_model_returns_one_scalar_per_sequence():
    rm = RewardModel(tiny_backbone()).eval()
    ids = torch.randint(0, 32, (3, 7))
    mask = torch.ones_like(ids)

    rewards = rm(ids, mask)

    assert rewards.shape == (3,)
    assert rewards.dtype == torch.float32


def test_reward_model_pools_last_non_pad_token():
    rm = RewardModel(tiny_backbone()).eval()
    ids = torch.randint(0, 32, (2, 8))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

    rewards = rm(ids, mask)

    assert rewards.shape == (2,)
    assert torch.isfinite(rewards).all()


def test_bradley_terry_loss_and_accuracy():
    chosen = torch.tensor([3.0, 2.0, 1.0])
    rejected = torch.tensor([1.0, 4.0, 1.0])

    loss = bradley_terry_loss(chosen, rejected)
    acc = preference_accuracy(chosen, rejected)

    assert loss.item() > 0
    assert acc == 1 / 3


def test_save_and_load_reward_model_round_trips(tmp_path):
    rm = RewardModel(tiny_backbone()).eval()
    ids = torch.randint(0, 32, (2, 6))
    mask = torch.ones_like(ids)
    before = rm(ids, mask).detach()

    save_reward_model(rm, tmp_path)
    loaded = load_reward_model(tmp_path, device="cpu").eval()
    after = loaded(ids, mask).detach()

    assert (tmp_path / "backbone" / "model.safetensors").exists()
    assert (tmp_path / "reward_head.pt").exists()
    assert (tmp_path / "reward_model_config.json").exists()
    assert torch.allclose(before, after)


def test_from_base_checkpoint_loads_backbone(tmp_path):
    base = tmp_path / "base"
    tiny_backbone().save_pretrained(base)

    rm = RewardModel.from_base_checkpoint(base, device="cpu")

    assert isinstance(rm, RewardModel)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk .venv/bin/pytest tests/test_reward_model.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.reward_model'`.

- [ ] **Step 3: Implement the reward model**

Create `src/tinyfables/reward_model.py`:

```python
"""Reward model: TinyFables GPT backbone plus scalar reward head."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyfables.model import GPT


class RewardModel(nn.Module):
    def __init__(self, backbone: GPT) -> None:
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(backbone.config.d_model, 1)

    @classmethod
    def from_base_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "RewardModel":
        backbone = GPT.from_pretrained(path).to(device)
        return cls(backbone).to(device)

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.backbone.config.n_ctx:
            raise ValueError(f"sequence length {T} exceeds n_ctx {self.backbone.config.n_ctx}")
        pos = torch.arange(T, device=input_ids.device)
        x = self.backbone.tok(input_ids) + self.backbone.pos(pos)[None, :, :]
        for block in self.backbone.blocks:
            x = block(x)
        return self.backbone.lnf(x)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.hidden_states(input_ids)
        if attention_mask is None:
            idx = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device)
        else:
            idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = h[batch, idx]
        return self.reward_head(pooled).squeeze(-1)


def bradley_terry_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def preference_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    return float((chosen_rewards > rejected_rewards).float().mean().item())


def save_reward_model(model: RewardModel, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out / "backbone")
    torch.save(model.reward_head.state_dict(), out / "reward_head.pt")
    cfg = {
        "backbone_model_type": model.backbone.config.model_type,
        "d_model": model.backbone.config.d_model,
        "pooling": "last_non_pad",
    }
    (out / "reward_model_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def load_reward_model(path: str | Path, device: str | torch.device = "cpu") -> RewardModel:
    root = Path(path)
    backbone = GPT.from_pretrained(root / "backbone").to(device)
    model = RewardModel(backbone).to(device)
    state = torch.load(root / "reward_head.pt", map_location=device)
    model.reward_head.load_state_dict(state)
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk .venv/bin/pytest tests/test_reward_model.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/tinyfables/reward_model.py tests/test_reward_model.py
rtk git commit -m "feat(reward): add scalar reward model"
```

---

### Task 3: Reward/Gate Configs and Lazy Stage Registration

**Files:**
- Modify: `src/tinyfables/config.py`
- Modify: `src/tinyfables/stages/__init__.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_full_configs.py`
- Modify: `tests/test_cli_dispatch.py`
- Create: `configs/reward_toy.yaml`
- Create: `configs/reward_full.yaml`
- Create: `configs/gate_full.yaml`

**Interfaces:**
- Consumes: existing `load_config`, `REGISTRY`, CLI stage dispatch.
- Produces:
  - `RewardTrainConfig`
  - `GateConfig`
  - `REGISTRY["reward"] == (RewardTrainConfig, "tinyfables.stages.reward")`
  - `REGISTRY["gate"] == (GateConfig, "tinyfables.stages.gate")`

- [ ] **Step 1: Write failing config and registry tests**

Append to `tests/test_config.py`:

```python
def test_reward_train_config_defaults_and_validation():
    from tinyfables.config import RewardTrainConfig

    cfg = RewardTrainConfig(
        preferences="runs/derive_base/preferences.jsonl",
        base_checkpoint="runs/base_model",
        tokenizer_dir="runs/tokenizer_full",
    )
    assert cfg.n_ctx == 1024
    assert cfg.batch_size == 16
    assert cfg.steps == 1000
    assert cfg.curve_sizes == [100, 500, 1000, 2000]
    assert cfg.curve_steps == 400
    assert cfg.accuracy_gate == 0.65
    assert cfg.device == "auto"
    assert cfg.amp is True


def test_reward_train_config_rejects_invalid_values():
    from tinyfables.config import RewardTrainConfig

    with pytest.raises(ValueError, match="batch_size"):
        RewardTrainConfig(preferences="p", base_checkpoint="b", tokenizer_dir="t", batch_size=0)
    with pytest.raises(ValueError, match="steps"):
        RewardTrainConfig(preferences="p", base_checkpoint="b", tokenizer_dir="t", steps=0)
    with pytest.raises(ValueError, match="accuracy_gate"):
        RewardTrainConfig(preferences="p", base_checkpoint="b", tokenizer_dir="t", accuracy_gate=1.5)
    with pytest.raises(ValueError, match="curve_sizes"):
        RewardTrainConfig(preferences="p", base_checkpoint="b", tokenizer_dir="t", curve_sizes=[])


def test_gate_config_defaults_and_validation():
    from tinyfables.config import GateConfig

    cfg = GateConfig(audit="runs/audit_base/audit.json", reward_summary="runs/reward_base/reward_summary.json")
    assert cfg.rm_accuracy_gate == 0.65
    assert cfg.require_labeler_self_consistency is True

    with pytest.raises(ValueError, match="rm_accuracy_gate"):
        GateConfig(audit="a", reward_summary="r", rm_accuracy_gate=-0.1)
```

Append to `tests/test_full_configs.py`:

```python
def test_reward_and_gate_full_configs_load():
    from tinyfables.config import GateConfig, RewardTrainConfig, load_config

    reward = load_config(REPO / "configs" / "reward_full.yaml", RewardTrainConfig)
    assert reward.preferences == "runs/derive_base/preferences.jsonl"
    assert reward.base_checkpoint == "runs/base_model"
    assert reward.tokenizer_dir == "runs/tokenizer_full"
    assert reward.n_ctx == 1024
    assert reward.batch_size == 16
    assert reward.steps == 1000
    assert reward.curve_sizes == [100, 500, 1000, 2000]
    assert reward.curve_steps == 400
    assert reward.accuracy_gate == 0.65
    assert reward.amp is True

    gate = load_config(REPO / "configs" / "gate_full.yaml", GateConfig)
    assert gate.audit == "runs/audit_base/audit.json"
    assert gate.reward_summary == "runs/reward_base/reward_summary.json"
    assert gate.rm_accuracy_gate == 0.65
    assert gate.require_labeler_self_consistency is True
```

Update the existing `test_cli_run_dispatches_feedback_stages_lazily` in `tests/test_cli_dispatch.py` so the loop includes `reward` and `gate`:

```python
    for stage in ("pairgen", "label", "derive", "audit", "reward", "gate"):
        out_dir = tmp_path / stage
        assert cli.main(["run", stage, "--config", str(cfg), "--out", str(out_dir)]) == 0
```

Update the expected `seen` list in the same test to:

```python
    assert seen == [
        ("load", "PairgenConfig", str(cfg)),
        ("run", "pairgen", "PairgenConfig", str(tmp_path / "pairgen")),
        ("load", "LabelConfig", str(cfg)),
        ("run", "label", "LabelConfig", str(tmp_path / "label")),
        ("load", "DeriveConfig", str(cfg)),
        ("run", "derive", "DeriveConfig", str(tmp_path / "derive")),
        ("load", "AuditConfig", str(cfg)),
        ("run", "audit", "AuditConfig", str(tmp_path / "audit")),
        ("load", "RewardTrainConfig", str(cfg)),
        ("run", "reward", "RewardTrainConfig", str(tmp_path / "reward")),
        ("load", "GateConfig", str(cfg)),
        ("run", "gate", "GateConfig", str(tmp_path / "gate")),
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk .venv/bin/pytest tests/test_config.py tests/test_full_configs.py tests/test_cli_dispatch.py -q
```

Expected: FAIL with `ImportError: cannot import name 'RewardTrainConfig'`.

- [ ] **Step 3: Add config dataclasses**

Modify `src/tinyfables/config.py` after `AuditConfig`:

```python
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
        if not 0.0 <= self.rm_accuracy_gate <= 1.0:
            raise ValueError("rm_accuracy_gate must be in [0, 1]")
```

- [ ] **Step 4: Register stages lazily**

Modify `src/tinyfables/stages/__init__.py` imports:

```python
from tinyfables.config import (
    AuditConfig,
    BenchmarkConfig,
    DeriveConfig,
    EvalConfig,
    GateConfig,
    LabelConfig,
    PairgenConfig,
    PrepConfig,
    PretrainConfig,
    RewardTrainConfig,
    TokenizerConfig,
)
```

Add registry entries:

```python
    "reward": (RewardTrainConfig, "tinyfables.stages.reward"),
    "gate": (GateConfig, "tinyfables.stages.gate"),
```

- [ ] **Step 5: Add YAML configs**

Create `configs/reward_toy.yaml`:

```yaml
preferences: tests/fixtures/preferences_replay.jsonl
base_checkpoint: runs/base_model_toy
tokenizer_dir: runs/tokenizer_toy
n_ctx: 128
batch_size: 2
steps: 8
curve_steps: 4
lr: 0.001
weight_decay: 0.0
warmup_steps: 0
grad_clip: 1.0
seed: 0
device: cpu
amp: false
log_every: 2
curve_sizes: [2, 4]
accuracy_gate: 0.65
```

Create `configs/reward_full.yaml`:

```yaml
preferences: runs/derive_base/preferences.jsonl
base_checkpoint: runs/base_model
tokenizer_dir: runs/tokenizer_full
n_ctx: 1024
batch_size: 16
steps: 1000
curve_steps: 400
lr: 0.00001
weight_decay: 0.0
warmup_steps: 50
grad_clip: 1.0
seed: 0
device: auto
amp: true
log_every: 20
curve_sizes: [100, 500, 1000, 2000]
accuracy_gate: 0.65
```

Create `configs/gate_full.yaml`:

```yaml
audit: runs/audit_base/audit.json
reward_summary: runs/reward_base/reward_summary.json
rm_accuracy_gate: 0.65
require_labeler_self_consistency: true
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
rtk .venv/bin/pytest tests/test_config.py tests/test_full_configs.py tests/test_cli_dispatch.py -q
```

Expected: PASS for the touched tests.

- [ ] **Step 7: Commit**

```bash
rtk git add src/tinyfables/config.py src/tinyfables/stages/__init__.py configs/reward_toy.yaml configs/reward_full.yaml configs/gate_full.yaml tests/test_config.py tests/test_full_configs.py tests/test_cli_dispatch.py
rtk git commit -m "feat(reward): configure reward and gate stages"
```

---

### Task 4: Reward Training Stage + Data Curve

**Files:**
- Create: `src/tinyfables/stages/reward.py`
- Create: `tests/test_reward_stage.py`

**Interfaces:**
- Consumes:
  - `RewardTrainConfig`
  - `load_preferences`, `select_train_subset`, `collate_preference_batch`
  - `RewardModel`, `bradley_terry_loss`, `preference_accuracy`, `save_reward_model`
- Produces:
  - CLI stage: `python -m tinyfables run reward --config ... --out ...`
  - Artifacts: `backbone/`, `reward_head.pt`, `reward_model_config.json`, `reward_summary.json`, `data_curve.json`, `loss_log.csv`, `manifest.json`
  - `reward_summary.json` fields consumed by Task 5: `held_out_accuracy`, `accuracy_gate`, `rm_gate_pass`, `n_train`, `n_held_out`.

- [ ] **Step 1: Write failing stage tests**

Create `tests/test_reward_stage.py`:

```python
import csv
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import RewardTrainConfig, SourceSpec, TokenizerConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.reward_model import load_reward_model
from tinyfables.stages import REGISTRY
from tinyfables.stages import reward as reward_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIX = Path(__file__).parent / "fixtures"
PREFS = FIX / "preferences_replay.jsonl"
CORPUS = FIX / "tiny_corpus.jsonl"


def make_toy_inputs(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(CORPUS)), vocab_size=512, seed=0),
        tok_dir,
    )
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    base = tmp_path / "base"
    torch.manual_seed(0)
    GPT(
        GPTConfig(
            vocab_size=tok.get_vocab_size(),
            n_layer=1,
            n_head=2,
            d_model=32,
            n_ctx=128,
        )
    ).save_pretrained(base)
    return tok_dir, base


def cfg(tmp_path, **overrides):
    tok_dir, base = make_toy_inputs(tmp_path)
    data = dict(
        preferences=str(PREFS),
        base_checkpoint=str(base),
        tokenizer_dir=str(tok_dir),
        n_ctx=128,
        batch_size=2,
        steps=8,
        curve_steps=4,
        lr=1e-3,
        weight_decay=0.0,
        warmup_steps=0,
        grad_clip=1.0,
        seed=0,
        device="cpu",
        amp=False,
        log_every=2,
        curve_sizes=[2, 4],
        accuracy_gate=0.65,
    )
    data.update(overrides)
    return RewardTrainConfig(**data)


def test_reward_stage_registered():
    from tinyfables.config import RewardTrainConfig

    assert REGISTRY["reward"] == (RewardTrainConfig, "tinyfables.stages.reward")


def test_reward_stage_trains_and_writes_artifacts(tmp_path):
    out = tmp_path / "reward"
    reward_stage.run(cfg(tmp_path), out)

    assert (out / "backbone" / "model.safetensors").exists()
    assert (out / "reward_head.pt").exists()
    assert (out / "reward_model_config.json").exists()
    assert (out / "reward_summary.json").exists()
    assert (out / "data_curve.json").exists()
    assert (out / "loss_log.csv").exists()
    assert (out / "manifest.json").exists()

    summary = json.loads((out / "reward_summary.json").read_text())
    assert summary["n_train"] == 4
    assert summary["n_held_out"] == 2
    assert 0.0 <= summary["held_out_accuracy"] <= 1.0
    assert summary["accuracy_gate"] == 0.65
    assert summary["rm_gate_pass"] == (summary["held_out_accuracy"] >= 0.65)

    curve = json.loads((out / "data_curve.json").read_text())
    assert [row["requested_train_size"] for row in curve["points"]] == [2, 4]
    assert all(0.0 <= row["held_out_accuracy"] <= 1.0 for row in curve["points"])

    rows = list(csv.DictReader(open(out / "loss_log.csv")))
    assert rows
    assert set(rows[0]) == {"step", "loss", "held_out_accuracy"}


def test_saved_reward_model_can_score_after_stage(tmp_path):
    out = tmp_path / "reward"
    reward_stage.run(cfg(tmp_path), out)
    model = load_reward_model(out, device="cpu").eval()
    ids = torch.randint(0, model.backbone.config.vocab_size, (2, 8))
    mask = torch.ones_like(ids)
    rewards = model(ids, mask)
    assert rewards.shape == (2,)


def test_reward_stage_manifest_records_inputs(tmp_path):
    out = tmp_path / "reward"
    reward_stage.run(cfg(tmp_path), out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "reward"
    assert "preferences.jsonl" in manifest["inputs"]
    assert "tokenizer.json" in manifest["inputs"]
    assert "reward_summary.json" in manifest["artifacts"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk .venv/bin/pytest tests/test_reward_stage.py -q
```

Expected: FAIL with `ImportError: cannot import name 'reward' from 'tinyfables.stages'`.

- [ ] **Step 3: Implement the reward stage**

Create `src/tinyfables/stages/reward.py`:

```python
"""Reward-model training stage.

Trains TinyFables Base Model + scalar head with Bradley-Terry loss, evaluates on
the deterministic held-out preference split, and reruns smaller train subsets for
the committed data-curve ablation.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import RewardTrainConfig
from tinyfables.reward_data import (
    PreferenceExample,
    collate_preference_batch,
    load_preferences,
    select_train_subset,
)
from tinyfables.reward_model import (
    RewardModel,
    bradley_terry_loss,
    preference_accuracy,
    save_reward_model,
)
from tinyfables.stage import write_manifest


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _lr_at(step: int, cfg: RewardTrainConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


def _batch(
    examples: list[PreferenceExample],
    batch_size: int,
    seed: int,
    step: int,
) -> list[PreferenceExample]:
    rng = random.Random(seed + step)
    if len(examples) <= batch_size:
        return list(examples)
    idx = rng.sample(range(len(examples)), batch_size)
    return [examples[i] for i in idx]


@torch.no_grad()
def _evaluate(
    model: RewardModel,
    examples: list[PreferenceExample],
    tokenizer,
    cfg: RewardTrainConfig,
    device: str,
) -> float:
    if not examples:
        return 0.0
    model.eval()
    accs = []
    for start in range(0, len(examples), cfg.batch_size):
        rows = examples[start : start + cfg.batch_size]
        batch = collate_preference_batch(rows, tokenizer, cfg.n_ctx, device)
        chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        accs.append(preference_accuracy(chosen, rejected) * len(rows))
    return sum(accs) / len(examples)


def _train_model(
    cfg: RewardTrainConfig,
    train: list[PreferenceExample],
    held_out: list[PreferenceExample],
    tokenizer,
    device: str,
    steps: int,
    csv_path: Path | None,
) -> tuple[RewardModel, dict]:
    torch.manual_seed(cfg.seed)
    model = RewardModel.from_base_checkpoint(cfg.base_checkpoint, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    writer = None
    fh = None
    if csv_path is not None:
        fh = csv_path.open("w", newline="")
        writer = csv.DictWriter(fh, fieldnames=["step", "loss", "held_out_accuracy"])
        writer.writeheader()

    last_loss = 0.0
    for step in range(steps):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = _lr_at(step, cfg)
        rows = _batch(train, cfg.batch_size, cfg.seed, step)
        batch = collate_preference_batch(rows, tokenizer, cfg.n_ctx, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_amp):
            chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = bradley_terry_loss(chosen, rejected)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        last_loss = float(loss.item())

        completed = step + 1
        if writer is not None and cfg.log_every and completed % cfg.log_every == 0:
            writer.writerow(
                {
                    "step": completed,
                    "loss": f"{last_loss:.6f}",
                    "held_out_accuracy": f"{_evaluate(model, held_out, tokenizer, cfg, device):.6f}",
                }
            )

    held_out_accuracy = _evaluate(model, held_out, tokenizer, cfg, device)
    if writer is not None:
        writer.writerow(
            {
                "step": steps,
                "loss": f"{last_loss:.6f}",
                "held_out_accuracy": f"{held_out_accuracy:.6f}",
            }
        )
    if fh is not None:
        fh.close()

    return model, {"final_loss": last_loss, "held_out_accuracy": held_out_accuracy}


def run(cfg: RewardTrainConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    tokenizer_path = Path(cfg.tokenizer_dir) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    train = load_preferences(cfg.preferences, split="train")
    held_out = load_preferences(cfg.preferences, split="held_out")
    if not train:
        raise ValueError("reward training requires at least one train preference")
    if not held_out:
        raise ValueError("reward training requires at least one held_out preference")

    model, metrics = _train_model(
        cfg, train, held_out, tokenizer, device, cfg.steps, out_dir / "loss_log.csv"
    )
    save_reward_model(model, out_dir)

    points = []
    for requested in cfg.curve_sizes:
        subset = select_train_subset(train, requested, cfg.seed)
        _curve_model, curve_metrics = _train_model(
            cfg, subset, held_out, tokenizer, device, cfg.curve_steps, None
        )
        points.append(
            {
                "requested_train_size": requested,
                "train_size": len(subset),
                "steps": cfg.curve_steps,
                "held_out_accuracy": round(curve_metrics["held_out_accuracy"], 6),
                "final_loss": round(curve_metrics["final_loss"], 6),
            }
        )

    curve_path = out_dir / "data_curve.json"
    curve_path.write_text(json.dumps({"points": points}, indent=2, sort_keys=True) + "\n")

    summary = {
        "n_train": len(train),
        "n_held_out": len(held_out),
        "steps": cfg.steps,
        "device": device,
        "amp": cfg.amp and device == "cuda",
        "final_loss": round(metrics["final_loss"], 6),
        "held_out_accuracy": round(metrics["held_out_accuracy"], 6),
        "accuracy_gate": cfg.accuracy_gate,
        "rm_gate_pass": metrics["held_out_accuracy"] >= cfg.accuracy_gate,
    }
    summary_path = out_dir / "reward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "reward",
        cfg,
        [
            out_dir / "backbone" / "config.json",
            out_dir / "backbone" / "generation_config.json",
            out_dir / "backbone" / "model.safetensors",
            out_dir / "reward_head.pt",
            out_dir / "reward_model_config.json",
            out_dir / "reward_summary.json",
            out_dir / "data_curve.json",
            out_dir / "loss_log.csv",
        ],
        inputs={
            "preferences.jsonl": Path(cfg.preferences),
            "tokenizer.json": tokenizer_path,
            "base_config.json": Path(cfg.base_checkpoint) / "config.json",
            "base_model.safetensors": Path(cfg.base_checkpoint) / "model.safetensors",
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk .venv/bin/pytest tests/test_reward_stage.py -q
```

Expected: PASS.

- [ ] **Step 5: Run related tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_reward_data.py tests/test_reward_model.py tests/test_reward_stage.py tests/test_config.py tests/test_full_configs.py tests/test_cli_dispatch.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add src/tinyfables/stages/reward.py tests/test_reward_stage.py
rtk git commit -m "feat(reward): train reward model and data curve"
```

---

### Task 5: Alignment Gate Stage + PPO-Facing Guard

**Files:**
- Create: `src/tinyfables/gates.py`
- Create: `src/tinyfables/stages/gate.py`
- Create: `tests/test_gate_stage.py`

**Interfaces:**
- Consumes:
  - `GateConfig`
  - `runs/audit_base/audit.json` schema from issue 06
  - `runs/reward_base/reward_summary.json` schema from Task 4
- Produces:
  - CLI stage: `python -m tinyfables run gate --config ... --out ...`
  - `gate.json` with `pass`, `reasons`, `warnings`, `inputs`, `thresholds`
  - `gate_report.md`
  - `assert_gate_passed(path: str | Path) -> dict`, which issue 08 PPO must call before training.

- [ ] **Step 1: Write failing tests**

Create `tests/test_gate_stage.py`:

```python
import json

import pytest

from tinyfables.config import GateConfig
from tinyfables.gates import GateError, assert_gate_passed
from tinyfables.stages import REGISTRY
from tinyfables.stages import gate as gate_stage


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def audit(pass_self=True, position_flag=False):
    return {
        "gate": {
            "self_consistency_gate": 0.85,
            "self_consistency_pass": pass_self,
            "position_swap_review_flag": position_flag,
            "position_swap_review_threshold": 0.5,
        },
        "self_consistency": {
            "mean_agreement": 0.9 if pass_self else 0.7777777777777778,
            "n_pairs": 30,
            "n_unanimous": 27 if pass_self else 21,
        },
        "position_swap": {
            "flip_rate": 0.2 if not position_flag else 0.6,
            "n_flipped": 40 if not position_flag else 120,
            "n_pairs": 200,
        },
    }


def reward(acc=0.72):
    return {
        "held_out_accuracy": acc,
        "accuracy_gate": 0.65,
        "rm_gate_pass": acc >= 0.65,
        "n_train": 1761,
        "n_held_out": 188,
    }


def test_gate_stage_registered():
    from tinyfables.config import GateConfig

    assert REGISTRY["gate"] == (GateConfig, "tinyfables.stages.gate")


def test_gate_passes_when_labeler_and_rm_pass(tmp_path):
    audit_path = write_json(tmp_path / "audit.json", audit(pass_self=True))
    reward_path = write_json(tmp_path / "reward_summary.json", reward(acc=0.72))
    out = tmp_path / "gate"

    gate_stage.run(GateConfig(audit=str(audit_path), reward_summary=str(reward_path)), out)

    gate = json.loads((out / "gate.json").read_text())
    assert gate["pass"] is True
    assert gate["reasons"] == []
    assert gate["inputs"]["reward"]["held_out_accuracy"] == 0.72
    assert (out / "gate_report.md").exists()
    assert (out / "manifest.json").exists()
    assert assert_gate_passed(out / "gate.json")["pass"] is True


def test_gate_fails_on_issue06_self_consistency_failure(tmp_path):
    audit_path = write_json(tmp_path / "audit.json", audit(pass_self=False))
    reward_path = write_json(tmp_path / "reward_summary.json", reward(acc=0.72))
    out = tmp_path / "gate"

    gate_stage.run(GateConfig(audit=str(audit_path), reward_summary=str(reward_path)), out)

    gate = json.loads((out / "gate.json").read_text())
    assert gate["pass"] is False
    assert "labeler self-consistency below gate" in gate["reasons"]
    with pytest.raises(GateError, match="alignment gate failed"):
        assert_gate_passed(out / "gate.json")


def test_gate_fails_on_rm_accuracy_below_threshold(tmp_path):
    audit_path = write_json(tmp_path / "audit.json", audit(pass_self=True))
    reward_path = write_json(tmp_path / "reward_summary.json", reward(acc=0.61))
    out = tmp_path / "gate"

    gate_stage.run(GateConfig(audit=str(audit_path), reward_summary=str(reward_path)), out)

    gate = json.loads((out / "gate.json").read_text())
    assert gate["pass"] is False
    assert "reward model held-out accuracy below gate" in gate["reasons"]


def test_position_swap_review_flag_is_warning_not_hard_fail(tmp_path):
    audit_path = write_json(tmp_path / "audit.json", audit(pass_self=True, position_flag=True))
    reward_path = write_json(tmp_path / "reward_summary.json", reward(acc=0.72))
    out = tmp_path / "gate"

    gate_stage.run(GateConfig(audit=str(audit_path), reward_summary=str(reward_path)), out)

    gate = json.loads((out / "gate.json").read_text())
    assert gate["pass"] is True
    assert "position-swap review flag set" in gate["warnings"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk .venv/bin/pytest tests/test_gate_stage.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.gates'`.

- [ ] **Step 3: Add the PPO-facing guard**

Create `src/tinyfables/gates.py`:

```python
"""Machine-readable alignment gate helpers.

Future PPO code must call assert_gate_passed before optimizing against a reward
model. This keeps "do not optimize noisy/self-inconsistent labels" executable.
"""

from __future__ import annotations

import json
from pathlib import Path


class GateError(RuntimeError):
    pass


def load_gate(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def assert_gate_passed(path: str | Path) -> dict:
    gate = load_gate(path)
    if not gate.get("pass", False):
        reasons = ", ".join(gate.get("reasons", [])) or "unknown reason"
        raise GateError(f"alignment gate failed: {reasons}")
    return gate
```

- [ ] **Step 4: Implement the gate stage**

Create `src/tinyfables/stages/gate.py`:

```python
"""Alignment gate stage for issue 07."""

from __future__ import annotations

import json
from pathlib import Path

from tinyfables.config import GateConfig
from tinyfables.stage import write_manifest


def run(cfg: GateConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = Path(cfg.audit)
    reward_path = Path(cfg.reward_summary)
    audit = json.loads(audit_path.read_text())
    reward = json.loads(reward_path.read_text())

    reasons: list[str] = []
    warnings: list[str] = []

    labeler_pass = bool(audit["gate"]["self_consistency_pass"])
    if cfg.require_labeler_self_consistency and not labeler_pass:
        reasons.append("labeler self-consistency below gate")

    rm_accuracy = float(reward["held_out_accuracy"])
    if rm_accuracy < cfg.rm_accuracy_gate:
        reasons.append("reward model held-out accuracy below gate")

    if audit["gate"].get("position_swap_review_flag", False):
        warnings.append("position-swap review flag set")

    gate = {
        "pass": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "thresholds": {
            "rm_accuracy_gate": cfg.rm_accuracy_gate,
            "labeler_self_consistency_required": cfg.require_labeler_self_consistency,
            "labeler_self_consistency_gate": audit["gate"]["self_consistency_gate"],
            "position_swap_review_threshold": audit["gate"]["position_swap_review_threshold"],
        },
        "inputs": {
            "audit": {
                "self_consistency_pass": labeler_pass,
                "self_consistency": audit["self_consistency"],
                "position_swap": audit["position_swap"],
                "position_swap_review_flag": audit["gate"].get("position_swap_review_flag", False),
            },
            "reward": {
                "held_out_accuracy": rm_accuracy,
                "n_train": reward["n_train"],
                "n_held_out": reward["n_held_out"],
                "rm_gate_pass": reward["rm_gate_pass"],
            },
        },
    }

    gate_path = out_dir / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    verdict = "PASS - PPO may run" if gate["pass"] else "NO-GO - PPO must not run"
    report_lines = [
        "# Alignment gate report",
        "",
        f"Verdict: {verdict}",
        "",
        "| input | value |",
        "|---|---|",
        f"| RM held-out accuracy | {rm_accuracy:.3f} (gate {cfg.rm_accuracy_gate:.2f}) |",
        f"| Labeler self-consistency | {audit['self_consistency']['mean_agreement']:.3f} (gate {audit['gate']['self_consistency_gate']:.2f}) |",
        f"| Position-swap flip rate | {audit['position_swap']['flip_rate']:.3f} |",
        "",
    ]
    if reasons:
        report_lines.extend(["Reasons:", *[f"- {reason}" for reason in reasons], ""])
    if warnings:
        report_lines.extend(["Warnings:", *[f"- {warning}" for warning in warnings], ""])
    report_path = out_dir / "gate_report.md"
    report_path.write_text("\n".join(report_lines))

    write_manifest(
        out_dir,
        "gate",
        cfg,
        [gate_path, report_path],
        inputs={"audit.json": audit_path, "reward_summary.json": reward_path},
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
rtk .venv/bin/pytest tests/test_gate_stage.py -q
```

Expected: PASS.

- [ ] **Step 6: Run all reward/gate tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_reward_data.py tests/test_reward_model.py tests/test_reward_stage.py tests/test_gate_stage.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add src/tinyfables/gates.py src/tinyfables/stages/gate.py tests/test_gate_stage.py
rtk git commit -m "feat(gates): emit alignment go-no-go record"
```

---

### Task 6: End-to-End Toy Chain and Full Test Suite

**Files:**
- Modify: `tests/test_feedback_chain.py`

**Interfaces:**
- Consumes: `reward` and `gate` stages from Tasks 4 and 5.
- Produces: regression coverage that toy-scale reward and gate stages fit into the existing pipeline.

- [ ] **Step 1: Add a toy-chain reward smoke test**

Append to `tests/test_feedback_chain.py`:

```python
def test_issue07_reward_and_gate_toy_chain(tmp_path):
    import json
    import torch
    from tokenizers import Tokenizer

    from tinyfables.config import GateConfig, RewardTrainConfig, SourceSpec, TokenizerConfig
    from tinyfables.model import GPT, GPTConfig
    from tinyfables.stages import gate as gate_stage
    from tinyfables.stages import reward as reward_stage
    from tinyfables.stages import tokenizer as tokenizer_stage

    fix = Path(__file__).parent / "fixtures"
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(fix / "tiny_corpus.jsonl")), vocab_size=512, seed=0),
        tok_dir,
    )
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    base = tmp_path / "base"
    torch.manual_seed(0)
    GPT(
        GPTConfig(
            vocab_size=tok.get_vocab_size(),
            n_layer=1,
            n_head=2,
            d_model=32,
            n_ctx=128,
        )
    ).save_pretrained(base)

    reward_out = tmp_path / "reward"
    reward_stage.run(
        RewardTrainConfig(
            preferences=str(fix / "preferences_replay.jsonl"),
            base_checkpoint=str(base),
            tokenizer_dir=str(tok_dir),
            n_ctx=128,
            batch_size=2,
            steps=4,
            curve_steps=2,
            lr=1e-3,
            warmup_steps=0,
            seed=0,
            device="cpu",
            amp=False,
            log_every=2,
            curve_sizes=[2],
        ),
        reward_out,
    )

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "gate": {
                    "self_consistency_gate": 0.85,
                    "self_consistency_pass": True,
                    "position_swap_review_flag": False,
                    "position_swap_review_threshold": 0.5,
                },
                "self_consistency": {"mean_agreement": 0.9, "n_pairs": 30, "n_unanimous": 27},
                "position_swap": {"flip_rate": 0.2, "n_flipped": 40, "n_pairs": 200},
            }
        )
        + "\n"
    )

    gate_out = tmp_path / "gate"
    gate_stage.run(
        GateConfig(audit=str(audit_path), reward_summary=str(reward_out / "reward_summary.json")),
        gate_out,
    )
    assert (gate_out / "gate.json").exists()
```

If `tests/test_feedback_chain.py` does not already import `Path`, add this at the top:

```python
from pathlib import Path
```

- [ ] **Step 2: Run the toy-chain test to verify it fails or passes**

Run:

```bash
rtk .venv/bin/pytest tests/test_feedback_chain.py::test_issue07_reward_and_gate_toy_chain -q
```

Expected before fixing imports: FAIL if `Path` is missing. Expected after adding import: PASS.

- [ ] **Step 3: Run the full test suite**

Run:

```bash
rtk .venv/bin/pytest -q
```

Expected: PASS, with the existing network marker deselected by default.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/test_feedback_chain.py
rtk git commit -m "test(reward): cover toy reward-gate chain"
```

---

### Task 7: Track B Real Reward Run, Gate Run, and Hub Push

**Files:**
- Read/write artifacts under `runs/reward_base/` and `runs/gate_base/` (gitignored).
- No source edits until Task 8 docs.

**Interfaces:**
- Consumes:
  - `runs/derive_base/preferences.jsonl`
  - `runs/audit_base/audit.json`
  - `runs/base_model`
  - `runs/tokenizer_full`
- Produces:
  - `runs/reward_base/reward_summary.json`
  - `runs/reward_base/data_curve.json`
  - `runs/reward_base/manifest.json`
  - `runs/gate_base/gate.json`
  - Hub repo `congthanh991/tinyfables-13m-rm`

- [ ] **Step 1: Verify required issue 06 artifacts exist**

Run:

```bash
rtk wc -l runs/derive_base/preferences.jsonl
rtk sed -n '1,220p' runs/derive_base/derive_summary.json
rtk sed -n '1,220p' runs/audit_base/audit.json
rtk test -f runs/base_model/model.safetensors
rtk test -f runs/tokenizer_full/tokenizer.json
```

Expected:
- `preferences.jsonl` line count is `1949`.
- `derive_summary.json` reports `n_train: 1761` and `n_held_out: 188`.
- `audit.json` reports `self_consistency_pass: false`.
- Both `test -f` commands exit with code 0.

- [ ] **Step 2: Run the real reward stage**

Recommended on Colab T4 or local MPS/CUDA:

```bash
rtk .venv/bin/python -m tinyfables run reward --config configs/reward_full.yaml --out runs/reward_base
```

Expected: command exits 0 and prints:

```text
[tinyfables] stage 'reward' complete -> runs/reward_base
```

- [ ] **Step 3: Verify reward artifacts**

Run:

```bash
rtk sed -n '1,220p' runs/reward_base/reward_summary.json
rtk sed -n '1,260p' runs/reward_base/data_curve.json
rtk sed -n '1,220p' runs/reward_base/manifest.json
```

Expected:
- `reward_summary.json` has `n_train: 1761`, `n_held_out: 188`, `held_out_accuracy` in `[0, 1]`, and `rm_gate_pass` matching `held_out_accuracy >= 0.65`.
- `data_curve.json` has four points with `requested_train_size` exactly `100`, `500`, `1000`, `2000`.
- The `2000` point has `train_size: 1761` because only 1,761 train preferences exist.
- `manifest.json` has `"stage": "reward"`.

- [ ] **Step 4: Run the gate stage**

Run:

```bash
rtk .venv/bin/python -m tinyfables run gate --config configs/gate_full.yaml --out runs/gate_base
```

Expected: command exits 0 and prints:

```text
[tinyfables] stage 'gate' complete -> runs/gate_base
```

- [ ] **Step 5: Verify gate artifact preserves issue 06 failure**

Run:

```bash
rtk sed -n '1,260p' runs/gate_base/gate.json
rtk sed -n '1,220p' runs/gate_base/gate_report.md
```

Expected:
- `gate.json` has `"pass": false` unless the gate config was intentionally changed.
- `gate.json["reasons"]` includes `"labeler self-consistency below gate"` because issue 06 self-consistency was `0.778 < 0.85`.
- If `held_out_accuracy < 0.65`, `gate.json["reasons"]` also includes `"reward model held-out accuracy below gate"`.

- [ ] **Step 6: Push the reward model folder to Hugging Face Hub**

Run after inspecting `runs/reward_base/manifest.json`:

```bash
rtk .venv/bin/python -m tinyfables push --kind model --path runs/reward_base --repo congthanh991/tinyfables-13m-rm --public
```

Expected:

```text
[tinyfables] pushed model -> congthanh991/tinyfables-13m-rm
```

- [ ] **Step 7: Commit only source/config/test changes**

Run:

```bash
rtk git status --short --branch
```

Expected:
- `runs/` artifacts are not staged because `runs/` is gitignored.
- Source/config/test commits from Tasks 1-6 are already committed.

Do not commit `runs/reward_base` or `runs/gate_base`.

---

### Task 8: Documentation and Issue Closeout

**Files:**
- Modify: `docs/design.md`
- Modify: `docs/issues/07-reward-model-gates.md`

**Interfaces:**
- Consumes real outputs:
  - `runs/reward_base/reward_summary.json`
  - `runs/reward_base/data_curve.json`
  - `runs/gate_base/gate.json`
  - Hub repo from Task 7
- Produces committed documentation of the RM run, data curve, gate verdict, and Hub publication.

- [ ] **Step 1: Extract the exact run numbers**

Run:

```bash
rtk sed -n '1,220p' runs/reward_base/reward_summary.json
rtk sed -n '1,260p' runs/reward_base/data_curve.json
rtk sed -n '1,260p' runs/gate_base/gate.json
```

Record these exact fields for the docs:
- `reward_summary.held_out_accuracy`
- `reward_summary.rm_gate_pass`
- `reward_summary.n_train`
- `reward_summary.n_held_out`
- every `data_curve.points[*].requested_train_size`
- every `data_curve.points[*].train_size`
- every `data_curve.points[*].held_out_accuracy`
- `gate.pass`
- `gate.reasons`

- [ ] **Step 2: Update `docs/design.md`**

Generate the exact subsection text from the real artifacts:

```bash
rtk .venv/bin/python -c 'import json
summary=json.load(open("runs/reward_base/reward_summary.json"))
curve=json.load(open("runs/reward_base/data_curve.json"))["points"]
gate=json.load(open("runs/gate_base/gate.json"))
acc={p["requested_train_size"]: p for p in curve}
gate_reasons=", ".join(gate["reasons"]) if gate["reasons"] else "no blocking reasons"
rm_gate="pass" if summary["rm_gate_pass"] else "fail"
gate_word="PASS" if gate["pass"] else "NO-GO"
print(f"""### Implementation (issue 07)

- **Reward model stage.** `reward` trains the Base Model plus a scalar head using Bradley-Terry
  loss over `runs/derive_base/preferences.jsonl`. The reward score is pooled from the last
  non-pad token of `prompt + fable + <|endoftext|>`, matching the prep/generation tokenization
  contract. The stage writes the RM folder (`backbone/`, `reward_head.pt`,
  `reward_model_config.json`), `reward_summary.json`, `data_curve.json`, `loss_log.csv`, and
  manifest last.
- **Track B real RM run (2026-07-10).** Trained on `{summary["n_train"]}` train preferences and
  evaluated on `{summary["n_held_out"]}` held-out preferences. Held-out accuracy was
  `{summary["held_out_accuracy"]:.3f}`, so the RM accuracy gate (`>=0.65`) was `{rm_gate}`.
  Data curve points were: `100 -> {acc[100]["held_out_accuracy"]:.3f}`,
  `500 -> {acc[500]["held_out_accuracy"]:.3f}`,
  `1000 -> {acc[1000]["held_out_accuracy"]:.3f}`,
  `2000 requested / {acc[2000]["train_size"]} available -> {acc[2000]["held_out_accuracy"]:.3f}`.
- **Alignment gate.** `gate` combines issue 06 audit numbers with RM accuracy into an explicit
  PPO go/no-go record. The real gate verdict was `{gate_word}`: `{gate_reasons}`. Because issue
  06 labeler self-consistency was `0.778 < 0.85`, the gate is expected to remain no-go unless
  the project intentionally accepts noisy-label risk in a later ADR. The Reward Model artifact
  was pushed to `congthanh991/tinyfables-13m-rm`.
""")'
```

Insert the printed subsection immediately after the issue 06 Track B paragraph in `docs/design.md`.

- [ ] **Step 3: Tick issue 07 acceptance criteria**

Modify `docs/issues/07-reward-model-gates.md`:

```markdown
## Acceptance criteria

- [x] RM trains at toy scale inside the test suite; real run reports held-out accuracy
- [x] Data-curve artifact with all four points, report-ready
- [x] Gate stage emits an explicit pass/fail record with its inputs (audit numbers, RM accuracy)
- [x] Reward Model pushed to the Hub
```

- [ ] **Step 4: Run documentation diff**

Run:

```bash
rtk git diff -- docs/design.md docs/issues/07-reward-model-gates.md
```

Expected:
- `docs/design.md` records actual reward/data-curve/gate values.
- `docs/issues/07-reward-model-gates.md` has all four acceptance criteria ticked.
- The gate failure caused by issue 06 is explicitly described if `gate.pass` is false.

- [ ] **Step 5: Run final tests**

Run:

```bash
rtk .venv/bin/pytest -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add docs/design.md docs/issues/07-reward-model-gates.md
rtk git commit -m "docs(reward): record reward model gates"
```

---

## Final Verification Checklist

- [ ] `rtk .venv/bin/pytest -q` passes.
- [ ] `rtk .venv/bin/python -m tinyfables run reward --config configs/reward_full.yaml --out runs/reward_base` completed.
- [ ] `runs/reward_base/reward_summary.json` reports held-out accuracy and `rm_gate_pass`.
- [ ] `runs/reward_base/data_curve.json` contains requested points `100`, `500`, `1000`, `2000`.
- [ ] `rtk .venv/bin/python -m tinyfables run gate --config configs/gate_full.yaml --out runs/gate_base` completed.
- [ ] `runs/gate_base/gate.json` includes issue 06 audit inputs and RM accuracy.
- [ ] `runs/gate_base/gate.json` is no-go if labeler self-consistency remains below `0.85`.
- [ ] `rtk .venv/bin/python -m tinyfables push --kind model --path runs/reward_base --repo congthanh991/tinyfables-13m-rm --public` completed.
- [ ] `docs/design.md` records real issue 07 results.
- [ ] `docs/issues/07-reward-model-gates.md` acceptance criteria are ticked.
- [ ] Worktree is clean except ignored `runs/` artifacts.

## Self-Review Results

- **Spec coverage:** The plan implements toy RM training, real held-out accuracy reporting, all four data-curve points, explicit gate artifact with audit + RM inputs, and Hub push.
- **Placeholder scan:** The plan avoids unspecified implementation steps. The only run-dependent values are explicitly sourced from generated JSON artifacts in Task 8.
- **Type consistency:** `RewardTrainConfig`, `GateConfig`, `PreferenceExample`, `RewardModel`, `reward_summary.json`, `data_curve.json`, and `gate.json` field names are consistent across tasks.
