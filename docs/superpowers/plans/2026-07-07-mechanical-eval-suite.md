# Mechanical Eval Suite (Issue 05) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deterministic `evaluate` stage runnable against any checkpoint that produces a report-ready metrics document — fable-token perplexity, verbatim-element spec adherence split into the canonical/seen/held-out robustness grid, moral delivery (bold-marker + fuzzy fallback), distinct-n/repetition, and length stats — plus a hand-validation worksheet for moral extraction.

**Architecture:** Small torch-free metric modules (`eval_metrics.py` text metrics, `moral.py` extraction+similarity, `report.py` Markdown/JSON export) that are trivially unit-tested, one torch module (`perplexity.py`), and a `stages/evaluate.py` orchestrator that generates under three prompt-family renderings per spec and aggregates into the grid. Registered as the `evaluate` stage so the existing generic CLI (`python -m tinyfables run evaluate --config … --out …`) drives it with no CLI change. Track A is the code (toy-tested); Track B runs it on the real Base Model and does the ~50-sample moral-extraction hand-validation.

**Tech Stack:** Python 3.10+, PyTorch (perplexity + generation), tokenizers, transformers (`GPT.from_pretrained`), stdlib only for metrics (`re`, `difflib.SequenceMatcher`, `statistics`) — no new dependencies. pytest. Local `.venv` (`.venv/bin/pytest`, `.venv/bin/python`).

## Global Constraints

- **Deterministic given seed + eval set** — perplexity has no sampling; generation uses seeded sampling (`seed = cfg.seed + spec_index`); row sampling is deterministic (`iter_rows(source, seed)`). Two runs of the same config+checkpoint must produce identical `eval_metrics.json`.
- **Runs at toy scale inside the test suite** — the toy path uses `tests/fixtures/tiny_corpus.jsonl` + a vocab-512 tokenizer + a tiny trained checkpoint; tests stay OFFLINE (no network; `pyproject.toml` deselects `network`-marked tests) and fast.
- **Manifest-written-last is the completion marker** — the `evaluate` stage writes its artifacts then `manifest.json` LAST via `stage.write_manifest`, matching `prep`/`pretrain`/`benchmark`.
- **Lazy stage registry intact** — `evaluate` is registered by module-path STRING in `REGISTRY`; the module (which imports torch) loads only at dispatch. The torch-free metric modules (`eval_metrics.py`, `moral.py`, `report.py`) MUST NOT import torch at module top level.
- **Single-band dataset (issue 04)** — ds-tf1-en-3m is entirely age group B, 4-7 years, ~250 words. FableSpecs parsed from val prompts carry `age_range="4-7"`; `render_canonical_prompt` raises on any other band. Do not construct eval specs with other age ranges.
- **Moral marker is rare in generations (issue-04 finding)** — the Base Model rarely emits `**bold**` moral markers (~2/20 in the spot check), so `extract_moral` MUST fall back to the trailing sentence, not rely on the marker. This is why moral extraction needs hand-validation against the naive-regex baseline (which the design says misses ~10%).
- **Verbatim-element adherence** — spec adherence is the case-insensitive substring presence of each requested Element value (character/setting/challenge/outcome) in the generated fable — a mechanical, imperfect-but-comparable proxy. Element values are preserved verbatim in prompts (issue 03), keeping this measurable. Moral is scored separately (moral delivery), not in the adherence substring set.
- **Robustness grid uses the paraphrase bank at EVAL time** — held-out templates (`heldout-*`, never in training) are USED here to render the held-out column: each spec is rendered under canonical + a seen template + a held-out template, generated, and adherence grouped by family. This is where held-out phrasing robustness is measured, not asserted.
- **Report is report-usable with provenance** — the stage writes `eval_metrics.json` (machine), `eval_report.md` (human tables), and `moral_calibration.jsonl` (hand-labeling worksheet); provenance (checkpoint sha, tokenizer sha, seed, versions) is embedded and also recorded in `manifest.json`.

---

## File Structure

**New modules:**
- `src/tinyfables/eval_metrics.py` — torch-free text metrics: `element_adherence`, `distinct_n`, `repetition_rate`, `length_stats`.
- `src/tinyfables/moral.py` — torch-free moral extraction + baseline + similarity: `extract_moral`, `naive_regex_moral`, `moral_similarity`, `moral_delivered`, `extraction_precision`.
- `src/tinyfables/perplexity.py` — torch: `fable_token_perplexity` (completion-only fable-token loss + ppl over val rows).
- `src/tinyfables/report.py` — torch-free: `write_eval_report(out_dir, metrics) -> Path` (Markdown tables from the metrics dict).
- `src/tinyfables/stages/evaluate.py` — the `evaluate` stage (perplexity + robustness-grid generation + calibration worksheet + report + manifest).

**Modified:**
- `src/tinyfables/config.py` — add `EvalConfig`.
- `src/tinyfables/stages/__init__.py` — register `evaluate`.
- `configs/eval_toy.yaml` (new), `configs/eval_full.yaml` (new).
- `docs/design.md` — "Implementation (issue 05)" note + (Track B) real-eval + moral-calibration record.
- `docs/issues/README.md` — tick issue 05 (end of Track B).

**New tests:** `tests/test_eval_metrics.py`, `tests/test_moral.py`, `tests/test_perplexity.py`, `tests/test_report.py`, `tests/test_evaluate_stage.py`; additions to `tests/test_config.py`.

**Branch:** `feat/issue-05-eval-suite` off `main`.

---

### Task 1: Text metrics (`eval_metrics.py`)

**Files:**
- Create: `src/tinyfables/eval_metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `tinyfables.prompts.FableSpec`.
- Produces:
  - `element_adherence(fable: str, spec: FableSpec) -> dict[str, bool]` — keys are the present non-moral elements among `("character","setting","challenge","outcome")`, value = case-insensitive substring presence.
  - `distinct_n(text: str, n: int) -> float` — unique n-grams / total n-grams (0.0 if fewer than n word-tokens).
  - `repetition_rate(text: str, n: int = 4) -> float` — `1 - distinct_n(text, n)` (higher = more repetitive; 0.0 if too short).
  - `length_stats(texts: list[str]) -> dict` — `{"mean","median","min","max"}` over word counts.
  - Used by `evaluate.run` (Task 6) and `report.py` (Task 5).

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_metrics.py`:

```python
from tinyfables.eval_metrics import distinct_n, element_adherence, length_stats, repetition_rate
from tinyfables.prompts import FableSpec

SPEC = FableSpec(character="a shy hedgehog", setting="a moonlit garden",
                 challenge="fear of the dark", outcome="finds a friend",
                 moral="courage grows when shared")


def test_element_adherence_case_insensitive_substring():
    fable = "Once a Shy Hedgehog in a bright cave faced fear of the dark and won."
    a = element_adherence(fable, SPEC)
    assert a == {"character": True, "setting": False, "challenge": True, "outcome": False}
    # moral is NOT in the adherence set (scored separately as moral delivery)
    assert "moral" not in a


def test_element_adherence_skips_missing_elements():
    a = element_adherence("a mouse ran", FableSpec(character="a mouse"))
    assert a == {"character": True}


def test_distinct_n_and_repetition():
    assert distinct_n("a a a a", 1) == 0.25          # 1 unique / 4
    assert distinct_n("the cat sat", 2) == 1.0        # 2 unique bigrams / 2
    assert distinct_n("hi", 3) == 0.0                 # fewer than 3 tokens
    assert repetition_rate("a a a a", 1) == 0.75
    assert repetition_rate("hi", 4) == 0.0


def test_length_stats():
    s = length_stats(["one two three", "one two three four five", "one"])
    assert s["min"] == 1 and s["max"] == 5 and s["mean"] == 3.0 and s["median"] == 3
    assert length_stats([]) == {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinyfables.eval_metrics'`.

- [ ] **Step 3: Write `src/tinyfables/eval_metrics.py`**

```python
"""Deterministic, torch-free text metrics for the eval suite: verbatim-element
spec adherence, distinct-n / repetition, and length statistics. These are crude
mechanical proxies (the model paraphrases requested Elements) but they are
deterministic and comparable across checkpoints — which is what the suite needs
to compare Base vs Aligned. Moral delivery is scored separately (see moral.py)."""

from __future__ import annotations

import re
from statistics import mean, median

from tinyfables.prompts import FableSpec

# Elements scored by verbatim presence. Moral is excluded — it gets the fuzzy
# moral-delivery treatment instead (a fable states its moral, rarely verbatim).
_ADHERENCE_ELEMENTS = ("character", "setting", "challenge", "outcome")

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(toks: list[str], n: int) -> list[tuple]:
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def element_adherence(fable: str, spec: FableSpec) -> dict[str, bool]:
    low = fable.lower()
    out: dict[str, bool] = {}
    for name in _ADHERENCE_ELEMENTS:
        value = getattr(spec, name)
        if value is not None:
            out[name] = value.lower() in low
    return out


def distinct_n(text: str, n: int) -> float:
    grams = _ngrams(_tokens(text), n)
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(text: str, n: int = 4) -> float:
    grams = _ngrams(_tokens(text), n)
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def length_stats(texts: list[str]) -> dict:
    wc = [len(_tokens(t)) for t in texts]
    if not wc:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {"mean": round(mean(wc), 1), "median": median(wc), "min": min(wc), "max": max(wc)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval_metrics.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/eval_metrics.py tests/test_eval_metrics.py
git commit -m "feat: text metrics (adherence, distinct-n, length) for eval suite (issue 05)"
```

---

### Task 2: Moral extraction (`moral.py`)

**Files:**
- Create: `src/tinyfables/moral.py`
- Test: `tests/test_moral.py`

**Interfaces:**
- Consumes: stdlib only (`re`, `difflib`).
- Produces:
  - `extract_moral(fable: str) -> str | None` — last `**bold**` segment's inner text if any; else the last non-empty sentence; else None. Requested-moral-agnostic.
  - `naive_regex_moral(fable: str) -> str | None` — the keyword-regex baseline (the ~90%-precision approach to beat).
  - `moral_similarity(a: str | None, b: str | None) -> float` — `SequenceMatcher` ratio on lowercased text (0.0 if either is falsy).
  - `moral_delivered(fable: str, requested: str, threshold: float) -> bool` — `moral_similarity(extract_moral(fable), requested) >= threshold`.
  - `extraction_precision(labels: list[bool]) -> float` — precision from hand labels (fraction True; 0.0 if empty).
  - Used by `evaluate.run` (Task 6), `report.py` (Task 5), and the Track B calibration.

- [ ] **Step 1: Write the failing test**

Create `tests/test_moral.py`:

```python
from tinyfables.moral import (
    extract_moral,
    extraction_precision,
    moral_delivered,
    moral_similarity,
    naive_regex_moral,
)


def test_extract_prefers_last_bold_segment():
    fable = "A tale. **Be kind.** More story. **Courage grows when shared**"
    assert extract_moral(fable) == "Courage grows when shared"


def test_extract_falls_back_to_last_sentence_when_no_bold():
    # The issue-04 finding: generations rarely have bold markers.
    fable = "The mouse shared its food. From that day, kindness returned to the meadow."
    assert extract_moral(fable) == "From that day, kindness returned to the meadow"
    assert extract_moral("") is None


def test_naive_regex_baseline_matches_keyword_forms_only():
    assert naive_regex_moral("The moral is: patience pays.") == "patience pays"
    assert naive_regex_moral("Lesson: share and thrive") == "share and thrive"
    # No keyword marker -> the naive baseline MISSES it (extract_moral would not).
    assert naive_regex_moral("And so kindness returned to the meadow.") is None


def test_moral_similarity_and_delivered():
    assert moral_similarity("courage grows when shared", "courage grows when shared") == 1.0
    assert moral_similarity(None, "x") == 0.0
    fable = "The friends learned that courage grows when it is shared."
    assert moral_delivered(fable, "courage grows when shared", threshold=0.5)
    assert not moral_delivered(fable, "greed leads to ruin", threshold=0.5)


def test_extraction_precision():
    assert extraction_precision([True, True, False, True]) == 0.75
    assert extraction_precision([]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_moral.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinyfables.moral'`.

- [ ] **Step 3: Write `src/tinyfables/moral.py`**

```python
"""Moral extraction and matching for the eval suite (torch-free).

`extract_moral` recovers the fable's OWN stated moral without peeking at the
requested moral (so moral-delivery isn't circular): it prefers the last **bold**
segment (the dataset marks morals in bold), but — per the issue-04 finding that
the Base Model rarely emits the marker — it falls back to the trailing sentence
(the prompt tells the model to "End with a clear connection to the moral").

`naive_regex_moral` is the keyword-regex baseline the design flags as missing
~10% of morals; our extraction must beat it (validated by hand in Track B)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NAIVE_RE = re.compile(
    r"(?:the\s+)?(?:moral|lesson|teaching)(?:\s+of\s+the\s+story)?\s*(?:is|:)\s*(.+)",
    re.IGNORECASE,
)


def extract_moral(fable: str) -> str | None:
    bolds = _BOLD_RE.findall(fable)
    if bolds:
        return bolds[-1].strip().strip(".")
    sentences = [s.strip() for s in _SENT_SPLIT.split(fable.strip()) if s.strip()]
    return sentences[-1].strip().strip(".") if sentences else None


def naive_regex_moral(fable: str) -> str | None:
    found = None
    for line in fable.splitlines():
        m = _NAIVE_RE.search(line)
        if m:
            found = m.group(1).strip().strip(".")
    return found


def moral_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def moral_delivered(fable: str, requested: str, threshold: float) -> bool:
    return moral_similarity(extract_moral(fable), requested) >= threshold


def extraction_precision(labels: list[bool]) -> float:
    return sum(labels) / len(labels) if labels else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_moral.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/moral.py tests/test_moral.py
git commit -m "feat: moral extraction (bold + fuzzy fallback) + naive baseline (issue 05)"
```

---

### Task 3: Fable-token perplexity (`perplexity.py`)

**Files:**
- Create: `src/tinyfables/perplexity.py`
- Test: `tests/test_perplexity.py`

**Interfaces:**
- Consumes: `tinyfables.constants.EOT`; torch; a model with `forward(input_ids=...) -> out.logits`; a `tokenizers.Tokenizer`.
- Produces:
  - `fable_token_perplexity(model, tokenizer, rows, n_ctx, device, max_rows) -> tuple[float, float]` — iterates val `rows` (each `{"prompt","fable"}`), encodes `prompt + fable + EOT` with the prompt masked, sums completion-only cross-entropy over fable tokens across up to `max_rows` rows (skipping any row longer than `n_ctx`), returns `(mean_loss, perplexity)`; `(nan, nan)` if no usable rows.
  - Used by `evaluate.run` (Task 6).

- [ ] **Step 1: Write the failing test**

Create `tests/test_perplexity.py`:

```python
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.constants import EOT
from tinyfables.data import read_rows
from tinyfables.model import GPT, GPTConfig
from tinyfables.perplexity import fable_token_perplexity
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def _tok(tmp_path):
    out = tmp_path / "tok"
    tokenizer_stage.run(TokenizerConfig(source=SourceSpec(jsonl_path=FIXTURE), vocab_size=512, seed=0), out)
    return Tokenizer.from_file(str(out / "tokenizer.json"))


def test_perplexity_is_exp_of_mean_loss_and_finite(tmp_path):
    tok = _tok(tmp_path)
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2, d_model=64, n_ctx=512)).eval()
    rows = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)
    mean_loss, ppl = fable_token_perplexity(model, tok, rows, n_ctx=512, device="cpu", max_rows=3)
    assert math.isfinite(mean_loss) and math.isfinite(ppl)
    assert abs(ppl - math.exp(mean_loss)) < 1e-4
    assert ppl > 1.0  # untrained model


def test_perplexity_matches_manual_completion_only_loss(tmp_path):
    tok = _tok(tmp_path)
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2, d_model=64, n_ctx=512)).eval()
    row = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)[0]
    mean_loss, _ = fable_token_perplexity(model, tok, [row], n_ctx=512, device="cpu", max_rows=1)
    # independent manual computation of the same completion-only loss
    p = tok.encode(row["prompt"]).ids
    f = tok.encode(row["fable"]).ids + [tok.token_to_id(EOT)]
    ids = torch.tensor([p + f]); labels = ids.clone(); labels[0, : len(p)] = -100
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1, :]
    tgt = labels[0, 1:]; keep = tgt != -100
    expected = F.cross_entropy(logits[keep], tgt[keep]).item()
    assert abs(mean_loss - expected) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_perplexity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinyfables.perplexity'`.

- [ ] **Step 3: Write `src/tinyfables/perplexity.py`**

```python
"""Fable-token perplexity: completion-only next-token cross-entropy over the
validation slice (prompt spans masked, exactly like training loss), so
"perplexity" means perplexity *of fables*. Each val row is scored in its own
context (prompt + fable + EOT); rows longer than n_ctx are skipped. Deterministic
(no sampling)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tinyfables.constants import EOT


def fable_token_perplexity(model, tokenizer, rows, n_ctx, device, max_rows):
    eot_id = tokenizer.token_to_id(EOT)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_used = 0
    with torch.no_grad():
        for r in rows:
            if n_used >= max_rows:
                break
            p = tokenizer.encode(r["prompt"]).ids
            f = tokenizer.encode(r["fable"]).ids + [eot_id]
            ids = p + f
            if len(ids) > n_ctx or len(f) < 2:
                continue
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            labels = input_ids.clone()
            labels[0, : len(p)] = -100
            logits = model(input_ids=input_ids).logits[0, :-1, :]
            tgt = labels[0, 1:]
            keep = tgt != -100
            total_loss += F.cross_entropy(logits[keep], tgt[keep], reduction="sum").item()
            total_tokens += int(keep.sum())
            n_used += 1
    if total_tokens == 0:
        return float("nan"), float("nan")
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_perplexity.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/perplexity.py tests/test_perplexity.py
git commit -m "feat: completion-only fable-token perplexity for eval suite (issue 05)"
```

---

### Task 4: `EvalConfig`

**Files:**
- Modify: `src/tinyfables/config.py`
- Test: `tests/test_config.py` (add cases)

**Interfaces:**
- Consumes: existing `SourceSpec`, `load_config`, `_build` (handles the `source` sub-mapping), unknown-key rejection.
- Produces: `EvalConfig` (frozen) with fields:
  `checkpoint: str`, `tokenizer_dir: str`, `source: SourceSpec`, `paraphrase_bank: str | None = None`, `n_ctx: int = 1024`, `n_perplexity_rows: int = 500`, `n_generations: int = 50`, `max_new_tokens: int = 320`, `min_new_tokens: int = 80`, `temperature: float = 0.9`, `top_k: int = 50`, `moral_threshold: float = 0.3`, `seed: int = 0`, `device: str = "auto"`.
- The `evaluate` REGISTRY entry is added in Task 6 (with the module), not here.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_eval_config_parses_source_and_defaults(tmp_path):
    from tinyfables.config import EvalConfig, load_config

    p = tmp_path / "e.yaml"
    p.write_text(
        "checkpoint: runs/base\ntokenizer_dir: runs/tok\n"
        "source:\n  jsonl_path: tests/fixtures/tiny_corpus.jsonl\n"
        "paraphrase_bank: configs/paraphrases.yaml\nn_generations: 5\n"
    )
    cfg = load_config(p, EvalConfig)
    assert cfg.checkpoint == "runs/base" and cfg.source.jsonl_path.endswith("tiny_corpus.jsonl")
    assert cfg.paraphrase_bank == "configs/paraphrases.yaml" and cfg.n_generations == 5
    assert cfg.n_ctx == 1024 and cfg.moral_threshold == 0.3 and cfg.top_k == 50


def test_eval_config_rejects_unknown_key(tmp_path):
    import pytest

    from tinyfables.config import EvalConfig, load_config

    p = tmp_path / "e.yaml"
    p.write_text("checkpoint: c\ntokenizer_dir: t\nsource:\n  jsonl_path: x.jsonl\nbogus: 1\n")
    with pytest.raises(KeyError):
        load_config(p, EvalConfig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -k eval_config -v`
Expected: FAIL — `ImportError: cannot import name 'EvalConfig'`.

- [ ] **Step 3: Add `EvalConfig` to `src/tinyfables/config.py`**

Add after `BenchmarkConfig`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/config.py tests/test_config.py
git commit -m "feat: EvalConfig (issue 05)"
```

---

### Task 5: Report export (`report.py`)

**Files:**
- Create: `src/tinyfables/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: a metrics dict shaped exactly as `evaluate.run` (Task 6) produces:
  ```python
  {
    "perplexity": {"fable_token_loss": float, "perplexity": float, "n_rows": int},
    "generation": {"n_specs": int, "distinct_1": float, "distinct_2": float,
                   "repetition_4": float, "length": {"mean","median","min","max"},
                   "moral_delivery_rate": float, "moral_threshold": float},
    "adherence_grid": {  # each family present only if computed
       "<family>": {"character": float, "setting": float, "challenge": float,
                    "outcome": float, "overall": float, "moral_delivery": float, "n": int}, ...},
    "provenance": {"checkpoint_sha": str, "tokenizer_sha": str, "seed": int, "versions": dict},
  }
  ```
- Produces: `write_eval_report(out_dir, metrics) -> Path` — writes `out_dir/eval_report.md` with a perplexity line, a generation-quality table, an adherence-grid table (one row per present family, in canonical/seen/held-out order), and a provenance block; returns the path. Torch-free.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
from tinyfables.report import write_eval_report

METRICS = {
    "perplexity": {"fable_token_loss": 1.4466, "perplexity": 4.249, "n_rows": 500},
    "generation": {"n_specs": 50, "distinct_1": 0.42, "distinct_2": 0.81,
                   "repetition_4": 0.05, "length": {"mean": 248.0, "median": 249, "min": 210, "max": 260},
                   "moral_delivery_rate": 0.6, "moral_threshold": 0.3},
    "adherence_grid": {
        "canonical": {"character": 0.9, "setting": 0.5, "challenge": 0.2, "outcome": 0.3,
                      "overall": 0.475, "moral_delivery": 0.62, "n": 50},
        "seen-template": {"character": 0.88, "setting": 0.48, "challenge": 0.18, "outcome": 0.28,
                          "overall": 0.455, "moral_delivery": 0.6, "n": 50},
        "held-out-template": {"character": 0.8, "setting": 0.44, "challenge": 0.16, "outcome": 0.26,
                              "overall": 0.415, "moral_delivery": 0.55, "n": 50},
    },
    "provenance": {"checkpoint_sha": "abc123def456", "tokenizer_sha": "def789", "seed": 0, "versions": {"torch": "2.11"}},
}


def test_write_eval_report_renders_tables(tmp_path):
    path = write_eval_report(tmp_path, METRICS)
    assert path == tmp_path / "eval_report.md"
    md = path.read_text()
    assert "4.249" in md                     # perplexity value
    assert "distinct-1" in md and "0.42" in md
    # adherence grid: one row per family, in canonical/seen/held-out order
    assert md.index("canonical") < md.index("seen-template") < md.index("held-out-template")
    assert "held-out-template" in md and "0.415" in md
    assert "abc123def456"[:12] in md         # provenance checkpoint sha (prefix)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tinyfables.report'`.

- [ ] **Step 3: Write `src/tinyfables/report.py`**

```python
"""Render the eval metrics dict into a report-usable Markdown document. Torch-free;
the machine-readable truth is eval_metrics.json + manifest.json (this is the human
view). Families are shown in a fixed canonical/seen/held-out order for the grid."""

from __future__ import annotations

from pathlib import Path

_FAMILY_ORDER = ("canonical", "seen-template", "held-out-template")


def write_eval_report(out_dir, metrics) -> Path:
    p = metrics["perplexity"]
    g = metrics["generation"]
    grid = metrics["adherence_grid"]
    prov = metrics["provenance"]
    lines = [
        "# Eval report",
        "",
        f"**Fable-token perplexity:** {p['perplexity']:.3f}  "
        f"(loss {p['fable_token_loss']:.4f}, {p['n_rows']} rows)",
        "",
        "## Generation quality",
        "",
        "| metric | value |",
        "|---|---|",
        f"| specs | {g['n_specs']} |",
        f"| distinct-1 | {g['distinct_1']:.3f} |",
        f"| distinct-2 | {g['distinct_2']:.3f} |",
        f"| repetition-4 | {g['repetition_4']:.3f} |",
        f"| length mean/median | {g['length']['mean']}/{g['length']['median']} |",
        f"| length min/max | {g['length']['min']}/{g['length']['max']} |",
        f"| moral delivery | {g['moral_delivery_rate']:.3f} (thr {g['moral_threshold']}) |",
        "",
        "## Adherence grid (verbatim element presence, by prompt family)",
        "",
        "| family | character | setting | challenge | outcome | overall | moral | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fam in _FAMILY_ORDER:
        if fam in grid:
            a = grid[fam]
            lines.append(
                f"| {fam} | {a['character']:.2f} | {a['setting']:.2f} | {a['challenge']:.2f} "
                f"| {a['outcome']:.2f} | {a['overall']:.2f} | {a['moral_delivery']:.2f} | {a['n']} |"
            )
    lines += [
        "",
        "## Provenance",
        f"- checkpoint: `{prov['checkpoint_sha'][:12]}`",
        f"- tokenizer: `{prov['tokenizer_sha'][:12]}`",
        f"- seed: {prov['seed']}",
        f"- versions: {prov['versions']}",
        "",
    ]
    out = Path(out_dir) / "eval_report.md"
    out.write_text("\n".join(lines))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/report.py tests/test_report.py
git commit -m "feat: eval report Markdown export (issue 05)"
```

---

### Task 6: The `evaluate` stage

**Files:**
- Create: `src/tinyfables/stages/evaluate.py`
- Modify: `src/tinyfables/stages/__init__.py` (register `evaluate`)
- Create: `configs/eval_toy.yaml`, `configs/eval_full.yaml`
- Modify: `docs/design.md` (Implementation (issue 05) note)
- Test: `tests/test_evaluate_stage.py`

**Interfaces:**
- Consumes: `EvalConfig`; `GPT.from_pretrained`; `Tokenizer`; `iter_rows`; `parse_canonical_prompt`, `load_bank`, `render_paraphrase`, `ParaphraseBank`; `render_canonical_prompt`, `FableSpec`; `generate_fable`; `fable_token_perplexity`; `element_adherence`, `distinct_n`, `repetition_rate`, `length_stats`; `extract_moral`, `naive_regex_moral`, `moral_delivered`; `write_eval_report`; `stage.write_manifest`, `stage.sha256_file`.
- Produces:
  - `run(cfg: EvalConfig, out_dir: Path) -> None` — writes `eval_metrics.json`, `eval_report.md`, `moral_calibration.jsonl`, then `manifest.json` LAST.
  - `REGISTRY["evaluate"] = (EvalConfig, "tinyfables.stages.evaluate")`.
  - `moral_calibration.jsonl` rows: `{"family":"canonical","prompt_index":int,"requested_moral":str,"ours":str|null,"naive":str|null,"fable":str}` — the hand-labeling worksheet (Track B adds `ours_ok`/`naive_ok`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate_stage.py`:

```python
import json
from pathlib import Path

from tinyfables.config import EvalConfig, PretrainConfig, PrepConfig, SourceSpec, TokenizerConfig
from tinyfables.stages import REGISTRY
from tinyfables.stages import evaluate as evaluate_stage
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import pretrain as pretrain_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
BANK = str(Path(__file__).parents[1] / "configs" / "paraphrases.yaml")
SRC = SourceSpec(jsonl_path=FIXTURE)


def _trained_checkpoint(tmp_path):
    tok = tmp_path / "tok"; prep = tmp_path / "prep"; ckpt = tmp_path / "ckpt"
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), tok)
    prep_stage.run(PrepConfig(source=SRC, tokenizer_dir=str(tok), window=512, seed=0), prep)
    pretrain_stage.run(PretrainConfig(prep_dir=str(prep), tokenizer_dir=str(tok),
                       n_layer=2, n_head=2, d_model=64, n_ctx=512, batch_size=4,
                       steps=4, warmup_steps=0, seed=0, device="cpu"), ckpt)
    return tok, ckpt


def _eval_cfg(tok, ckpt, **kw):
    base = dict(checkpoint=str(ckpt), tokenizer_dir=str(tok), source=SRC,
                paraphrase_bank=BANK, n_ctx=512, n_perplexity_rows=4, n_generations=2,
                max_new_tokens=24, min_new_tokens=8, top_k=10, seed=0, device="cpu")
    base.update(kw)
    return EvalConfig(**base)


def test_evaluate_registered():
    assert REGISTRY["evaluate"] == (EvalConfig, "tinyfables.stages.evaluate")


def test_evaluate_writes_metrics_report_and_manifest(tmp_path):
    tok, ckpt = _trained_checkpoint(tmp_path)
    out = tmp_path / "eval"
    evaluate_stage.run(_eval_cfg(tok, ckpt), out)
    assert {"eval_metrics.json", "eval_report.md", "moral_calibration.jsonl", "manifest.json"} <= {
        p.name for p in out.iterdir()}
    m = json.loads((out / "eval_metrics.json").read_text())
    import math
    assert math.isfinite(m["perplexity"]["perplexity"])
    # robustness grid has all three families when a bank is given
    assert set(m["adherence_grid"]) == {"canonical", "seen-template", "held-out-template"}
    assert m["adherence_grid"]["canonical"]["n"] == 2
    assert 0.0 <= m["generation"]["moral_delivery_rate"] <= 1.0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "evaluate"
    # calibration worksheet has one row per canonical generation with both extractions
    cal = [json.loads(l) for l in (out / "moral_calibration.jsonl").read_text().splitlines()]
    assert len(cal) == 2 and set(cal[0]) >= {"requested_moral", "ours", "naive", "fable"}


def test_evaluate_is_deterministic(tmp_path):
    tok, ckpt = _trained_checkpoint(tmp_path)
    a = tmp_path / "a"; b = tmp_path / "b"
    evaluate_stage.run(_eval_cfg(tok, ckpt), a)
    evaluate_stage.run(_eval_cfg(tok, ckpt), b)
    assert json.loads((a / "eval_metrics.json").read_text()) == json.loads((b / "eval_metrics.json").read_text())


def test_evaluate_without_bank_is_canonical_only(tmp_path):
    tok, ckpt = _trained_checkpoint(tmp_path)
    out = tmp_path / "nb"
    evaluate_stage.run(_eval_cfg(tok, ckpt, paraphrase_bank=None), out)
    m = json.loads((out / "eval_metrics.json").read_text())
    assert set(m["adherence_grid"]) == {"canonical"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluate_stage.py -v`
Expected: FAIL — `KeyError: 'evaluate'` / `ModuleNotFoundError: ...stages.evaluate`.

- [ ] **Step 3: Write `src/tinyfables/stages/evaluate.py`**

```python
"""Evaluate stage: score any checkpoint into a report-ready metrics document —
fable-token perplexity, the canonical/seen/held-out spec-adherence robustness
grid, moral delivery, distinct-n/repetition, and length stats — plus a moral-
extraction hand-labeling worksheet. Deterministic given seed + eval set. This is
how any two checkpoints are compared for the rest of the project.

Held-out paraphrase templates are USED here (they are eval-only, never trained on)
to render the held-out column of the grid — robustness to unseen phrasing measured,
not asserted."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import EvalConfig
from tinyfables.data import iter_rows
from tinyfables.eval_metrics import distinct_n, element_adherence, length_stats, repetition_rate
from tinyfables.generate import generate_fable
from tinyfables.model import GPT
from tinyfables.moral import extract_moral, moral_delivered, moral_similarity, naive_regex_moral
from tinyfables.paraphrases import load_bank, parse_canonical_prompt, render_paraphrase
from tinyfables.perplexity import fable_token_perplexity
from tinyfables.prompts import FableSpec, render_canonical_prompt
from tinyfables.report import write_eval_report
from tinyfables.stage import _versions, sha256_file, write_manifest

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
    return FableSpec(**{f: getattr(parsed, f) for f in _ELEMENTS},
                     age_range=parsed.age_range, word_count=parsed.word_count)


def _grid_family(records) -> dict:
    """Aggregate per-family: mean verbatim presence per element, overall mean, and
    moral-delivery rate over the family's generations."""
    n = len(records)
    per_el = {}
    for el in ("character", "setting", "challenge", "outcome"):
        vals = [r["adherence"][el] for r in records if el in r["adherence"]]
        per_el[el] = round(sum(vals) / len(vals), 4) if vals else 0.0
    overall = round(sum(per_el.values()) / len(per_el), 4)
    moral = round(sum(r["moral"] for r in records) / n, 4) if n else 0.0
    return {**per_el, "overall": overall, "moral_delivery": moral, "n": n}


def run(cfg: EvalConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    model = GPT.from_pretrained(cfg.checkpoint).to(device).eval()
    bank = load_bank(cfg.paraphrase_bank) if cfg.paraphrase_bank else None

    # materialize enough val rows for both perplexity and spec sampling
    need = max(cfg.n_perplexity_rows, cfg.n_generations * 4)
    rows = list(itertools.islice(iter_rows(cfg.source, cfg.seed), need))

    mean_loss, ppl = fable_token_perplexity(model, tok, rows, cfg.n_ctx, device, cfg.n_perplexity_rows)

    specs: list[FableSpec] = []
    for r in rows:
        spec = _spec_from_prompt(r["prompt"])
        if spec is not None:
            specs.append(spec)
        if len(specs) >= cfg.n_generations:
            break

    families: dict[str, list] = {"canonical": []}
    if bank is not None:
        families["seen-template"] = []
        families["held-out-template"] = []
    canonical_fables: list[str] = []
    calibration: list[dict] = []
    rng = random.Random(cfg.seed)

    def _gen(prompt_text: str, idx: int) -> str:
        return generate_fable(model, tok, prompt_text, max_new_tokens=cfg.max_new_tokens,
                              min_new_tokens=cfg.min_new_tokens, do_sample=True,
                              temperature=cfg.temperature, top_k=cfg.top_k,
                              seed=cfg.seed + idx, device=device)

    for idx, spec in enumerate(specs):
        renderings = [("canonical", render_canonical_prompt(spec))]
        if bank is not None:
            renderings.append(("seen-template", render_paraphrase(bank.choose_seen(rng), spec)))
            renderings.append(("held-out-template", render_paraphrase(rng.choice(bank.held_out_templates), spec)))
        for family, prompt_text in renderings:
            fable = _gen(prompt_text, idx)
            families[family].append({
                "adherence": element_adherence(fable, spec),
                "moral": moral_delivered(fable, spec.moral, cfg.moral_threshold),
            })
            if family == "canonical":
                canonical_fables.append(fable)
                calibration.append({
                    "family": "canonical", "prompt_index": idx,
                    "requested_moral": spec.moral,
                    "ours": extract_moral(fable), "naive": naive_regex_moral(fable),
                    "fable": fable,
                })

    grid = {fam: _grid_family(recs) for fam, recs in families.items()}
    delivered = [moral_similarity(extract_moral(f), s.moral) >= cfg.moral_threshold
                 for f, s in zip(canonical_fables, specs)]
    joined = "\n".join(canonical_fables)
    metrics = {
        "perplexity": {"fable_token_loss": round(mean_loss, 4), "perplexity": round(ppl, 4),
                       "n_rows": min(cfg.n_perplexity_rows, len(rows))},
        "generation": {
            "n_specs": len(canonical_fables),
            "distinct_1": round(distinct_n(joined, 1), 4),
            "distinct_2": round(distinct_n(joined, 2), 4),
            "repetition_4": round(repetition_rate(joined, 4), 4),
            "length": length_stats(canonical_fables),
            "moral_delivery_rate": round(sum(delivered) / len(delivered), 4) if delivered else 0.0,
            "moral_threshold": cfg.moral_threshold,
        },
        "adherence_grid": grid,
        "provenance": {
            "checkpoint_sha": sha256_file(Path(cfg.checkpoint) / "model.safetensors"),
            "tokenizer_sha": sha256_file(Path(cfg.tokenizer_dir) / "tokenizer.json"),
            "seed": cfg.seed,
            "versions": _versions(),
        },
    }

    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    cal_path = out_dir / "moral_calibration.jsonl"
    cal_path.write_text("".join(json.dumps(c) + "\n" for c in calibration))
    report_path = write_eval_report(out_dir, metrics)

    write_manifest(
        out_dir, "evaluate", cfg,
        [metrics_path, report_path, cal_path],
        inputs={
            "model.safetensors": Path(cfg.checkpoint) / "model.safetensors",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
```

- [ ] **Step 4: Register the stage in `src/tinyfables/stages/__init__.py`**

Replace the import + registry with:

```python
from tinyfables.config import BenchmarkConfig, EvalConfig, PrepConfig, PretrainConfig, TokenizerConfig

REGISTRY: dict[str, tuple[type, str]] = {
    "tokenizer": (TokenizerConfig, "tinyfables.stages.tokenizer"),
    "prep": (PrepConfig, "tinyfables.stages.prep"),
    "benchmark": (BenchmarkConfig, "tinyfables.stages.benchmark"),
    "pretrain": (PretrainConfig, "tinyfables.stages.pretrain"),
    "evaluate": (EvalConfig, "tinyfables.stages.evaluate"),
}
```

- [ ] **Step 5: Create `configs/eval_toy.yaml`**

```yaml
# Toy eval over the fixture (CPU, seconds) — smoke-tests the whole suite.
checkpoint: runs/pretrain_toy
tokenizer_dir: runs/tokenizer_toy
source:
  jsonl_path: tests/fixtures/tiny_corpus.jsonl
paraphrase_bank: configs/paraphrases.yaml
n_ctx: 512
n_perplexity_rows: 8
n_generations: 4
max_new_tokens: 48
min_new_tokens: 8
top_k: 20
seed: 0
device: cpu
```

- [ ] **Step 6: Create `configs/eval_full.yaml`**

```yaml
# Real eval (issue 05, Track B): evaluate the Base Model on the val split.
# checkpoint is set at run time to the downloaded/loaded Base Model dir (or a
# local snapshot of congthanh991/tinyfables-13m-base). Held-out templates render
# the robustness grid's unseen-phrasing column.
checkpoint: runs/base_model
tokenizer_dir: runs/tokenizer_full
source:
  hf_dataset: klusai/ds-tf1-en-3m
  hf_split: validation
  max_rows: 2000
paraphrase_bank: configs/paraphrases.yaml
n_ctx: 1024
n_perplexity_rows: 500
n_generations: 50
max_new_tokens: 320
min_new_tokens: 80
temperature: 0.9
top_k: 50
moral_threshold: 0.3
seed: 0
device: auto
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_evaluate_stage.py -v`
Expected: PASS (4 passed).

- [ ] **Step 8: Add the "Implementation (issue 05)" note to `docs/design.md`**

Insert directly after the `### Implementation (issue 04)` block's last bullet (before `## Build vs buy`):

```markdown
### Implementation (issue 05)

- **`evaluate` stage.** One command (`python -m tinyfables run evaluate --config … --out …`)
  scores any checkpoint into `eval_metrics.json` (machine), `eval_report.md` (report tables),
  and `moral_calibration.jsonl` (hand-labeling worksheet), manifest last. Deterministic given
  seed + eval set: perplexity has no sampling; generation is seeded per spec.
- **Fable-token perplexity** (`perplexity.py`): completion-only next-token loss over the val
  slice (prompt masked), so it means perplexity *of fables*; exp of the mean.
- **Adherence robustness grid** (`eval_metrics.element_adherence`): verbatim (case-insensitive)
  presence of each requested Element (character/setting/challenge/outcome) in the generated
  fable — a mechanical, comparable proxy. Each spec is rendered under canonical + a seen
  template + a held-out template and adherence is grouped by family; held-out templates are
  eval-only (used here, never trained on).
- **Moral delivery** (`moral.py`): `extract_moral` prefers the last `**bold**` segment but
  falls back to the trailing sentence (the issue-04 Base Model rarely emits the marker), then
  fuzzy-matches (SequenceMatcher) to the requested moral above a threshold. `naive_regex_moral`
  is the keyword-regex baseline the extraction must beat; precision is hand-validated in Track B.
- **Distinct-n / repetition / length** round out the generation-quality view. No new deps
  (stdlib `re`/`difflib`/`statistics`).
```

- [ ] **Step 9: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — all green.

- [ ] **Step 10: Commit**

```bash
git add src/tinyfables/stages/evaluate.py src/tinyfables/stages/__init__.py configs/eval_toy.yaml configs/eval_full.yaml docs/design.md tests/test_evaluate_stage.py
git commit -m "feat: evaluate stage — perplexity + adherence grid + moral delivery + report (issue 05)"
```

---

## Track A completion: merge to main

- [ ] Run the full suite on the branch: `.venv/bin/pytest -q` → all pass.
- [ ] Request code review (superpowers:requesting-code-review) over the branch diff vs `main`; address Critical/Important findings.
- [ ] Merge with history preserved:

```bash
git checkout main
git merge --no-ff feat/issue-05-eval-suite -m "Merge feat/issue-05-eval-suite: mechanical eval suite (issue 05 Track A)"
git push origin main
```

Do NOT tick issue 05 yet — the moral-extraction hand-validation (AC-2) and the real Base Model eval are Track B.

---

## Track B: real eval + moral-extraction hand-validation (operational)

> Runs the suite against the real Base Model and does the one manual deliverable — moral-extraction precision hand-validated on ~50 samples vs the naive-regex baseline. Runnable locally (download the model from the Hub; 14M params, CPU-feasible though generation is slow) or on Colab (faster). Record numbers in `docs/design.md` immediately.

- [ ] **B1 — Get the Base Model + tokenizer locally (or on Colab).** Download a snapshot:
  `python -c "from huggingface_hub import snapshot_download as d; d('congthanh991/tinyfables-13m-base', local_dir='runs/base_model'); d('congthanh991/tinyfables-tokenizer', local_dir='runs/tokenizer_full')"`. (On Colab, reuse the existing `runs/tokenizer_full` + `base_out`.) Point `configs/eval_full.yaml`'s `checkpoint`/`tokenizer_dir` at those dirs.

- [ ] **B2 — Run the real eval.** `python -m tinyfables run evaluate --config configs/eval_full.yaml --out runs/eval_base` (streams the val split; needs network). Read `runs/eval_base/eval_report.md`: record fable-token perplexity, the canonical/seen/held-out adherence grid, moral-delivery rate, distinct-n, and length stats into `docs/design.md`. Sanity-check the held-out column is ≤ the canonical column (robustness cost of unseen phrasing).

- [ ] **B3 — Hand-validate moral extraction (AC-2).** Open `runs/eval_base/moral_calibration.jsonl` (~50 canonical generations, each with `ours` = our extraction and `naive` = the baseline). For each row, judge whether `ours` and `naive` correctly capture the fable's actual moral; record two boolean lists. Compute precision with `tinyfables.moral.extraction_precision(ours_labels)` and `extraction_precision(naive_labels)`. **Requirement:** our precision must beat the naive baseline (design: naive misses ~10%). Record both numbers, the sample size, and the chosen `moral_threshold` in `docs/design.md`. If ours does NOT beat naive, tune `extract_moral` (e.g., prefer the last 1–2 sentences, or strip trailing dialogue) and re-validate — this is the calibration the AC demands.

- [ ] **B4 — Close out.** Record the final eval numbers + moral-extraction precision (ours vs naive) in `docs/design.md`; tick issue 05 in `docs/issues/README.md`. Commit + push:

```bash
git add docs/design.md docs/issues/README.md
git commit -m "docs: record real Base Model eval + moral-extraction precision; tick issue 05 done"
```

**Done when:** one command evals any checkpoint into the metrics document (deterministic); moral-extraction precision hand-validated on ~50 samples, recorded, and beats the naive baseline; the adherence grid splits by prompt family; the metrics tables export with provenance; the suite runs at toy scale in the test suite; issue 05 ticked.

---

## Self-Review

**1. Spec coverage** (issue 05 acceptance criteria):
- AC "one command evaluates any checkpoint → deterministic metrics document" → Task 6 (`evaluate` stage, `eval_metrics.json`) + `test_evaluate_is_deterministic`. ✓
- AC "moral-extraction precision hand-validated on ~50 samples, beats naive baseline" → Task 2 (`extract_moral`, `naive_regex_moral`, `extraction_precision`) + Task 6 (`moral_calibration.jsonl` worksheet) + Track B B3. ✓
- AC "adherence grid splits by prompt family when tags present" → Task 6 (three-family rendering + `_grid_family`) + `test_evaluate_writes...` (asserts all three families) + `test_evaluate_without_bank_is_canonical_only`. ✓
- AC "metrics tables export in report-usable form with provenance" → Task 5 (`report.py`) + Task 6 (provenance block + manifest). ✓
- AC "suite runs at toy scale inside the test suite" → Task 6 toy tests (fixture + tiny checkpoint). ✓
- Metrics: perplexity (Task 3), verbatim adherence (Task 1), moral delivery (Task 2), distinct-n/repetition (Task 1), length (Task 1) — all present. ✓
- Design constraints honored: single-band specs (`_spec_from_prompt` carries age_range 4-7); moral fuzzy fallback (issue-04 finding); manifest-last; lazy registry (torch-free metric modules import no torch at top level; `evaluate` registered by string). ✓

**2. Placeholder scan:** No "TBD/handle edge cases/similar to Task N" in code steps — every code step is complete. Track B's "record in design.md" values are produced by the operational run (as for issue 04) — that track is a runbook, not TDD, matching the one manual AC (hand-validation).

**3. Type consistency:** `element_adherence(fable, spec) -> dict[str,bool]` (Task 1) is consumed by `_grid_family` and the calibration in Task 6 with those keys. `extract_moral`/`naive_regex_moral`/`moral_similarity`/`moral_delivered`/`extraction_precision` (Task 2) are used with those exact signatures in Task 6 + Track B. `fable_token_perplexity(model, tokenizer, rows, n_ctx, device, max_rows) -> (mean_loss, ppl)` (Task 3) matches the Task 6 call. `EvalConfig` fields (Task 4) match every read in Task 6 and both eval configs. The metrics dict shape produced in Task 6 matches exactly what `write_eval_report` (Task 5) consumes (`perplexity`/`generation`/`adherence_grid`/`provenance` with the named sub-keys). `_FAMILY_ORDER` in Task 5 matches the family names produced in Task 6 (`canonical`/`seen-template`/`held-out-template`, the `paraphrases.FAMILIES` names). No drift found.
