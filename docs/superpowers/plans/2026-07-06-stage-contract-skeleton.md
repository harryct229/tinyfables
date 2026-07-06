# Stage-Contract Skeleton (Issue 01) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the pipeline-stage contract (config in → artifacts out, deterministic, hash-stable) and prove it with the first two real stages: tokenizer training (8k BPE + compression report) and data prep (packed token shards with loss masks + prep summary).

**Architecture:** A `src/tinyfables/` package where every stage is a function `run(cfg, out_dir)` registered in a stage registry and invoked via one CLI (`python -m tinyfables run <stage> --config <yaml> --out <dir>`). Every stage writes its artifacts plus a `manifest.json` containing the config echo and SHA-256 of each artifact — determinism is tested by running a stage twice and comparing artifact hashes. Tests run everything at toy scale on a committed 24-row fixture corpus, offline, on CPU.

**Tech Stack:** Python ≥3.10, `tokenizers` (BPE training), `numpy` (shards), `pyyaml` (configs), `datasets` (HF streaming source — network-marked tests only), `pytest`.

## Global Constraints

- Python ≥ 3.10; package lives under `src/tinyfables/`; install with `pip install -e ".[dev]"`.
- Special tokens are exactly `<|endoftext|>` (id 0) and `<|pad|>` (id 1), reserved at tokenizer creation (ADR-0002). They cannot be added later.
- Full-scale vocab is 8192; toy configs may use smaller vocab (fixture corpus can't fill 8192).
- Shards: `tokens.bin` is raw little-endian uint16, `mask.bin` is raw uint8 (1 = loss token), both trimmed to a multiple of `window`; full-scale window is 1024.
- Loss mask covers fable tokens **and** the end-of-text token; prompt tokens are masked out (mask 0).
- Prompt and fable are tokenized **separately** and concatenated (no separator token, no merged encoding across the boundary) — the generation contract in issue 02 must encode prompts the same way.
- Determinism: same config + seed ⇒ byte-identical artifacts. Set `TOKENIZERS_PARALLELISM=false` inside stages.
- Tests run offline by default: `pytest` deselects `network`-marked tests via `addopts`. The full offline suite must finish in under 2 minutes on CPU (issue 01 AC).
- No pipeline logic in notebooks (there are no notebooks in this slice).
- Every commit message ends with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Package scaffold and frozen config loading

**Files:**
- Create: `pyproject.toml`
- Create: `src/tinyfables/__init__.py`
- Create: `src/tinyfables/constants.py`
- Create: `src/tinyfables/config.py`
- Test: `tests/test_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `tinyfables.constants.EOT: str`, `PAD: str`, `SPECIALS: list[str]`; dataclasses `tinyfables.config.SourceSpec(jsonl_path: str|None, hf_dataset: str|None, hf_split: str, max_rows: int|None)`, `TokenizerConfig(source: SourceSpec, vocab_size: int, seed: int, compare_gpt2: bool)`, `PrepConfig(source: SourceSpec, tokenizer_dir: str, window: int, seed: int)`; loader `load_config(path: str|Path, cls: type[T]) -> T` that rejects unknown keys.

- [ ] **Step 1: Create the package skeleton and project metadata**

`pyproject.toml`:

```toml
[project]
name = "tinyfables"
version = "0.1.0"
description = "From-scratch moral-fable GPT with RLAIF alignment (course project)"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "tokenizers>=0.15",
    "pyyaml>=6.0",
    "datasets>=2.19",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
gpt2 = ["transformers>=4.40"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["network: requires internet access (deselected by default)"]
addopts = "-m 'not network'"
```

`src/tinyfables/__init__.py`:

```python
"""TinyFables: from-scratch fable GPT aligned with AI feedback."""

__version__ = "0.1.0"
```

`src/tinyfables/constants.py`:

```python
"""Vocabulary constants. Special tokens are reserved at tokenizer creation
(ADR-0002) and can never be added later — do not change these strings."""

EOT = "<|endoftext|>"
PAD = "<|pad|>"
SPECIALS = [EOT, PAD]  # trained first, so ids are 0 and 1
```

Append to `.gitignore`:

```
runs/
*.egg-info/
```

- [ ] **Step 2: Create the venv and install**

Run:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -q -e ".[dev]"
```

Expected: ends without error; `pytest --version` prints a version.

- [ ] **Step 3: Write the failing config tests**

`tests/test_config.py`:

```python
import dataclasses

import pytest

from tinyfables.config import PrepConfig, SourceSpec, TokenizerConfig, load_config


def write_yaml(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_load_tokenizer_config(tmp_path):
    p = write_yaml(
        tmp_path,
        "source:\n  jsonl_path: corpus.jsonl\n  max_rows: 10\n"
        "vocab_size: 512\nseed: 7\ncompare_gpt2: false\n",
    )
    cfg = load_config(p, TokenizerConfig)
    assert cfg.source.jsonl_path == "corpus.jsonl"
    assert cfg.source.max_rows == 10
    assert cfg.vocab_size == 512
    assert cfg.seed == 7


def test_load_prep_config(tmp_path):
    p = write_yaml(
        tmp_path,
        "source:\n  jsonl_path: corpus.jsonl\n"
        "tokenizer_dir: runs/tok\nwindow: 256\nseed: 0\n",
    )
    cfg = load_config(p, PrepConfig)
    assert cfg.tokenizer_dir == "runs/tok"
    assert cfg.window == 256


def test_configs_are_frozen(tmp_path):
    p = write_yaml(tmp_path, "source:\n  jsonl_path: c.jsonl\nvocab_size: 64\n")
    cfg = load_config(p, TokenizerConfig)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.vocab_size = 999


def test_unknown_key_rejected(tmp_path):
    p = write_yaml(tmp_path, "source:\n  jsonl_path: c.jsonl\nvocabsize: 64\n")
    with pytest.raises(KeyError):
        load_config(p, TokenizerConfig)


def test_source_requires_exactly_one_backend():
    with pytest.raises(ValueError):
        SourceSpec()  # neither jsonl nor hf
    with pytest.raises(ValueError):
        SourceSpec(jsonl_path="a.jsonl", hf_dataset="klusai/ds-tf1-en-3m")  # both
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'tinyfables.config'`

- [ ] **Step 5: Implement the config module**

`src/tinyfables/config.py`:

```python
"""Frozen dataclass configs loaded from YAML. Unknown keys are errors so a
typo in an experiment config fails loudly instead of silently using defaults."""

from __future__ import annotations

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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/tinyfables tests/test_config.py .gitignore
git commit -m "feat: package scaffold with frozen YAML configs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Committed tiny fixture corpus

**Files:**
- Create: `tests/fixtures/make_tiny_corpus.py`
- Create: `tests/fixtures/tiny_corpus.jsonl` (generated by the script, then committed)
- Test: `tests/test_fixture_corpus.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/fixtures/tiny_corpus.jsonl` — 24 JSONL rows, each `{"prompt": str, "fable": str}`, mimicking the real dataset's Canonical Prompt shape (bulleted Elements) and fable shape (story ending with a `**The Moral:** …` line).

- [ ] **Step 1: Write the failing fixture test**

`tests/test_fixture_corpus.py`:

```python
import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl"


def test_fixture_corpus_shape():
    rows = [json.loads(l) for l in FIXTURE.read_text().splitlines() if l.strip()]
    assert len(rows) == 24
    for r in rows:
        assert set(r) == {"prompt", "fable"}
        assert r["prompt"].startswith("Create a fable based on the following elements")
        assert "- Main Character:" in r["prompt"]
        assert "**The Moral:**" in r["fable"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fixture_corpus.py -v`
Expected: FAIL with `FileNotFoundError` (no tiny_corpus.jsonl yet)

- [ ] **Step 3: Write the deterministic generator**

`tests/fixtures/make_tiny_corpus.py`:

```python
"""Deterministically generate the committed 24-row toy corpus.
Run from the repo root:  python tests/fixtures/make_tiny_corpus.py"""

import itertools
import json
from pathlib import Path

CHARACTERS = ["a shy octopus", "a stubborn raccoon", "a persuasive firefly", "a patient tortoise"]
SETTINGS = ["a quiet tide pool", "a misty marsh", "a deep canyon"]
MORALS = ["courage grows by small steps", "timely help earns lasting loyalty"]

PROMPT_TMPL = (
    "Create a fable based on the following elements. Weave them naturally into a story:\n"
    "- Main Character: {character}\n"
    "- Setting: {setting}\n"
    "- Challenge: doubting oneself\n"
    "- Outcome: a friend helps just in time\n"
    "- Teaching: {moral}\n"
    "Keep it age-appropriate for ages 4-7 and about 60 words."
)

FABLE_TMPL = (
    "Once, in {setting}, there lived {character} who doubted every step. "
    "One grey morning a storm rolled in, and the little one froze with fear. "
    "A friend arrived just in time, and together they found the way home. "
    "From that day on, they practiced one small brave thing each morning.\n\n"
    "**The Moral:** {moral}"
)


def main() -> None:
    out = Path(__file__).parent / "tiny_corpus.jsonl"
    rows = []
    for character, setting, moral in itertools.product(CHARACTERS, SETTINGS, MORALS):
        rows.append(
            {
                "prompt": PROMPT_TMPL.format(character=character, setting=setting, moral=moral),
                "fable": FABLE_TMPL.format(character=character, setting=setting, moral=moral),
            }
        )
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the fixture and verify the test passes**

Run:

```bash
python tests/fixtures/make_tiny_corpus.py && pytest tests/test_fixture_corpus.py -v
```

Expected: `wrote 24 rows to …/tiny_corpus.jsonl`, then 1 passed

- [ ] **Step 5: Commit (fixture file included — it is a committed artifact, not generated at test time)**

```bash
git add tests/fixtures/make_tiny_corpus.py tests/fixtures/tiny_corpus.jsonl tests/test_fixture_corpus.py
git commit -m "feat: committed 24-row toy fixture corpus + generator

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Stage manifest and hashing

**Files:**
- Create: `src/tinyfables/stage.py`
- Test: `tests/test_stage.py`

**Interfaces:**
- Consumes: config dataclasses from Task 1.
- Produces: `tinyfables.stage.sha256_file(path: Path) -> str`; `tinyfables.stage.write_manifest(out_dir: Path, stage: str, config_obj, artifacts: list[Path]) -> Path` writing `manifest.json` with keys `stage` (str), `config` (nested dict), `artifacts` (dict filename → sha256 hex), `created_unix` (int). Determinism comparisons across runs use the `artifacts` dict only (timestamps differ).

- [ ] **Step 1: Write the failing tests**

`tests/test_stage.py`:

```python
import json

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.stage import sha256_file, write_manifest


def test_sha256_is_stable(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello fables")
    assert sha256_file(f) == sha256_file(f)
    g = tmp_path / "b.bin"
    g.write_bytes(b"hello fables!")
    assert sha256_file(f) != sha256_file(g)


def test_write_manifest(tmp_path):
    art = tmp_path / "tokenizer.json"
    art.write_text("{}")
    cfg = TokenizerConfig(source=SourceSpec(jsonl_path="x.jsonl"), vocab_size=64)
    path = write_manifest(tmp_path, "tokenizer", cfg, [art])
    m = json.loads(path.read_text())
    assert m["stage"] == "tokenizer"
    assert m["config"]["vocab_size"] == 64
    assert m["config"]["source"]["jsonl_path"] == "x.jsonl"
    assert list(m["artifacts"]) == ["tokenizer.json"]
    assert len(m["artifacts"]["tokenizer.json"]) == 64  # sha256 hex
    assert isinstance(m["created_unix"], int)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.stage'`

- [ ] **Step 3: Implement**

`src/tinyfables/stage.py`:

```python
"""The stage contract: every stage writes its artifacts plus a manifest.json
recording the config echo and a SHA-256 per artifact. Two runs of the same
stage with the same config+seed must produce identical `artifacts` maps."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, stage: str, config_obj, artifacts: list[Path]) -> Path:
    manifest = {
        "stage": stage,
        "config": dataclasses.asdict(config_obj),
        "artifacts": {p.name: sha256_file(p) for p in sorted(artifacts)},
        "created_unix": int(time.time()),
    }
    out = out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stage.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/stage.py tests/test_stage.py
git commit -m "feat: stage manifest with per-artifact sha256

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Source reader (JSONL + HF streaming)

**Files:**
- Create: `src/tinyfables/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `SourceSpec` from Task 1.
- Produces: `tinyfables.data.read_rows(src: SourceSpec, seed: int) -> list[dict]` — each row exactly `{"prompt": str, "fable": str}`, shuffled deterministically by `seed`, truncated to `src.max_rows` when set. The HF branch streams `src.hf_dataset` (network) — covered by a `network`-marked test only.

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:

```python
from pathlib import Path

import pytest

from tinyfables.config import SourceSpec
from tinyfables.data import read_rows

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def test_reads_all_rows_with_expected_keys():
    rows = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)
    assert len(rows) == 24
    assert all(set(r) == {"prompt", "fable"} for r in rows)


def test_shuffle_is_deterministic_and_seed_sensitive():
    a = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)
    b = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)
    c = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=1)
    assert a == b
    assert a != c  # 24 rows: astronomically unlikely to collide


def test_max_rows_truncates():
    rows = read_rows(SourceSpec(jsonl_path=FIXTURE, max_rows=5), seed=0)
    assert len(rows) == 5


@pytest.mark.network
def test_hf_streaming_source_smoke():
    rows = read_rows(
        SourceSpec(hf_dataset="klusai/ds-tf1-en-3m", hf_split="train", max_rows=3), seed=0
    )
    assert len(rows) == 3
    assert all(set(r) == {"prompt", "fable"} for r in rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.data'` (network test deselected)

- [ ] **Step 3: Implement**

`src/tinyfables/data.py`:

```python
"""Source reading: local JSONL (fixtures, toy runs) or HF streaming (real runs).
Rows are reduced to the two fields the pipeline uses: prompt and fable."""

from __future__ import annotations

import json
import random

from tinyfables.config import SourceSpec


def read_rows(src: SourceSpec, seed: int) -> list[dict]:
    if src.jsonl_path is not None:
        with open(src.jsonl_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rng = random.Random(seed)
        rng.shuffle(rows)
        if src.max_rows is not None:
            rows = rows[: src.max_rows]
        return [{"prompt": r["prompt"], "fable": r["fable"]} for r in rows]

    from datasets import load_dataset  # imported lazily: network dependency

    ds = load_dataset(src.hf_dataset, split=src.hf_split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    rows = []
    for r in ds:
        rows.append({"prompt": r["prompt"], "fable": r["fable"]})
        if src.max_rows is not None and len(rows) >= src.max_rows:
            break
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: 3 passed, 1 deselected

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/data.py tests/test_data.py
git commit -m "feat: deterministic source reader (jsonl + hf streaming)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Tokenizer stage

**Files:**
- Create: `src/tinyfables/stages/__init__.py` (empty for now; registry lands in Task 7)
- Create: `src/tinyfables/stages/tokenizer.py`
- Test: `tests/test_tokenizer_stage.py`

**Interfaces:**
- Consumes: `TokenizerConfig`, `read_rows`, `SPECIALS`, `write_manifest`.
- Produces: `tinyfables.stages.tokenizer.run(cfg: TokenizerConfig, out_dir: Path) -> None` writing `tokenizer.json` (loadable via `tokenizers.Tokenizer.from_file`), `compression_report.json` (keys: `n_fables`, `vocab_size`, `ours_avg_tokens_per_fable`, `gpt2_avg_tokens_per_fable` — null unless `compare_gpt2`), and `manifest.json`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tokenizer_stage.py`:

```python
import json
from pathlib import Path

from tokenizers import Tokenizer

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.constants import EOT, PAD
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
CFG = TokenizerConfig(source=SourceSpec(jsonl_path=FIXTURE), vocab_size=512, seed=0)


def run_stage(tmp_path, name="tok"):
    out = tmp_path / name
    tokenizer_stage.run(CFG, out)
    return out


def test_specials_reserved_at_ids_0_and_1(tmp_path):
    out = run_stage(tmp_path)
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    assert tok.token_to_id(EOT) == 0
    assert tok.token_to_id(PAD) == 1


def test_round_trip_is_lossless(tmp_path):
    out = run_stage(tmp_path)
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    samples = [
        "Once, in a misty marsh, there lived a stubborn raccoon.",
        "**The Moral:** courage grows by small steps",
        "- Main Character: a shy octopus\n- Setting: a quiet tide pool",
    ]
    for text in samples:
        assert tok.decode(tok.encode(text).ids) == text


def test_vocab_size_bounded_by_config(tmp_path):
    out = run_stage(tmp_path)
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    assert tok.get_vocab_size() <= 512


def test_compression_report_shape(tmp_path):
    out = run_stage(tmp_path)
    report = json.loads((out / "compression_report.json").read_text())
    assert report["n_fables"] == 24
    assert report["ours_avg_tokens_per_fable"] > 0
    assert report["gpt2_avg_tokens_per_fable"] is None  # compare_gpt2 is false offline


def test_stage_is_hash_deterministic(tmp_path):
    m1 = json.loads((run_stage(tmp_path, "a") / "manifest.json").read_text())
    m2 = json.loads((run_stage(tmp_path, "b") / "manifest.json").read_text())
    assert m1["artifacts"] == m2["artifacts"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tokenizer_stage.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` on `tinyfables.stages`

- [ ] **Step 3: Implement**

`src/tinyfables/stages/__init__.py`:

```python
```

(empty file for now — the registry is added in Task 7)

`src/tinyfables/stages/tokenizer.py`:

```python
"""Tokenizer stage: train the byte-level BPE on the fable corpus and emit the
compression report that justifies the vocab choice (ADR-0002)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

from tinyfables.config import TokenizerConfig
from tinyfables.constants import SPECIALS
from tinyfables.data import read_rows
from tinyfables.stage import write_manifest


def run(cfg: TokenizerConfig, out_dir: Path) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # determinism over speed
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(cfg.source, cfg.seed)
    texts = [r["prompt"] + "\n" + r["fable"] for r in rows]

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        texts, vocab_size=cfg.vocab_size, min_frequency=2, special_tokens=SPECIALS
    )
    tok_path = out_dir / "tokenizer.json"
    tok.save(str(tok_path))

    report_path = out_dir / "compression_report.json"
    report = _compression_report(tok, [r["fable"] for r in rows], cfg)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    write_manifest(out_dir, "tokenizer", cfg, [tok_path, report_path])


def _compression_report(tok, fables: list[str], cfg: TokenizerConfig) -> dict:
    ours = [len(tok.encode(t).ids) for t in fables]
    report = {
        "n_fables": len(fables),
        "vocab_size": tok.get_vocab_size(),
        "ours_avg_tokens_per_fable": round(sum(ours) / len(ours), 2),
        "gpt2_avg_tokens_per_fable": None,
    }
    if cfg.compare_gpt2:
        from transformers import GPT2TokenizerFast  # optional dep (network on first use)

        gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
        counts = [len(gpt2(t)["input_ids"]) for t in fables]
        report["gpt2_avg_tokens_per_fable"] = round(sum(counts) / len(counts), 2)
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tokenizer_stage.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/stages tests/test_tokenizer_stage.py
git commit -m "feat: tokenizer stage with compression report

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Prep stage (packed shards with loss masks)

**Files:**
- Create: `src/tinyfables/stages/prep.py`
- Test: `tests/test_prep_stage.py`

**Interfaces:**
- Consumes: `PrepConfig`, `read_rows`, `EOT`, `write_manifest`, and a Task-5 tokenizer artifact directory (`cfg.tokenizer_dir`).
- Produces: `tinyfables.stages.prep.run(cfg: PrepConfig, out_dir: Path) -> None` writing `tokens.bin` (raw uint16), `mask.bin` (raw uint8, same length; 1 = loss token = fable or EOT), `prep_summary.json` (keys: `n_rows`, `window`, `n_windows`, `n_tokens_written`, `n_prompt_tokens_total`, `n_fable_tokens_total`, `loss_token_fraction`, `vocab_size`), and `manifest.json`. Total written length is `n_windows * window` (trailing partial window dropped). Prompt and fable are encoded separately, concatenated, EOT appended.

- [ ] **Step 1: Write the failing tests**

`tests/test_prep_stage.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from tinyfables.config import PrepConfig, SourceSpec, TokenizerConfig
from tinyfables.data import read_rows
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
SRC = SourceSpec(jsonl_path=FIXTURE)


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("tok")
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), out)
    return out


def prep_cfg(tok_dir):
    return PrepConfig(source=SRC, tokenizer_dir=str(tok_dir), window=256, seed=0)


def run_prep(tmp_path, tok_dir, name="prep"):
    out = tmp_path / name
    prep_stage.run(prep_cfg(tok_dir), out)
    return out


def test_shards_are_window_aligned_uint16(tmp_path, tok_dir):
    out = run_prep(tmp_path, tok_dir)
    summary = json.loads((out / "prep_summary.json").read_text())
    tokens = np.frombuffer((out / "tokens.bin").read_bytes(), dtype=np.uint16)
    mask = np.frombuffer((out / "mask.bin").read_bytes(), dtype=np.uint8)
    assert len(tokens) == len(mask) == summary["n_tokens_written"]
    assert summary["n_tokens_written"] == summary["n_windows"] * summary["window"]
    assert summary["n_windows"] > 0
    assert set(np.unique(mask)) <= {0, 1}


def test_mask_marks_prompt_zero_then_fable_one(tmp_path, tok_dir):
    out = run_prep(tmp_path, tok_dir)
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    first = read_rows(SRC, seed=0)[0]  # same seed => same order as the stage
    n_p = len(tok.encode(first["prompt"]).ids)
    n_f = len(tok.encode(first["fable"]).ids)
    mask = np.frombuffer((out / "mask.bin").read_bytes(), dtype=np.uint8)
    tokens = np.frombuffer((out / "tokens.bin").read_bytes(), dtype=np.uint16)
    assert not mask[:n_p].any()                       # prompt tokens carry no loss
    assert mask[n_p : n_p + n_f + 1].all()            # fable + EOT carry loss
    assert tokens[n_p + n_f] == tok.token_to_id("<|endoftext|>")


def test_summary_accounting(tmp_path, tok_dir):
    out = run_prep(tmp_path, tok_dir)
    s = json.loads((out / "prep_summary.json").read_text())
    assert s["n_rows"] == 24
    assert 0.3 < s["loss_token_fraction"] < 0.9
    assert s["vocab_size"] <= 512


def test_prep_is_hash_deterministic(tmp_path, tok_dir):
    a = json.loads((run_prep(tmp_path, tok_dir, "a") / "manifest.json").read_text())
    b = json.loads((run_prep(tmp_path, tok_dir, "b") / "manifest.json").read_text())
    assert a["artifacts"] == b["artifacts"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prep_stage.py -v`
Expected: FAIL with `ImportError: cannot import name 'prep'`

- [ ] **Step 3: Implement**

`src/tinyfables/stages/prep.py`:

```python
"""Prep stage: tokenize prompt+fable rows, mark loss spans (fable + EOT only,
prompts are masked out), pack into a contiguous stream, trim to a multiple of
the window, and write uint16/uint8 shards.

Contract note: prompt and fable are encoded SEPARATELY and concatenated — no
separator token, no merged encoding across the boundary. Generation (issue 02)
must encode prompts the same way so train and inference token streams match."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tinyfables.config import PrepConfig
from tinyfables.constants import EOT
from tinyfables.data import read_rows
from tinyfables.stage import write_manifest


def run(cfg: PrepConfig, out_dir: Path) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    if tok.get_vocab_size() > 65535:
        raise ValueError("vocab too large for uint16 shards")
    eot_id = tok.token_to_id(EOT)

    rows = read_rows(cfg.source, cfg.seed)
    toks: list[int] = []
    mask: list[int] = []
    n_prompt, n_fable = 0, 0
    for r in rows:
        p = tok.encode(r["prompt"]).ids
        f = tok.encode(r["fable"]).ids
        toks.extend(p)
        mask.extend([0] * len(p))
        toks.extend(f)
        toks.append(eot_id)
        mask.extend([1] * (len(f) + 1))
        n_prompt += len(p)
        n_fable += len(f) + 1

    n_windows = len(toks) // cfg.window
    n_keep = n_windows * cfg.window
    tokens_path = out_dir / "tokens.bin"
    mask_path = out_dir / "mask.bin"
    tokens_path.write_bytes(np.asarray(toks[:n_keep], dtype=np.uint16).tobytes())
    mask_path.write_bytes(np.asarray(mask[:n_keep], dtype=np.uint8).tobytes())

    summary = {
        "n_rows": len(rows),
        "window": cfg.window,
        "n_windows": n_windows,
        "n_tokens_written": n_keep,
        "n_prompt_tokens_total": n_prompt,
        "n_fable_tokens_total": n_fable,
        "loss_token_fraction": round(sum(mask[:n_keep]) / n_keep, 4) if n_keep else 0.0,
        "vocab_size": tok.get_vocab_size(),
    }
    summary_path = out_dir / "prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(out_dir, "prep", cfg, [tokens_path, mask_path, summary_path])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prep_stage.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/tinyfables/stages/prep.py tests/test_prep_stage.py
git commit -m "feat: prep stage packing shards with loss masks

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Stage registry, CLI, toy/full configs, end-to-end chain

**Files:**
- Modify: `src/tinyfables/stages/__init__.py`
- Create: `src/tinyfables/cli.py`
- Create: `src/tinyfables/__main__.py`
- Create: `configs/tokenizer_toy.yaml`, `configs/prep_toy.yaml`
- Create: `configs/tokenizer_full.yaml`, `configs/prep_full.yaml`
- Test: `tests/test_cli_e2e.py`
- Modify: `docs/issues/README.md` (tick issue 01)

**Interfaces:**
- Consumes: everything above.
- Produces: `tinyfables.stages.REGISTRY: dict[str, tuple[type, Callable[[object, Path], None]]]` mapping stage name → (config class, run function); `tinyfables.cli.main(argv: list[str] | None = None) -> int`; invocable as `python -m tinyfables run <stage> --config <yaml> --out <dir>`. Issue 04 consumes the `*_full.yaml` configs.

- [ ] **Step 1: Write the failing end-to-end test**

`tests/test_cli_e2e.py`:

```python
import json
import time
from pathlib import Path

from tinyfables.cli import main

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def write_configs(tmp_path):
    tok_cfg = tmp_path / "tok.yaml"
    tok_cfg.write_text(
        f"source:\n  jsonl_path: {FIXTURE}\nvocab_size: 512\nseed: 0\ncompare_gpt2: false\n"
    )
    prep_cfg = tmp_path / "prep.yaml"
    prep_cfg.write_text(
        f"source:\n  jsonl_path: {FIXTURE}\n"
        f"tokenizer_dir: {tmp_path / 'runs' / 'tok'}\nwindow: 256\nseed: 0\n"
    )
    return tok_cfg, prep_cfg


def test_toy_chain_via_cli_under_two_minutes(tmp_path):
    tok_cfg, prep_cfg = write_configs(tmp_path)
    t0 = time.monotonic()
    assert main(["run", "tokenizer", "--config", str(tok_cfg), "--out", str(tmp_path / "runs" / "tok")]) == 0
    assert main(["run", "prep", "--config", str(prep_cfg), "--out", str(tmp_path / "runs" / "prep")]) == 0
    elapsed = time.monotonic() - t0
    assert elapsed < 120  # issue 01 acceptance criterion

    for stage_dir, expected in [
        (tmp_path / "runs" / "tok", {"tokenizer.json", "compression_report.json", "manifest.json"}),
        (tmp_path / "runs" / "prep", {"tokens.bin", "mask.bin", "prep_summary.json", "manifest.json"}),
    ]:
        assert {p.name for p in stage_dir.iterdir()} == expected
        manifest = json.loads((stage_dir / "manifest.json").read_text())
        assert manifest["artifacts"]  # every stage records artifact hashes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_e2e.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tinyfables.cli'`

- [ ] **Step 3: Implement registry, CLI, and entrypoint**

Replace `src/tinyfables/stages/__init__.py` with:

```python
"""Stage registry: name -> (config class, run function). The CLI dispatches
through this dict; later slices add stages here (pretrain, label, rm, ppo...)."""

from tinyfables.config import PrepConfig, TokenizerConfig
from tinyfables.stages import prep, tokenizer

REGISTRY = {
    "tokenizer": (TokenizerConfig, tokenizer.run),
    "prep": (PrepConfig, prep.run),
}
```

`src/tinyfables/cli.py`:

```python
"""One CLI for every stage: python -m tinyfables run <stage> --config c.yaml --out dir"""

from __future__ import annotations

import argparse
from pathlib import Path

from tinyfables.config import load_config
from tinyfables.stages import REGISTRY


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinyfables")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run a pipeline stage")
    run_p.add_argument("stage", choices=sorted(REGISTRY))
    run_p.add_argument("--config", required=True, help="path to the stage's YAML config")
    run_p.add_argument("--out", required=True, help="output directory for artifacts")
    args = parser.parse_args(argv)

    config_cls, run_fn = REGISTRY[args.stage]
    cfg = load_config(args.config, config_cls)
    run_fn(cfg, Path(args.out))
    print(f"[tinyfables] stage '{args.stage}' complete -> {args.out}")
    return 0
```

`src/tinyfables/__main__.py`:

```python
from tinyfables.cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes, then the whole suite**

Run: `pytest tests/test_cli_e2e.py -v && pytest`
Expected: 1 passed; then full suite all green (1 deselected network test), total well under 2 minutes

- [ ] **Step 5: Add the checked-in configs**

`configs/tokenizer_toy.yaml`:

```yaml
source:
  jsonl_path: tests/fixtures/tiny_corpus.jsonl
vocab_size: 512
seed: 0
compare_gpt2: false
```

`configs/prep_toy.yaml`:

```yaml
source:
  jsonl_path: tests/fixtures/tiny_corpus.jsonl
tokenizer_dir: runs/tokenizer_toy
window: 256
seed: 0
```

`configs/tokenizer_full.yaml`:

```yaml
# Real run (issue 04, Colab): tokenizer trained on a 50k-fable sample.
source:
  hf_dataset: klusai/ds-tf1-en-3m
  hf_split: train
  max_rows: 50000
vocab_size: 8192
seed: 0
compare_gpt2: true   # produces the ADR-0002 report figure
```

`configs/prep_full.yaml`:

```yaml
# Real run (issue 04, Colab): ~450k fables ≈ 250M tokens (design.md data budget).
source:
  hf_dataset: klusai/ds-tf1-en-3m
  hf_split: train
  max_rows: 450000
tokenizer_dir: runs/tokenizer_full
window: 1024
seed: 0
```

Verify the toy configs work from the repo root exactly as a fresh engineer would:

```bash
python -m tinyfables run tokenizer --config configs/tokenizer_toy.yaml --out runs/tokenizer_toy \
  && python -m tinyfables run prep --config configs/prep_toy.yaml --out runs/prep_toy \
  && cat runs/prep_toy/prep_summary.json
```

Expected: both stages print `complete`, and the summary shows `"n_rows": 24` with a `loss_token_fraction` between 0.3 and 0.9.

- [ ] **Step 6: Tick issue 01 in the queue and commit**

In `docs/issues/README.md`, change the issue 01 row's `☐` to `☑`, then:

```bash
git add src/tinyfables tests/test_cli_e2e.py configs docs/issues/README.md
git commit -m "feat: stage registry + CLI, toy/full configs, e2e toy chain

Closes issue 01 (stage-contract skeleton): all acceptance criteria covered
by tests; full offline suite runs in seconds on CPU.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review (completed)

- **Spec coverage** (issue 01 ACs): uniform invocation + hash-stable artifacts → Tasks 3, 5, 6 determinism tests + Task 7 CLI. Shards with masks + summary → Task 6. 8k BPE with specials + round-trip → Task 5 (vocab 8192 encoded in `tokenizer_full.yaml`; toy tests use 512 because 24 rows can't fill 8192). Compression report vs GPT-2 → Task 5 report schema + `compare_gpt2: true` in the full config (the real number lands in issue 04's Colab session; offline tests assert the schema with the field null). Toy-scale offline <2 min → Task 7 asserts elapsed <120s. No notebooks → none exist.
- **Placeholder scan**: no TBDs; every code step contains complete code; every run step has expected output.
- **Type consistency**: `read_rows(src, seed) -> list[dict]` used identically in Tasks 4–6; `run(cfg, out_dir)` signature identical across both stages and matches the `REGISTRY` tuple type; `EOT`/`SPECIALS` names consistent; `manifest["artifacts"]` is the determinism comparison key everywhere.
