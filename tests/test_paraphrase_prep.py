import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tinyfables.config import PrepConfig, SourceSpec, TokenizerConfig
from tinyfables.paraphrases import FAMILIES, load_families
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
BANK = str(Path(__file__).parents[1] / "configs" / "paraphrases.yaml")
SRC = SourceSpec(jsonl_path=FIXTURE)


def _tok_dir(tmp_path):
    out = tmp_path / "tok"
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), out)
    return out


def _run(tmp_path, coverage, name, bank=BANK):
    out = tmp_path / name
    cfg = PrepConfig(
        source=SRC,
        tokenizer_dir=str(_tok_dir(tmp_path)),
        window=256,
        seed=0,
        paraphrase_bank=bank if coverage > 0 else bank,
        paraphrase_coverage=coverage,
    )
    prep_stage.run(cfg, out)
    return out


def test_families_written_one_per_row(tmp_path):
    out = _run(tmp_path, 0.5, "half")
    fam = load_families(out / "families.bin")
    summary = json.loads((out / "prep_summary.json").read_text())
    assert len(fam) == summary["n_rows"] == 24
    assert set(np.unique(fam)) <= {0, 1}  # canonical / seen only in training


def test_summary_counts_show_expected_proportions(tmp_path):
    out = _run(tmp_path, 0.5, "half")
    s = json.loads((out / "prep_summary.json").read_text())
    fc = s["family_counts"]
    assert fc["canonical"] + fc["seen-template"] == 24
    assert fc["held-out-template"] == 0
    assert 6 <= fc["seen-template"] <= 18  # ~half of 24, deterministic-but-banded
    assert s["paraphrase_coverage"] == 0.5


def test_coverage_zero_is_all_canonical_and_tokens_match_no_bank(tmp_path):
    # Prompt tokens must be byte-identical whether paraphrasing is off via
    # coverage=0 or absent entirely (backward compatibility).
    with_bank = _run(tmp_path, 0.0, "cov0")
    no_bank = tmp_path / "nobank"
    prep_stage.run(
        PrepConfig(source=SRC, tokenizer_dir=str(_tok_dir(tmp_path)), window=256, seed=0),
        no_bank,
    )
    fam = load_families(with_bank / "families.bin")
    assert set(np.unique(fam)) == {0}  # all canonical
    a = (with_bank / "tokens.bin").read_bytes()
    b = (no_bank / "tokens.bin").read_bytes()
    assert a == b


def test_held_out_never_in_training_artifacts(tmp_path):
    # At full coverage every parseable row is paraphrased; held-out must be 0.
    out = _run(tmp_path, 1.0, "full")
    s = json.loads((out / "prep_summary.json").read_text())
    assert s["family_counts"]["held-out-template"] == 0
    assert s["family_counts"]["seen-template"] == 24
    fam = load_families(out / "families.bin")
    assert 2 not in set(np.unique(fam))  # code 2 == held-out-template


def test_prep_families_are_hash_deterministic(tmp_path):
    a = json.loads((_run(tmp_path, 0.5, "a") / "manifest.json").read_text())
    b = json.loads((_run(tmp_path, 0.5, "b") / "manifest.json").read_text())
    assert a["artifacts"] == b["artifacts"]  # includes families.bin + summary


def test_manifest_tracks_bank_input_hash(tmp_path):
    out = _run(tmp_path, 0.5, "prov")
    manifest = json.loads((out / "manifest.json").read_text())
    assert "paraphrases.yaml" in manifest["inputs"]
    assert "families.bin" in manifest["artifacts"]


def test_parse_failures_counted_and_left_canonical(tmp_path):
    corpus = tmp_path / "mixed.jsonl"
    canonical = json.loads(Path(FIXTURE).read_text().splitlines()[0])
    bad = {"prompt": "Write a fable about a fox.", "fable": "A fox learned to share.\n\n**The Moral:** share"}
    corpus.write_text(json.dumps(canonical) + "\n" + json.dumps(bad) + "\n")
    src = SourceSpec(jsonl_path=str(corpus))
    tok = tmp_path / "tok2"
    tokenizer_stage.run(TokenizerConfig(source=src, vocab_size=512, seed=0), tok)
    out = tmp_path / "mixedprep"
    prep_stage.run(
        PrepConfig(source=src, tokenizer_dir=str(tok), window=256, seed=0,
                   paraphrase_bank=BANK, paraphrase_coverage=1.0),
        out,
    )
    s = json.loads((out / "prep_summary.json").read_text())
    assert s["n_parse_failures"] == 1  # the non-canonical row fell back to canonical
    fam = load_families(out / "families.bin")
    assert (fam == FAMILIES.index("canonical")).sum() >= 1
