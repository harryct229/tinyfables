import json
from pathlib import Path

import numpy as np
import pytest
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
        paraphrase_bank=bank,
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


def test_all_parse_failures_raises(tmp_path):
    corpus = tmp_path / "allbad.jsonl"
    corpus.write_text(
        json.dumps({"prompt": "Write a fable about a fox.", "fable": "A fox learned.\n\n**The Moral:** x"}) + "\n"
        + json.dumps({"prompt": "Another non-canonical prompt.", "fable": "B story.\n\n**The Moral:** y"}) + "\n"
    )
    src = SourceSpec(jsonl_path=str(corpus))
    tok = tmp_path / "tokbad"
    tokenizer_stage.run(TokenizerConfig(source=src, vocab_size=512, seed=0), tok)
    with pytest.raises(ValueError):
        prep_stage.run(
            PrepConfig(source=src, tokenizer_dir=str(tok), window=256, seed=0,
                       paraphrase_bank=BANK, paraphrase_coverage=1.0),
            tmp_path / "allbadprep",
        )


def test_mask_covers_paraphrased_prompt_span(tmp_path):
    from tinyfables.data import read_rows
    from tinyfables.paraphrases import load_bank, select_row

    tok_dir = _tok_dir(tmp_path)
    out = tmp_path / "cov1mask"
    prep_stage.run(
        PrepConfig(source=SRC, tokenizer_dir=str(tok_dir), window=256, seed=0,
                   paraphrase_bank=BANK, paraphrase_coverage=1.0),
        out,
    )
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    first = read_rows(SRC, seed=0)[0]  # same seed => same post-shuffle order as the stage
    row = select_row(first["prompt"], 0, 1.0, 0, load_bank(BANK))  # row 0 is paraphrased at coverage 1.0
    n_p = len(tok.encode(row.prompt).ids)
    mask = np.frombuffer((out / "mask.bin").read_bytes(), dtype=np.uint8)
    assert not mask[:n_p].any()  # the paraphrased prompt span carries no loss
    assert mask[n_p]             # the first fable token carries loss
