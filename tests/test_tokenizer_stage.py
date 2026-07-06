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
