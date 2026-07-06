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
