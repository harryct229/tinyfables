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
