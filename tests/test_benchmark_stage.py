import json
from pathlib import Path

from tinyfables.config import BenchmarkConfig, SourceSpec, TokenizerConfig
from tinyfables.stages import REGISTRY
from tinyfables.stages import benchmark as benchmark_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def _tok_dir(tmp_path):
    out = tmp_path / "tok"
    tokenizer_stage.run(TokenizerConfig(source=SourceSpec(jsonl_path=FIXTURE), vocab_size=512, seed=0), out)
    return out


def _cfg(tok_dir, **kw):
    base = dict(
        tokenizer_dir=str(tok_dir), n_layer=2, n_head=2, d_model=64, n_ctx=64,
        batch_size=2, window=64, warmup_steps=1, measure_steps=2, amp=False, device="cpu",
    )
    base.update(kw)
    return BenchmarkConfig(**base)


def test_benchmark_is_registered():
    assert REGISTRY["benchmark"] == (BenchmarkConfig, "tinyfables.stages.benchmark")


def test_benchmark_writes_summary_and_manifest(tmp_path):
    tok_dir = _tok_dir(tmp_path)
    out = tmp_path / "bench"
    benchmark_stage.run(_cfg(tok_dir), out)
    summary = json.loads((out / "benchmark_summary.json").read_text())
    assert summary["tokens_per_second"] > 0
    assert summary["decision"] in {"go", "revise"}
    assert summary["est_seconds_per_epoch"] is not None
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage"] == "benchmark"
    assert "tokenizer.json" in manifest["inputs"]


def test_decision_go_when_no_target(tmp_path):
    tok_dir = _tok_dir(tmp_path)
    out = tmp_path / "b2"
    benchmark_stage.run(_cfg(tok_dir, target_tokens_per_sec=0.0), out)
    assert json.loads((out / "benchmark_summary.json").read_text())["decision"] == "go"


def test_decision_revise_when_target_unreachable(tmp_path):
    tok_dir = _tok_dir(tmp_path)
    out = tmp_path / "b3"
    benchmark_stage.run(_cfg(tok_dir, target_tokens_per_sec=1e12), out)
    assert json.loads((out / "benchmark_summary.json").read_text())["decision"] == "revise"
