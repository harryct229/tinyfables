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
