import json
from pathlib import Path

from tinyfables.config import EvalConfig, PretrainConfig, PrepConfig, SourceSpec, TokenizerConfig
from tinyfables.stage import sha256_file
from tinyfables.stages import evaluate as evaluate_stage
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import pretrain as pretrain_stage
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl"
BANK = Path(__file__).resolve().parents[1] / "configs" / "paraphrases.yaml"


def test_toy_noaug_chain_reuses_tokenizer_and_reaches_three_family_eval(tmp_path):
    source = SourceSpec(jsonl_path=str(FIXTURE))
    tokenizer_dir = tmp_path / "tokenizer"
    prep_dir = tmp_path / "prep_noaug"
    model_dir = tmp_path / "base_noaug"
    eval_dir = tmp_path / "eval_noaug"

    tokenizer_stage.run(
        TokenizerConfig(source=source, vocab_size=512, seed=0),
        tokenizer_dir,
    )
    tokenizer_sha = sha256_file(tokenizer_dir / "tokenizer.json")
    prep_stage.run(
        PrepConfig(
            source=source,
            tokenizer_dir=str(tokenizer_dir),
            window=512,
            seed=0,
            paraphrase_bank=str(BANK),
            paraphrase_coverage=0.0,
        ),
        prep_dir,
    )
    prep_summary = json.loads((prep_dir / "prep_summary.json").read_text())
    assert prep_summary["n_rows"] == 24
    assert prep_summary["paraphrase_coverage"] == 0.0
    assert prep_summary["family_counts"] == {
        "canonical": 24,
        "seen-template": 0,
        "held-out-template": 0,
    }
    prep_manifest = json.loads((prep_dir / "manifest.json").read_text())
    assert prep_manifest["inputs"]["tokenizer.json"] == tokenizer_sha

    pretrain_stage.run(
        PretrainConfig(
            prep_dir=str(prep_dir),
            tokenizer_dir=str(tokenizer_dir),
            n_layer=1,
            n_head=2,
            d_model=32,
            n_ctx=512,
            batch_size=2,
            steps=2,
            warmup_steps=0,
            lr=1e-3,
            seed=0,
            device="cpu",
        ),
        model_dir,
    )
    pretrain_manifest = json.loads((model_dir / "manifest.json").read_text())
    assert pretrain_manifest["inputs"]["tokenizer.json"] == tokenizer_sha

    evaluate_stage.run(
        EvalConfig(
            checkpoint=str(model_dir),
            tokenizer_dir=str(tokenizer_dir),
            source=source,
            paraphrase_bank=str(BANK),
            n_ctx=512,
            n_perplexity_rows=2,
            n_generations=1,
            max_new_tokens=8,
            min_new_tokens=2,
            temperature=0.9,
            top_k=10,
            moral_threshold=0.3,
            seed=0,
            device="cpu",
        ),
        eval_dir,
    )
    metrics = json.loads((eval_dir / "eval_metrics.json").read_text())
    assert set(metrics["adherence_grid"]) == {
        "canonical",
        "seen-template",
        "held-out-template",
    }
    assert {row["n"] for row in metrics["adherence_grid"].values()} == {1}
    eval_manifest = json.loads((eval_dir / "manifest.json").read_text())
    assert eval_manifest["inputs"]["tokenizer.json"] == tokenizer_sha
    assert eval_manifest["inputs"]["paraphrases.yaml"] == sha256_file(BANK)
