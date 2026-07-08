import json
from pathlib import Path

from tinyfables.config import (
    AuditConfig,
    DeriveConfig,
    LabelConfig,
    PairgenConfig,
    PretrainConfig,
    PrepConfig,
    SourceSpec,
    TokenizerConfig,
)
from tinyfables.feedback import AXES
from tinyfables.stages import audit as audit_stage
from tinyfables.stages import derive as derive_stage
from tinyfables.stages import label as label_stage
from tinyfables.stages import pairgen as pairgen_stage
from tinyfables.stages import prep as prep_stage
from tinyfables.stages import pretrain as pretrain_stage
from tinyfables.stages import tokenizer as tokenizer_stage

REPO = Path(__file__).resolve().parents[1]
FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
SRC = SourceSpec(jsonl_path=FIXTURE)


def _fake_runner(prompt, model):
    ids = [ln.split("pair_id: ")[1].strip() for ln in prompt.splitlines() if "pair_id: " in ln]

    def rate(seedtext):
        h = abs(hash(seedtext))
        return {a: 1 + (h >> (3 * j)) % 5 for j, a in enumerate(AXES)}

    return json.dumps(
        {
            "labels": [
                {"pair_id": pid, "fable_a": rate(pid + "a"), "fable_b": rate(pid + "b")}
                for pid in ids
            ]
        }
    )


def test_feedback_path_composes_offline(tmp_path):
    tok = tmp_path / "tok"
    prep = tmp_path / "prep"
    ckpt = tmp_path / "ckpt"
    tokenizer_stage.run(TokenizerConfig(source=SRC, vocab_size=512, seed=0), tok)
    prep_stage.run(PrepConfig(source=SRC, tokenizer_dir=str(tok), window=512, seed=0), prep)
    pretrain_stage.run(
        PretrainConfig(
            prep_dir=str(prep),
            tokenizer_dir=str(tok),
            n_layer=2,
            n_head=2,
            d_model=64,
            n_ctx=512,
            batch_size=4,
            steps=4,
            warmup_steps=0,
            seed=0,
            device="cpu",
        ),
        ckpt,
    )

    pairs_dir = tmp_path / "pairs"
    pairgen_stage.run(
        PairgenConfig(
            checkpoint=str(ckpt),
            tokenizer_dir=str(tok),
            source=SRC,
            n_ctx=512,
            n_pairs=6,
            max_new_tokens=24,
            min_new_tokens=8,
            top_k=10,
            seed=0,
            device="cpu",
        ),
        pairs_dir,
    )

    labels_dir = tmp_path / "labels"
    label_stage.run(
        LabelConfig(
            pairs=str(pairs_dir / "pairs.jsonl"),
            rubric=str(REPO / "RUBRIC.md"),
            labeler_prompt=str(REPO / "configs" / "labeler_prompt.yaml"),
            model="fake-model",
            batch_size=2,
            swap_fraction=0.5,
            calibration_size=2,
            seed=0,
        ),
        labels_dir,
        runner=_fake_runner,
    )

    derive_dir = tmp_path / "derive"
    derive_stage.run(
        DeriveConfig(labels=str(labels_dir / "labels.jsonl"), pairs=str(pairs_dir / "pairs.jsonl")),
        derive_dir,
    )

    audit_dir = tmp_path / "audit"
    audit_stage.run(AuditConfig(labels=str(labels_dir / "labels.jsonl")), audit_dir)

    assert (pairs_dir / "manifest.json").exists()
    assert (labels_dir / "labels.jsonl").exists()
    assert (derive_dir / "preferences.jsonl").exists()
    assert json.loads((audit_dir / "audit.json").read_text())["self_consistency"]["n_pairs"] == 2
