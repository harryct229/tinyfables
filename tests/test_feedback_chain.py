import hashlib
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
        h = int.from_bytes(hashlib.sha256(seedtext.encode("utf-8")).digest()[:8], "big")
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


def test_issue07_reward_and_gate_toy_chain(tmp_path):
    import json
    import torch
    from tokenizers import Tokenizer

    from tinyfables.config import GateConfig, RewardTrainConfig, SourceSpec, TokenizerConfig
    from tinyfables.model import GPT, GPTConfig
    from tinyfables.stages import gate as gate_stage
    from tinyfables.stages import reward as reward_stage
    from tinyfables.stages import tokenizer as tokenizer_stage

    fix = Path(__file__).parent / "fixtures"
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(fix / "tiny_corpus.jsonl")), vocab_size=512, seed=0),
        tok_dir,
    )
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    base = tmp_path / "base"
    torch.manual_seed(0)
    GPT(
        GPTConfig(
            vocab_size=tok.get_vocab_size(),
            n_layer=1,
            n_head=2,
            d_model=32,
            n_ctx=128,
        )
    ).save_pretrained(base)

    reward_out = tmp_path / "reward"
    reward_stage.run(
        RewardTrainConfig(
            preferences=str(fix / "preferences_replay.jsonl"),
            base_checkpoint=str(base),
            tokenizer_dir=str(tok_dir),
            n_ctx=128,
            batch_size=2,
            steps=4,
            curve_steps=2,
            lr=1e-3,
            warmup_steps=0,
            seed=0,
            device="cpu",
            amp=False,
            log_every=2,
            curve_sizes=[2],
        ),
        reward_out,
    )

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "gate": {
                    "self_consistency_gate": 0.85,
                    "self_consistency_pass": True,
                    "position_swap_review_flag": False,
                    "position_swap_review_threshold": 0.5,
                },
                "self_consistency": {"mean_agreement": 0.9, "n_pairs": 30, "n_unanimous": 27},
                "position_swap": {"flip_rate": 0.2, "n_flipped": 40, "n_pairs": 200},
            }
        )
        + "\n"
    )

    gate_out = tmp_path / "gate"
    gate_stage.run(
        GateConfig(audit=str(audit_path), reward_summary=str(reward_out / "reward_summary.json")),
        gate_out,
    )
    assert (gate_out / "gate.json").exists()
