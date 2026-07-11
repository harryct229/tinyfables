from pathlib import Path

from tinyfables.cli import main

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def test_toy_chain_tokenizer_prep_pretrain_generate(tmp_path, capsys):
    runs = tmp_path / "runs"
    tok_dir = runs / "tok"
    prep_dir = runs / "prep"
    ckpt = runs / "pretrain"

    (tmp_path / "tok.yaml").write_text(
        f"source:\n  jsonl_path: {FIXTURE}\nvocab_size: 512\nseed: 0\ncompare_gpt2: false\n"
    )
    (tmp_path / "prep.yaml").write_text(
        f"source:\n  jsonl_path: {FIXTURE}\ntokenizer_dir: {tok_dir}\nwindow: 512\nseed: 0\n"
    )
    (tmp_path / "pre.yaml").write_text(
        f"prep_dir: {prep_dir}\ntokenizer_dir: {tok_dir}\n"
        "n_layer: 2\nn_head: 2\nd_model: 64\nn_ctx: 512\n"
        "batch_size: 4\nsteps: 100\nwarmup_steps: 10\nlr: 0.001\nseed: 0\ndevice: cpu\n"
    )

    assert main(["run", "tokenizer", "--config", str(tmp_path / "tok.yaml"), "--out", str(tok_dir)]) == 0
    assert main(["run", "prep", "--config", str(tmp_path / "prep.yaml"), "--out", str(prep_dir)]) == 0
    assert main(["run", "pretrain", "--config", str(tmp_path / "pre.yaml"), "--out", str(ckpt)]) == 0

    capsys.readouterr()  # clear the stage-completion prints
    rc = main([
        "generate",
        "--checkpoint", str(ckpt),
        "--tokenizer", str(tok_dir),
        "--character", "a shy octopus",
        "--setting", "a quiet tide pool",
        "--challenge", "doubting oneself",
        "--outcome", "a friend helps just in time",
        "--moral", "courage grows by small steps",
        "--max-new-tokens", "60",
        "--min-new-tokens", "8",
        "--seed", "0",
    ])
    assert rc == 0
    fable = capsys.readouterr().out.strip()
    assert len(fable) > 0
    assert (ckpt / "model.safetensors").exists()
    assert (ckpt / "manifest.json").exists()


def test_toy_chain_extends_through_ppo_and_eval(tmp_path, capsys):
    import json

    import pytest

    pytest.importorskip("trl")

    from .conftest import fake_labeler_runner

    from tinyfables.config import (
        EvalConfig, PairgenConfig, PPOStageConfig, DeriveConfig, LabelConfig, SourceSpec,
    )
    from tinyfables.stages import derive as derive_stage
    from tinyfables.stages import evaluate as evaluate_stage
    from tinyfables.stages import label as label_stage
    from tinyfables.stages import pairgen as pairgen_stage
    from tinyfables.stages import ppo as ppo_stage

    runs = tmp_path / "runs"
    tok_dir, prep_dir, ckpt = runs / "tok", runs / "prep", runs / "pretrain"
    # reuse the toy front half exactly as test_toy_chain_tokenizer_prep_pretrain_generate builds it
    (tmp_path / "tok.yaml").write_text(
        f"source:\n  jsonl_path: {FIXTURE}\nvocab_size: 512\nseed: 0\ncompare_gpt2: false\n"
    )
    (tmp_path / "prep.yaml").write_text(
        f"source:\n  jsonl_path: {FIXTURE}\ntokenizer_dir: {tok_dir}\nwindow: 512\nseed: 0\n"
    )
    (tmp_path / "pre.yaml").write_text(
        f"prep_dir: {prep_dir}\ntokenizer_dir: {tok_dir}\n"
        "n_layer: 2\nn_head: 2\nd_model: 64\nn_ctx: 512\n"
        "batch_size: 4\nsteps: 20\nwarmup_steps: 2\nlr: 0.001\nseed: 0\ndevice: cpu\n"
    )
    for stage, cfg_name, out in (("tokenizer", "tok.yaml", tok_dir), ("prep", "prep.yaml", prep_dir), ("pretrain", "pre.yaml", ckpt)):
        assert main(["run", stage, "--config", str(tmp_path / cfg_name), "--out", str(out)]) == 0

    src = SourceSpec(jsonl_path=FIXTURE)
    pairgen_stage.run(
        PairgenConfig(checkpoint=str(ckpt), tokenizer_dir=str(tok_dir), source=src, n_ctx=512,
                      n_pairs=6, max_new_tokens=16, min_new_tokens=4, top_k=10, seed=0, device="cpu"),
        runs / "pairs",
    )
    label_stage.run(
        LabelConfig(pairs=str(runs / "pairs" / "pairs.jsonl"), rubric=str(Path(__file__).parents[1] / "RUBRIC.md"),
                    labeler_prompt=str(Path(__file__).parents[1] / "configs" / "labeler_prompt.yaml"),
                    model="fake-model", batch_size=2, swap_fraction=0.5, calibration_size=2, seed=0),
        runs / "labels", runner=fake_labeler_runner,
    )
    # NOTE: n_pairs=6 + held_out_fraction=0.5 is a pinned, checked split for this
    # fixed fixture + fake labeler (not a guess): pair-000000/pair-000004 land in
    # held_out, pair-000002/000003/000005 land in train, pair-000001 ties and is
    # dropped -> preferences.jsonl has 3 train + 2 held_out rows. Both the reward
    # stage (>=1 held_out) and PPO's probe-prompt fallback (>=1 train) need that.
    derive_stage.run(
        DeriveConfig(labels=str(runs / "labels" / "labels.jsonl"), pairs=str(runs / "pairs" / "pairs.jsonl"),
                     held_out_fraction=0.5),
        runs / "derive",
    )

    # toy RM trained on the derived toy preferences
    from tinyfables.config import RewardTrainConfig
    from tinyfables.stages import reward as reward_stage
    reward_stage.run(
        RewardTrainConfig(preferences=str(runs / "derive" / "preferences.jsonl"), base_checkpoint=str(ckpt),
                          tokenizer_dir=str(tok_dir), n_ctx=512, batch_size=2, steps=2, curve_steps=1,
                          lr=1e-3, warmup_steps=0, seed=0, device="cpu", amp=False, log_every=1, curve_sizes=[2]),
        runs / "rm",
    )

    ppo_stage.run(
        PPOStageConfig(
            gate=str(Path(__file__).parent / "fixtures" / "gate_pass.json"),
            preferences=str(runs / "derive" / "preferences.jsonl"),
            base_checkpoint=str(ckpt), reward_model_dir=str(runs / "rm"), tokenizer_dir=str(tok_dir),
            adr_decision="ADR-0005-toy", n_ctx=512, response_length=12, total_episodes=2, batch_size=2,
            gradient_accumulation_steps=1, local_rollout_forward_batch_size=2, num_ppo_epochs=1,
            num_mini_batches=1, lr=1e-4, n_probe_prompts=1, seed=0, device="cpu",
        ),
        runs / "aligned",
    )

    evaluate_stage.run(
        EvalConfig(checkpoint=str(runs / "aligned"), tokenizer_dir=str(tok_dir), source=src,
                   n_ctx=512, n_perplexity_rows=4, n_generations=2, max_new_tokens=16,
                   min_new_tokens=4, seed=0, device="cpu"),
        runs / "eval_aligned",
    )
    assert json.loads((runs / "eval_aligned" / "eval_metrics.json").read_text())
    assert (runs / "aligned" / "manifest.json").exists()
