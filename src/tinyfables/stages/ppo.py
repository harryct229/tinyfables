"""PPO alignment stage (issue 08): TRL experimental PPOTrainer, KL-anchored to
the frozen Base Model, gate-guarded (refuses to start without a gate-pass
record — tinyfables.gates keeps "do not optimize a noisy signal" executable),
with a per-iteration length-drift alarm and reward/KL curves.

TRL 1.8.0: PPOTrainer lives ONLY in trl.experimental.ppo. It calls every model
as model(input_ids, attention_mask, position_ids, return_dict=True,
output_hidden_states=True) over LEFT-padded batches, and generates through
lm_backbone.generate(...) — the GPT's mask/position/cache-proof support
(issue-08 model change) exists precisely for this."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from tinyfables.config import PPOStageConfig
from tinyfables.gates import assert_gate_passed
from tinyfables.stage import sha256_file, write_manifest


def _resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(cfg: PPOStageConfig, out_dir: Path) -> None:
    # Gate FIRST: no torch/trl import, no out_dir writes, before a failing gate.
    gate = assert_gate_passed(cfg.gate)

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from transformers import TrainerCallback
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    from tinyfables.hf_tokenizer import load_hf_tokenizer
    from tinyfables.length_alarm import length_drift, probe_lengths
    from tinyfables.model import GPT
    from tinyfables.reward_data import load_preferences
    from tinyfables.trl_compat import ScoredModelAdapter

    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    raw_tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    hf_tok = load_hf_tokenizer(cfg.tokenizer_dir, padding_side="left")

    policy = GPT.from_pretrained(cfg.base_checkpoint).to(device)
    ref = GPT.from_pretrained(cfg.base_checkpoint).to(device)
    reward_model = ScoredModelAdapter.from_reward_model_dir(cfg.reward_model_dir, device=device)
    value_model = ScoredModelAdapter.from_reward_model_dir(cfg.reward_model_dir, device=device)

    train_prompts = sorted({ex.prompt for ex in load_preferences(cfg.preferences, split="train")})
    probe_prompts = sorted({ex.prompt for ex in load_preferences(cfg.preferences, split="held_out")})[: cfg.n_probe_prompts]
    if not train_prompts:
        raise ValueError("no train-split prompts in preferences")
    if not probe_prompts:
        probe_prompts = train_prompts[: cfg.n_probe_prompts]

    max_prompt = cfg.n_ctx - cfg.response_length
    def _tokenize(row):
        return {"input_ids": hf_tok(row["prompt"], add_special_tokens=False)["input_ids"][:max_prompt]}

    dataset = Dataset.from_list([{"prompt": p} for p in train_prompts]).map(_tokenize, remove_columns=["prompt"])
    # PPOTrainer.__init__ unconditionally builds a DataLoader over `eval_dataset`
    # (trl/experimental/ppo/ppo_trainer.py: `self.eval_dataloader = DataLoader(self.eval_dataset, ...)`,
    # no `is not None` guard) — passing None crashes construction. Reuse the held-out
    # probe prompts (already tokenized the same way) so no extra data is invented.
    eval_dataset = Dataset.from_list([{"prompt": p} for p in probe_prompts]).map(_tokenize, remove_columns=["prompt"])

    baseline = probe_lengths(
        policy.eval(), raw_tok, probe_prompts,
        max_new_tokens=cfg.response_length, temperature=cfg.temperature, seed=cfg.seed, device=device,
    )

    curves_path = out_dir / "ppo_curves.csv"
    fieldnames = [
        "episode", "objective_kl", "objective_scores", "objective_rlhf_reward",
        "objective_non_score_reward", "loss_policy_avg", "loss_value_avg",
        "probe_mean_words", "probe_mean_new_tokens", "length_drift_pct", "length_alarm",
    ]
    curves_fh = curves_path.open("w", newline="")
    writer = csv.DictWriter(curves_fh, fieldnames=fieldnames)
    writer.writeheader()
    alarm_state = {"triggered": False, "final": None}

    class LengthAlarmCallback(TrainerCallback):
        """Per-iteration probe: PPO logs once per rollout batch (logging_steps=1),
        so on_log is the every-iteration hook; TRL logs no length metric itself."""

        def __init__(self, trainer_ref):
            self.trainer_ref = trainer_ref

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None or "objective/kl" not in logs:
                return
            pol = self.trainer_ref["trainer"].policy_model  # PPOTrainer keeps the raw policy here
            was_training = pol.training
            pol.eval()
            probe = probe_lengths(
                pol, raw_tok, probe_prompts,
                max_new_tokens=cfg.response_length, temperature=cfg.temperature,
                seed=cfg.seed, device=next(pol.parameters()).device,
            )
            if was_training:
                pol.train()
            drift = length_drift(baseline["mean_words"], probe["mean_words"], cfg.length_alarm_threshold)
            alarm_state["triggered"] = alarm_state["triggered"] or drift["alarm"]
            alarm_state["final"] = drift
            writer.writerow({
                "episode": logs.get("episode", ""),
                "objective_kl": logs.get("objective/kl", ""),
                "objective_scores": logs.get("objective/scores", ""),
                "objective_rlhf_reward": logs.get("objective/rlhf_reward", ""),
                "objective_non_score_reward": logs.get("objective/non_score_reward", ""),
                "loss_policy_avg": logs.get("loss/policy_avg", ""),
                "loss_value_avg": logs.get("loss/value_avg", ""),
                "probe_mean_words": probe["mean_words"],
                "probe_mean_new_tokens": probe["mean_new_tokens"],
                "length_drift_pct": drift["drift_pct"],
                "length_alarm": drift["alarm"],
            })
            curves_fh.flush()

    args = PPOConfig(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        total_episodes=cfg.total_episodes,
        num_ppo_epochs=cfg.num_ppo_epochs,
        num_mini_batches=cfg.num_mini_batches,
        local_rollout_forward_batch_size=cfg.local_rollout_forward_batch_size,
        response_length=cfg.response_length,
        stop_token="eos",
        temperature=cfg.temperature,
        missing_eos_penalty=cfg.missing_eos_penalty,
        kl_coef=cfg.kl_coef,
        whiten_rewards=cfg.whiten_rewards,
        num_sample_generations=0,
        logging_steps=1,
        report_to=[],
        seed=cfg.seed,
        use_cpu=(device == "cpu"),
        fp16=cfg.fp16 and device == "cuda",
        gradient_checkpointing=False,  # hand-written GPT has no checkpointing support; not needed at toy/full scale here
        save_strategy="no",
    )
    trainer_ref: dict = {}
    trainer = PPOTrainer(
        args=args,
        processing_class=hf_tok,
        model=policy,
        ref_model=ref,
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=value_model,
        eval_dataset=eval_dataset,
        callbacks=[LengthAlarmCallback(trainer_ref)],
    )
    trainer_ref["trainer"] = trainer
    trainer.train()
    curves_fh.close()

    aligned = trainer.policy_model
    aligned.save_pretrained(out_dir)

    summary = {
        "adr_decision": cfg.adr_decision,
        "gate": "pass",
        "gate_sha": sha256_file(Path(cfg.gate)),
        "preferences_sha": sha256_file(Path(cfg.preferences)),
        "base_checkpoint_sha": sha256_file(Path(cfg.base_checkpoint) / "model.safetensors"),
        "reward_model_sha": sha256_file(Path(cfg.reward_model_dir) / "reward_head.pt"),
        "total_episodes": cfg.total_episodes,
        "kl_coef": cfg.kl_coef,
        "baseline_probe": baseline,
        "final_length_drift": alarm_state["final"],
        "length_alarm_triggered": alarm_state["triggered"],
        "device": device,
    }
    (out_dir / "ppo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "ppo",
        cfg,
        [
            out_dir / "config.json",
            out_dir / "generation_config.json",
            out_dir / "model.safetensors",
            out_dir / "ppo_curves.csv",
            out_dir / "ppo_summary.json",
        ],
        inputs={
            "gate.json": Path(cfg.gate),
            "preferences.jsonl": Path(cfg.preferences),
            "base_model.safetensors": Path(cfg.base_checkpoint) / "model.safetensors",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
            "reward_head.pt": Path(cfg.reward_model_dir) / "reward_head.pt",
        },
    )
