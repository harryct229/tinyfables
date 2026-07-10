"""DPO alignment stage (issue 08): the pre-declared fallback (ADR-0004 /
design.md Feedback stage). Consumes the same derived-preferences artifact as
the reward model, optimizes the policy directly (no reward model at
optimization time — nothing to reward-hack), and applies the same
length-drift alarm via before/after probes. `min_margin` implements the cheap
margin-filtered variant."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tinyfables.config import DPOStageConfig
from tinyfables.stage import sha256_file, write_manifest


def _resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(cfg: DPOStageConfig, out_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from trl import DPOConfig, DPOTrainer

    from tinyfables.hf_tokenizer import load_hf_tokenizer
    from tinyfables.length_alarm import length_drift, probe_lengths
    from tinyfables.model import GPT

    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    rows = [
        json.loads(line)
        for line in Path(cfg.preferences).read_text().splitlines()
        if line.strip()
    ]
    train_rows = [r for r in rows if r["split"] == "train"]
    kept = [
        r for r in train_rows
        if (r["aggregate_chosen"] - r["aggregate_rejected"]) >= cfg.min_margin
    ]
    if not kept:
        raise ValueError(
            f"min_margin={cfg.min_margin} filtered out all {len(train_rows)} train preferences"
        )
    probe_prompts = sorted({r["prompt"] for r in rows if r["split"] == "held_out"})[: cfg.n_probe_prompts]
    if not probe_prompts:
        probe_prompts = sorted({r["prompt"] for r in kept})[: cfg.n_probe_prompts]

    dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in kept]
    )

    raw_tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    hf_tok = load_hf_tokenizer(cfg.tokenizer_dir)
    policy = GPT.from_pretrained(cfg.base_checkpoint).to(device)
    ref = GPT.from_pretrained(cfg.base_checkpoint).to(device)

    before = probe_lengths(
        policy.eval(), raw_tok, probe_prompts,
        max_new_tokens=cfg.probe_max_new_tokens, temperature=cfg.temperature,
        seed=cfg.seed, device=device,
    )

    args = DPOConfig(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.num_train_epochs,
        beta=cfg.beta,
        max_length=cfg.n_ctx,
        logging_steps=cfg.logging_steps,
        report_to=[],
        seed=cfg.seed,
        use_cpu=(device == "cpu"),
        fp16=cfg.fp16 and device == "cuda",
        # DPOConfig defaults gradient_checkpointing=True (unlike plain
        # TrainingArguments). The hand-written GPT never sets
        # `supports_gradient_checkpointing = True`, so
        # PreTrainedModel.gradient_checkpointing_enable() raises ValueError
        # ("GPT does not support gradient checkpointing") the moment the
        # trainer is constructed. Same fix as the ppo stage.
        gradient_checkpointing=False,
        save_strategy="no",
    )
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=args,
        train_dataset=dataset,
        processing_class=hf_tok,
    )
    trainer.train()

    curves_path = out_dir / "dpo_curves.csv"
    logged = [h for h in trainer.state.log_history if "loss" in h]
    with curves_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "loss", "rewards_margins"])
        writer.writeheader()
        for h in logged:
            writer.writerow({
                "step": h.get("step", ""),
                "loss": h.get("loss", ""),
                "rewards_margins": h.get("rewards/margins", ""),
            })

    aligned = trainer.model
    aligned.save_pretrained(out_dir)

    after = probe_lengths(
        aligned.eval(), raw_tok, probe_prompts,
        max_new_tokens=cfg.probe_max_new_tokens, temperature=cfg.temperature,
        seed=cfg.seed, device=device,
    )
    drift = length_drift(before["mean_words"], after["mean_words"], cfg.length_alarm_threshold)

    summary = {
        "adr_decision": cfg.adr_decision,
        "preferences_sha": sha256_file(Path(cfg.preferences)),
        "base_checkpoint_sha": sha256_file(Path(cfg.base_checkpoint) / "model.safetensors"),
        "min_margin": cfg.min_margin,
        "n_train_pairs_available": len(train_rows),
        "n_train_pairs": len(kept),
        "n_filtered_out": len(train_rows) - len(kept),
        "beta": cfg.beta,
        "final_loss": logged[-1]["loss"] if logged else None,
        "probe_before": before,
        "probe_after": after,
        "length_drift": drift,
        "length_alarm_triggered": drift["alarm"],
        "device": device,
    }
    (out_dir / "dpo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "dpo",
        cfg,
        [
            out_dir / "config.json",
            out_dir / "generation_config.json",
            out_dir / "model.safetensors",
            out_dir / "dpo_curves.csv",
            out_dir / "dpo_summary.json",
        ],
        inputs={
            "preferences.jsonl": Path(cfg.preferences),
            "base_model.safetensors": Path(cfg.base_checkpoint) / "model.safetensors",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
