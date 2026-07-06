"""Benchmark stage: measure real T4 throughput (tokens/sec) at the TARGET model
geometry BEFORE committing to the multi-hour pretrain — the week-1 size gate
(design.md: measure, then commit). Runs synthetic batches (real vocab from the
tokenizer, real batch×window shape, real AMP path), excludes warmup, and records
tokens/sec, an epoch-time estimate for the token budget, and a go/no-go decision.

Synthetic batches (not prep shards) keep this stage a pure compute probe: GPU
matmul throughput dominates; the memmap read is negligible. It depends only on the
tokenizer (vocab size drives the embedding/head cost), so it can run the moment
the tokenizer exists."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import BenchmarkConfig
from tinyfables.model import GPT, GPTConfig
from tinyfables.stage import write_manifest


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(cfg: BenchmarkConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    vocab_size = tok.get_vocab_size()

    model = GPT(
        GPTConfig(
            vocab_size=vocab_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            d_model=cfg.d_model,
            n_ctx=cfg.n_ctx,
        )
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    use_amp = cfg.amp and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    gen = torch.Generator().manual_seed(cfg.seed)

    def make_batch() -> torch.Tensor:
        ids = torch.randint(0, vocab_size, (cfg.batch_size, cfg.window), generator=gen)
        return ids.to(device)

    def one_step(ids: torch.Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_amp):
            out = model(input_ids=ids, labels=ids)
        scaler.scale(out.loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(cfg.warmup_steps):
        one_step(make_batch())
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(cfg.measure_steps):
        one_step(make_batch())
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = cfg.measure_steps * cfg.batch_size * cfg.window
    tps = tokens / elapsed if elapsed > 0 else 0.0
    decision = "go" if (cfg.target_tokens_per_sec <= 0 or tps >= cfg.target_tokens_per_sec) else "revise"

    summary = {
        "device": device,
        "amp": use_amp,
        "vocab_size": vocab_size,
        "batch_size": cfg.batch_size,
        "window": cfg.window,
        "measure_steps": cfg.measure_steps,
        "elapsed_sec": round(elapsed, 4),
        "tokens_per_second": round(tps, 2),
        "est_seconds_per_epoch": round(cfg.token_budget / tps, 1) if tps > 0 else None,
        "token_budget": cfg.token_budget,
        "target_tokens_per_sec": cfg.target_tokens_per_sec,
        "decision": decision,
    }
    summary_path = out_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    write_manifest(
        out_dir,
        "benchmark",
        cfg,
        [summary_path],
        inputs={"tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json"},
    )
