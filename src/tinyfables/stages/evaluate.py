"""Evaluate stage: score any checkpoint into a report-ready metrics document —
fable-token perplexity, the canonical/seen/held-out spec-adherence robustness
grid, moral delivery, distinct-n/repetition, and length stats — plus a moral-
extraction hand-labeling worksheet. Deterministic given seed + eval set. This is
how any two checkpoints are compared for the rest of the project.

Held-out paraphrase templates are USED here (they are eval-only, never trained on)
to render the held-out column of the grid — robustness to unseen phrasing measured,
not asserted."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import EvalConfig
from tinyfables.data import iter_rows
from tinyfables.eval_metrics import distinct_n, element_adherence, length_stats, repetition_rate
from tinyfables.generate import generate_fable
from tinyfables.model import GPT
from tinyfables.moral import extract_moral, moral_delivered, moral_similarity, naive_regex_moral
from tinyfables.paraphrases import load_bank, parse_canonical_prompt, render_paraphrase
from tinyfables.perplexity import fable_token_perplexity
from tinyfables.prompts import FableSpec, render_canonical_prompt
from tinyfables.report import write_eval_report
from tinyfables.stage import _versions, sha256_file, write_manifest

_ELEMENTS = ("character", "setting", "challenge", "outcome", "moral")


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _spec_from_prompt(prompt: str) -> FableSpec | None:
    parsed = parse_canonical_prompt(prompt)
    if parsed is None:
        return None
    return FableSpec(**{f: getattr(parsed, f) for f in _ELEMENTS},
                     age_range=parsed.age_range, word_count=parsed.word_count)


def _grid_family(records) -> dict:
    """Aggregate per-family: mean verbatim presence per element, overall mean, and
    moral-delivery rate over the family's generations."""
    n = len(records)
    per_el = {}
    for el in ("character", "setting", "challenge", "outcome"):
        vals = [r["adherence"][el] for r in records if el in r["adherence"]]
        per_el[el] = round(sum(vals) / len(vals), 4) if vals else 0.0
    overall = round(sum(per_el.values()) / len(per_el), 4)
    moral = round(sum(r["moral"] for r in records) / n, 4) if n else 0.0
    return {**per_el, "overall": overall, "moral_delivery": moral, "n": n}


def run(cfg: EvalConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    tok = Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / "tokenizer.json"))
    model = GPT.from_pretrained(cfg.checkpoint).to(device).eval()
    bank = load_bank(cfg.paraphrase_bank) if cfg.paraphrase_bank else None

    # materialize enough val rows for both perplexity and spec sampling
    need = max(cfg.n_perplexity_rows, cfg.n_generations * 4)
    rows = list(itertools.islice(iter_rows(cfg.source, cfg.seed), need))

    mean_loss, ppl, n_ppl_rows = fable_token_perplexity(
        model, tok, rows, cfg.n_ctx, device, cfg.n_perplexity_rows
    )

    specs: list[FableSpec] = []
    for r in rows:
        spec = _spec_from_prompt(r["prompt"])
        if spec is not None:
            specs.append(spec)
        if len(specs) >= cfg.n_generations:
            break

    families: dict[str, list] = {"canonical": []}
    if bank is not None:
        families["seen-template"] = []
        families["held-out-template"] = []
    canonical_fables: list[str] = []
    calibration: list[dict] = []
    rng = random.Random(cfg.seed)

    def _gen(prompt_text: str, idx: int) -> str:
        return generate_fable(model, tok, prompt_text, max_new_tokens=cfg.max_new_tokens,
                              min_new_tokens=cfg.min_new_tokens, do_sample=True,
                              temperature=cfg.temperature, top_k=cfg.top_k,
                              seed=cfg.seed + idx, device=device)

    for idx, spec in enumerate(specs):
        renderings = [("canonical", render_canonical_prompt(spec))]
        if bank is not None:
            renderings.append(("seen-template", render_paraphrase(bank.choose_seen(rng), spec)))
            renderings.append(("held-out-template", render_paraphrase(rng.choice(bank.held_out_templates), spec)))
        for family, prompt_text in renderings:
            fable = _gen(prompt_text, idx)
            families[family].append({
                "adherence": element_adherence(fable, spec),
                "moral": moral_delivered(fable, spec.moral, cfg.moral_threshold),
            })
            if family == "canonical":
                canonical_fables.append(fable)
                calibration.append({
                    "family": "canonical", "prompt_index": idx,
                    "requested_moral": spec.moral,
                    "ours": extract_moral(fable), "naive": naive_regex_moral(fable),
                    "fable": fable,
                })

    grid = {fam: _grid_family(recs) for fam, recs in families.items()}
    delivered = [moral_similarity(extract_moral(f), s.moral) >= cfg.moral_threshold
                 for f, s in zip(canonical_fables, specs)]
    joined = "\n".join(canonical_fables)
    metrics = {
        "perplexity": {"fable_token_loss": round(mean_loss, 4), "perplexity": round(ppl, 4),
                       "n_rows": n_ppl_rows},
        "generation": {
            "n_specs": len(canonical_fables),
            "distinct_1": round(distinct_n(joined, 1), 4),
            "distinct_2": round(distinct_n(joined, 2), 4),
            "repetition_4": round(repetition_rate(joined, 4), 4),
            "length": length_stats(canonical_fables),
            "moral_delivery_rate": round(sum(delivered) / len(delivered), 4) if delivered else 0.0,
            "moral_threshold": cfg.moral_threshold,
        },
        "adherence_grid": grid,
        "provenance": {
            "checkpoint_sha": sha256_file(Path(cfg.checkpoint) / "model.safetensors"),
            "tokenizer_sha": sha256_file(Path(cfg.tokenizer_dir) / "tokenizer.json"),
            "seed": cfg.seed,
            "versions": _versions(),
        },
    }

    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    cal_path = out_dir / "moral_calibration.jsonl"
    cal_path.write_text("".join(json.dumps(c) + "\n" for c in calibration))
    report_path = write_eval_report(out_dir, metrics)

    write_manifest(
        out_dir, "evaluate", cfg,
        [metrics_path, report_path, cal_path],
        inputs={
            "model.safetensors": Path(cfg.checkpoint) / "model.safetensors",
            "tokenizer.json": Path(cfg.tokenizer_dir) / "tokenizer.json",
        },
    )
