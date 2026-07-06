"""One CLI for every stage plus the generation demo:
    python -m tinyfables run <stage> --config c.yaml --out dir
    python -m tinyfables generate --checkpoint dir --tokenizer dir [--text ... | --character ... ]
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from tinyfables.config import load_config
from tinyfables.stages import REGISTRY


def _add_run_parser(sub) -> None:
    run_p = sub.add_parser("run", help="run a pipeline stage")
    run_p.add_argument("stage", choices=sorted(REGISTRY))
    run_p.add_argument("--config", required=True, help="path to the stage's YAML config")
    run_p.add_argument("--out", required=True, help="output directory for artifacts")


def _add_generate_parser(sub) -> None:
    gen_p = sub.add_parser("generate", help="write a fable from a trained checkpoint")
    gen_p.add_argument("--checkpoint", required=True, help="model checkpoint dir (save_pretrained)")
    gen_p.add_argument("--tokenizer", required=True, help="dir containing tokenizer.json")
    gen_p.add_argument("--text", help="raw instruction text (mutually exclusive with the element flags)")
    gen_p.add_argument("--character")
    gen_p.add_argument("--setting")
    gen_p.add_argument("--challenge")
    gen_p.add_argument("--outcome")
    gen_p.add_argument("--moral")
    gen_p.add_argument("--age-range", default="4-7")
    gen_p.add_argument("--word-count", type=int, default=250)
    gen_p.add_argument("--max-new-tokens", type=int, default=256)
    gen_p.add_argument("--min-new-tokens", type=int, default=0)
    gen_p.add_argument("--temperature", type=float, default=1.0)
    gen_p.add_argument("--top-k", type=int)
    gen_p.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    gen_p.add_argument("--seed", type=int)


def _run_stage(args) -> int:
    config_cls, module_path = REGISTRY[args.stage]
    cfg = load_config(args.config, config_cls)
    run_fn = importlib.import_module(module_path).run
    run_fn(cfg, Path(args.out))
    print(f"[tinyfables] stage '{args.stage}' complete -> {args.out}")
    return 0


def _generate(args) -> int:
    # torch/transformers imported lazily so `import tinyfables.cli` stays light.
    from tokenizers import Tokenizer

    from tinyfables.generate import generate_fable
    from tinyfables.model import GPT
    from tinyfables.prompts import FableSpec

    tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    model = GPT.from_pretrained(args.checkpoint).eval()
    if args.text is not None:
        request: object = args.text
    else:
        request = FableSpec(
            character=args.character,
            setting=args.setting,
            challenge=args.challenge,
            outcome=args.outcome,
            moral=args.moral,
            age_range=args.age_range,
            word_count=args.word_count,
        )
    fable = generate_fable(
        model,
        tok,
        request,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=args.sample,
        seed=args.seed,
    )
    print(fable)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinyfables")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_run_parser(sub)
    _add_generate_parser(sub)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _run_stage(args)
    return _generate(args)
