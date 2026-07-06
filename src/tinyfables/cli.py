"""One CLI for every stage: python -m tinyfables run <stage> --config c.yaml --out dir"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from tinyfables.config import load_config
from tinyfables.stages import REGISTRY


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinyfables")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run a pipeline stage")
    run_p.add_argument("stage", choices=sorted(REGISTRY))
    run_p.add_argument("--config", required=True, help="path to the stage's YAML config")
    run_p.add_argument("--out", required=True, help="output directory for artifacts")
    args = parser.parse_args(argv)

    config_cls, module_path = REGISTRY[args.stage]
    cfg = load_config(args.config, config_cls)
    run_fn = importlib.import_module(module_path).run
    run_fn(cfg, Path(args.out))
    print(f"[tinyfables] stage '{args.stage}' complete -> {args.out}")
    return 0
