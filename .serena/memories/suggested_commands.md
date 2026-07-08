# Suggested Commands

Run from repo root unless noted.

- Full tests: `.venv/bin/pytest -q`
- Focused tests: `.venv/bin/pytest tests/<file>.py -v`
- Toy chain stages:
  - `.venv/bin/python -m tinyfables run tokenizer --config configs/tokenizer_toy.yaml --out runs/tokenizer_toy`
  - `.venv/bin/python -m tinyfables run prep --config configs/prep_toy.yaml --out runs/prep_toy`
  - `.venv/bin/python -m tinyfables run pretrain --config configs/pretrain_toy.yaml --out runs/pretrain_toy`
- Generate: `.venv/bin/python -m tinyfables generate --checkpoint <ckpt> --tokenizer <tok_dir> ...`
- Prefer `rg` / `rg --files` for search.
- SDD helpers: `/Users/thanh/.codex/plugins/cache/claude-plugins-official/superpowers/6.1.1/skills/subagent-driven-development/scripts/task-brief` and `review-package`.