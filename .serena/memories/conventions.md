# Conventions

- Frozen dataclass configs in `src/tinyfables/config.py`; unknown YAML keys are errors via `load_config`.
- Stage modules expose `run(cfg, out_dir) -> None`; they write artifacts first and `manifest.json` last.
- Keep torch/transformers imports lazy where possible; torch-free helper modules should not import torch at top level.
- Model wrapper targets transformers 5.x; weight tying handled via config/post_init, not manual parameter aliasing.
- Prep encodes prompt and fable separately, concatenates with no separator, masks prompt tokens, and writes `tokens.bin`/`mask.bin`.
- Generation encodes prompts exactly as prep encodes prompt alone; dataset is single-band 4-7 only.
- Paraphrase augmentation preserves Element values verbatim; held-out templates are never used in training and are eval-only.
- Tests should be deterministic and offline by default; use local fixture `tests/fixtures/tiny_corpus.jsonl` for toy scale.