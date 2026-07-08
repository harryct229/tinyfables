# Tech Stack

- Python package using `pyproject.toml`, source layout under `src/`.
- Python >=3.10. Local env: `.venv`; shell state does not persist between commands.
- Core deps: numpy, tokenizers, pyyaml, datasets, torch, transformers 5.x.
- Optional extras: `dev` for pytest; `train` for trackio/matplotlib; `gpt2` for tokenizer comparison.
- Tests: pytest, with `pyproject.toml` deselecting `network` tests by default.
- No formatter/linter configured. Keep edits ASCII unless file already requires Unicode.
- Network/real-data operations are Colab/HF driven; ordinary tests should remain offline.