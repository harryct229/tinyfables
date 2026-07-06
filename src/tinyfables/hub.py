"""Hub push utilities. huggingface_hub is imported lazily (keeps package import
light and offline). Every function takes an injectable `api` so tests exercise the
call sequence without touching the network. Repos default to private (course
artifacts go public deliberately, via --public / private=False)."""

from __future__ import annotations

from pathlib import Path


def _default_api():
    from huggingface_hub import HfApi  # lazy: heavy + network-adjacent

    return HfApi()


def push_tokenizer_to_hub(tokenizer_dir, repo_id, *, private=True, api=None) -> str:
    api = api or _default_api()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(Path(tokenizer_dir) / "tokenizer.json"),
        path_in_repo="tokenizer.json",
        repo_id=repo_id,
        repo_type="model",
    )
    return repo_id


def push_model_to_hub(model_dir, repo_id, *, private=True, api=None) -> str:
    api = api or _default_api()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["optimizer.pt", "checkpoint_state.json"],
    )
    return repo_id


def upload_checkpoint_dir(ckpt_dir, repo_id, *, private=True, api=None) -> str:
    api = api or _default_api()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(ckpt_dir), repo_id=repo_id, repo_type="model")
    return repo_id
