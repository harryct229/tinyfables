import dataclasses

import pytest

from tinyfables.config import PrepConfig, SourceSpec, TokenizerConfig, load_config


def write_yaml(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_load_tokenizer_config(tmp_path):
    p = write_yaml(
        tmp_path,
        "source:\n  jsonl_path: corpus.jsonl\n  max_rows: 10\n"
        "vocab_size: 512\nseed: 7\ncompare_gpt2: false\n",
    )
    cfg = load_config(p, TokenizerConfig)
    assert cfg.source.jsonl_path == "corpus.jsonl"
    assert cfg.source.max_rows == 10
    assert cfg.vocab_size == 512
    assert cfg.seed == 7


def test_load_prep_config(tmp_path):
    p = write_yaml(
        tmp_path,
        "source:\n  jsonl_path: corpus.jsonl\n"
        "tokenizer_dir: runs/tok\nwindow: 256\nseed: 0\n",
    )
    cfg = load_config(p, PrepConfig)
    assert cfg.tokenizer_dir == "runs/tok"
    assert cfg.window == 256


def test_configs_are_frozen(tmp_path):
    p = write_yaml(tmp_path, "source:\n  jsonl_path: c.jsonl\nvocab_size: 64\n")
    cfg = load_config(p, TokenizerConfig)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.vocab_size = 999


def test_unknown_key_rejected(tmp_path):
    p = write_yaml(tmp_path, "source:\n  jsonl_path: c.jsonl\nvocabsize: 64\n")
    with pytest.raises(KeyError):
        load_config(p, TokenizerConfig)


def test_source_requires_exactly_one_backend():
    with pytest.raises(ValueError):
        SourceSpec()  # neither jsonl nor hf
    with pytest.raises(ValueError):
        SourceSpec(jsonl_path="a.jsonl", hf_dataset="klusai/ds-tf1-en-3m")  # both
