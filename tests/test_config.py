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


def test_load_pretrain_config(tmp_path):
    p = write_yaml(
        tmp_path,
        "prep_dir: runs/prep\ntokenizer_dir: runs/tok\n"
        "n_layer: 2\nd_model: 64\nn_head: 2\nn_ctx: 256\n"
        "batch_size: 4\nsteps: 50\nlr: 0.001\nseed: 0\n",
    )
    from tinyfables.config import PretrainConfig

    cfg = load_config(p, PretrainConfig)
    assert cfg.prep_dir == "runs/prep"
    assert cfg.tokenizer_dir == "runs/tok"
    assert cfg.n_layer == 2 and cfg.d_model == 64 and cfg.n_head == 2
    assert cfg.n_ctx == 256 and cfg.batch_size == 4 and cfg.steps == 50
    assert cfg.resume_from is None  # default


def test_pretrain_config_rejects_unknown_key(tmp_path):
    from tinyfables.config import PretrainConfig

    p = write_yaml(tmp_path, "prep_dir: a\ntokenizer_dir: b\nlayers: 6\n")
    with pytest.raises(KeyError):
        load_config(p, PretrainConfig)
