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


def test_prep_paraphrase_defaults():
    cfg = PrepConfig(source=SourceSpec(jsonl_path="c.jsonl"), tokenizer_dir="t")
    assert cfg.paraphrase_bank is None
    assert cfg.paraphrase_coverage == 0.0


def test_prep_loads_paraphrase_knobs(tmp_path):
    p = write_yaml(
        tmp_path,
        "source:\n  jsonl_path: corpus.jsonl\ntokenizer_dir: runs/tok\n"
        "paraphrase_bank: configs/paraphrases.yaml\nparaphrase_coverage: 0.15\n",
    )
    cfg = load_config(p, PrepConfig)
    assert cfg.paraphrase_bank == "configs/paraphrases.yaml"
    assert cfg.paraphrase_coverage == 0.15


def test_prep_rejects_coverage_out_of_range():
    with pytest.raises(ValueError):
        PrepConfig(
            source=SourceSpec(jsonl_path="c.jsonl"),
            tokenizer_dir="t",
            paraphrase_bank="b.yaml",
            paraphrase_coverage=1.5,
        )


def test_prep_coverage_requires_bank():
    with pytest.raises(ValueError):
        PrepConfig(
            source=SourceSpec(jsonl_path="c.jsonl"),
            tokenizer_dir="t",
            paraphrase_coverage=0.2,
        )


def test_benchmark_config_defaults_and_overrides(tmp_path):
    from tinyfables.config import BenchmarkConfig, load_config

    p = tmp_path / "b.yaml"
    p.write_text("tokenizer_dir: runs/tok\nmeasure_steps: 30\ntarget_tokens_per_sec: 8000\n")
    cfg = load_config(p, BenchmarkConfig)
    assert cfg.tokenizer_dir == "runs/tok"
    assert cfg.measure_steps == 30 and cfg.target_tokens_per_sec == 8000
    assert cfg.n_layer == 6 and cfg.window == 1024 and cfg.amp is True  # design defaults


def test_pretrain_config_new_knobs_default_to_current_behavior(tmp_path):
    from tinyfables.config import PretrainConfig, load_config

    p = tmp_path / "p.yaml"
    p.write_text("prep_dir: runs/prep\ntokenizer_dir: runs/tok\n")
    cfg = load_config(p, PretrainConfig)
    assert cfg.amp is False and cfg.ckpt_dir is None and cfg.ckpt_every == 0
    assert cfg.log_every == 0 and cfg.keep_last_k == 2
    assert cfg.trackio_project is None and cfg.run_name is None and cfg.ckpt_hub_repo is None


def test_pretrain_config_accepts_issue04_knobs(tmp_path):
    from tinyfables.config import PretrainConfig, load_config

    p = tmp_path / "p.yaml"
    p.write_text(
        "prep_dir: runs/prep\ntokenizer_dir: runs/tok\n"
        "amp: true\nckpt_dir: /content/drive/MyDrive/ckpt\nckpt_every: 500\n"
        "log_every: 50\ntrackio_project: tinyfables\nrun_name: base-v1\n"
    )
    cfg = load_config(p, PretrainConfig)
    assert cfg.amp is True and cfg.ckpt_every == 500 and cfg.log_every == 50
    assert cfg.ckpt_dir.endswith("/ckpt") and cfg.run_name == "base-v1"


def test_benchmark_config_rejects_unknown_key(tmp_path):
    import pytest

    from tinyfables.config import BenchmarkConfig, load_config

    p = tmp_path / "b.yaml"
    p.write_text("tokenizer_dir: runs/tok\nbogus: 1\n")
    with pytest.raises(KeyError):
        load_config(p, BenchmarkConfig)
