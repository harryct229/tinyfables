from pathlib import Path

from tinyfables.config import PrepConfig, load_config
from tinyfables.paraphrases import load_bank

REPO = Path(__file__).resolve().parents[1]


def test_prep_full_trains_with_augmentation():
    cfg = load_config(REPO / "configs" / "prep_full.yaml", PrepConfig)
    # The main Base Model MUST train with augmentation, or issue-10's no-aug
    # ablation sibling would be identical to the main model.
    assert cfg.paraphrase_bank == "configs/paraphrases.yaml"
    assert 0.10 <= cfg.paraphrase_coverage <= 0.20  # design range
    bank = load_bank(REPO / cfg.paraphrase_bank)
    assert bank.seen_templates  # bank actually usable for training


def test_pretrain_full_enables_amp_and_checkpointing():
    from tinyfables.config import PretrainConfig

    cfg = load_config(REPO / "configs" / "pretrain_full.yaml", PretrainConfig)
    assert cfg.amp is True
    assert cfg.ckpt_every > 0 and cfg.log_every > 0
    assert cfg.ckpt_dir is not None
    assert cfg.trackio_project  # non-empty -> live dashboard on Colab
    # geometry is unchanged from the design table
    assert (cfg.n_layer, cfg.d_model, cfg.n_head, cfg.n_ctx) == (6, 384, 6, 1024)
    assert cfg.batch_size == 16 and cfg.steps == 20000


def test_benchmark_full_is_valid():
    from tinyfables.config import BenchmarkConfig

    cfg = load_config(REPO / "configs" / "benchmark_full.yaml", BenchmarkConfig)
    assert cfg.measure_steps > 0 and cfg.token_budget >= 200_000_000
    assert (cfg.n_layer, cfg.d_model, cfg.n_ctx) == (6, 384, 1024)


def test_pairgen_full_targets_2000_pairs_at_temp_0_9():
    from tinyfables.config import PairgenConfig

    cfg = load_config(REPO / "configs" / "pairgen_full.yaml", PairgenConfig)
    assert cfg.checkpoint == "runs/base_model"
    assert cfg.tokenizer_dir == "runs/tokenizer_full"
    assert cfg.source.hf_dataset == "klusai/ds-tf1-en-3m"
    assert cfg.source.hf_split == "validation"
    assert cfg.source.max_rows == 8000
    assert cfg.n_ctx == 1024
    assert cfg.n_pairs == 2000
    assert cfg.max_new_tokens == 320
    assert cfg.min_new_tokens == 80
    assert cfg.temperature == 0.9
    assert cfg.top_k == 50
    assert cfg.seed == 0
    assert cfg.device == "auto"


def test_label_full_uses_rubric_and_versioned_prompt():
    from tinyfables.config import LabelConfig

    cfg = load_config(REPO / "configs" / "label_full.yaml", LabelConfig)
    assert cfg.pairs == "runs/pairgen_base/pairs.jsonl"
    assert cfg.rubric == "RUBRIC.md"
    assert cfg.labeler_prompt == "configs/labeler_prompt.yaml"
    assert cfg.model == "claude-sonnet-5"
    assert cfg.batch_size == 10
    assert cfg.workers == 4
    assert cfg.swap_fraction == 0.10
    assert cfg.calibration_size == 30
    assert cfg.seed == 0


def test_derive_and_audit_full_configs_load():
    from tinyfables.config import AuditConfig, DeriveConfig

    d = load_config(REPO / "configs" / "derive_full.yaml", DeriveConfig)
    assert d.labels == "runs/labels_base/labels.jsonl"
    assert d.pairs == "runs/pairgen_base/pairs.jsonl"
    assert d.weight_delta == 0.1
    assert d.held_out_fraction == 0.10
    assert d.seed == 0
    a = load_config(REPO / "configs" / "audit_full.yaml", AuditConfig)
    assert a.labels == "runs/labels_base/labels.jsonl"
    assert a.self_consistency_gate == 0.85
    assert a.position_swap_review_threshold == 0.5


def test_reward_and_gate_full_configs_load():
    from tinyfables.config import GateConfig, RewardTrainConfig, load_config

    reward = load_config(REPO / "configs" / "reward_full.yaml", RewardTrainConfig)
    assert reward.preferences == "runs/derive_base/preferences.jsonl"
    assert reward.base_checkpoint == "runs/base_model"
    assert reward.tokenizer_dir == "runs/tokenizer_full"
    assert reward.n_ctx == 1024
    assert reward.batch_size == 16
    assert reward.steps == 1000
    assert reward.curve_sizes == [100, 500, 1000, 2000]
    assert reward.curve_steps == 400
    assert reward.accuracy_gate == 0.65
    assert reward.amp is True

    gate = load_config(REPO / "configs" / "gate_full.yaml", GateConfig)
    assert gate.audit == "runs/audit_base/audit.json"
    assert gate.reward_summary == "runs/reward_base/reward_summary.json"
    assert gate.rm_accuracy_gate == 0.65
    assert gate.require_labeler_self_consistency is True


def test_issue10_noaug_configs_preserve_the_controlled_experiment():
    from dataclasses import asdict

    from tinyfables.config import EvalConfig, PretrainConfig

    prep_aug = asdict(load_config(REPO / "configs" / "prep_full.yaml", PrepConfig))
    prep_noaug = asdict(load_config(REPO / "configs" / "prep_noaug_full.yaml", PrepConfig))
    prep_diffs = {
        key: (prep_aug[key], prep_noaug[key])
        for key in prep_aug
        if prep_aug[key] != prep_noaug[key]
    }
    assert prep_diffs == {"paraphrase_coverage": (0.15, 0.0)}
    assert prep_noaug["tokenizer_dir"] == "runs/tokenizer_full"
    assert prep_noaug["source"] == {
        "jsonl_path": None,
        "hf_dataset": "klusai/ds-tf1-en-3m",
        "hf_split": "train",
        "max_rows": 450000,
    }

    pre_aug = asdict(load_config(REPO / "configs" / "pretrain_full.yaml", PretrainConfig))
    pre_noaug = asdict(
        load_config(REPO / "configs" / "pretrain_noaug_full.yaml", PretrainConfig)
    )
    operational = {"prep_dir", "ckpt_dir", "run_name", "ckpt_hub_repo"}
    scientific_aug = {key: value for key, value in pre_aug.items() if key not in operational}
    scientific_noaug = {key: value for key, value in pre_noaug.items() if key not in operational}
    assert scientific_noaug == scientific_aug
    assert {
        key: (pre_aug[key], pre_noaug[key])
        for key in operational
        if pre_aug[key] != pre_noaug[key]
    } == {
        "prep_dir": ("runs/prep_full", "runs/prep_noaug"),
        "ckpt_dir": (
            "/content/drive/MyDrive/tinyfables/ckpt_base",
            "/content/drive/MyDrive/tinyfables/ckpt_base_noaug",
        ),
        "run_name": ("base-v1", "base-noaug-v1"),
        "ckpt_hub_repo": (
            None,
            "congthanh991/tinyfables-13m-base-noaug-ckpts",
        ),
    }

    eval_aug = asdict(load_config(REPO / "configs" / "eval_full.yaml", EvalConfig))
    eval_noaug = asdict(load_config(REPO / "configs" / "eval_noaug_full.yaml", EvalConfig))
    eval_diffs = {
        key: (eval_aug[key], eval_noaug[key])
        for key in eval_aug
        if eval_aug[key] != eval_noaug[key]
    }
    assert eval_diffs == {"checkpoint": ("runs/base_model", "runs/base_noaug")}


def test_margins_full_config_loads():
    from tinyfables.config import MarginsConfig

    cfg = load_config(REPO / "configs" / "margins_full.yaml", MarginsConfig)
    assert cfg.preferences == "runs/derive_base/preferences.jsonl"
    assert cfg.reward_model_dir == "runs/rm_hub"
    assert cfg.tokenizer_dir == "runs/tokenizer_full"
    assert cfg.n_ctx == 1024
    assert cfg.batch_size == 16
    assert cfg.bucket_edges == [0.1, 0.3, 0.6, 1.0]
    assert cfg.device == "auto"
    assert cfg.seed == 0


def test_ppo_full_config_loads():
    from tinyfables.config import PPOStageConfig

    # ADR-0005a: PPO runs only after a passing re-gate; this test asserts the
    # config loads and the gate path is the honest, currently-NO-GO record —
    # not that PPO would actually be allowed to run today.
    cfg = load_config(REPO / "configs" / "ppo_full.yaml", PPOStageConfig)
    assert cfg.gate == "runs/gate_base/gate.json"
    assert cfg.preferences == "runs/derive_base/preferences.jsonl"
    assert cfg.base_checkpoint == "runs/base_model"
    assert cfg.reward_model_dir == "runs/rm_hub"
    assert cfg.tokenizer_dir == "runs/tokenizer_full"
    assert cfg.adr_decision == "ADR-0005a"
    assert cfg.n_ctx == 1024
    assert cfg.response_length == 320
    assert cfg.total_episodes == 2000
    assert cfg.batch_size == 8
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.local_rollout_forward_batch_size == 8
    assert cfg.num_ppo_epochs == 4
    assert cfg.kl_coef == 0.2
    assert cfg.lr == 3.0e-6
    assert cfg.temperature == 0.9
    assert cfg.missing_eos_penalty == 1.0
    assert cfg.n_probe_prompts == 8
    assert cfg.length_alarm_threshold == 0.25
    assert cfg.seed == 0
    assert cfg.device == "auto"
    assert cfg.fp16 is True


def test_samples_full_config_loads():
    from tinyfables.config import SamplesConfig

    cfg = load_config(REPO / "configs" / "samples_full.yaml", SamplesConfig)
    assert cfg.base_checkpoint == "runs/base_model"
    assert cfg.aligned_checkpoint == "runs/aligned_model"
    assert cfg.tokenizer_dir == "runs/tokenizer_full"
    assert cfg.source.hf_dataset == "klusai/ds-tf1-en-3m"
    assert cfg.source.hf_split == "validation"
    assert cfg.n_specs == 20
    assert cfg.max_new_tokens == 320
    assert cfg.min_new_tokens == 80
    assert cfg.temperature == 0.9
    assert cfg.top_k == 50
    assert cfg.seed == 0
    assert cfg.device == "auto"


def test_dpo_full_config_loads():
    from tinyfables.config import DPOStageConfig

    # ADR-0005b: DPO is the pre-declared fallback branch of ADR-0005; it ships
    # the Aligned Model if the re-gate still fails. This test only asserts the
    # config loads.
    cfg = load_config(REPO / "configs" / "dpo_full.yaml", DPOStageConfig)
    assert cfg.preferences == "runs/derive_base/preferences.jsonl"
    assert cfg.base_checkpoint == "runs/base_model"
    assert cfg.tokenizer_dir == "runs/tokenizer_full"
    assert cfg.adr_decision == "ADR-0005b"
    assert cfg.n_ctx == 1024
    assert cfg.beta == 0.1
    assert cfg.lr == 5.0e-6
    assert cfg.num_train_epochs == 3.0
    assert cfg.batch_size == 8
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.min_margin == 0.0
    assert cfg.logging_steps == 10
    assert cfg.n_probe_prompts == 8
    assert cfg.probe_max_new_tokens == 320
    assert cfg.temperature == 0.9
    assert cfg.length_alarm_threshold == 0.25
    assert cfg.seed == 0
    assert cfg.device == "auto"
    assert cfg.fp16 is True
