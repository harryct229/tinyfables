from types import SimpleNamespace

from tinyfables import cli


def test_cli_run_dispatches_feedback_stages_lazily(tmp_path, monkeypatch):
    seen = []
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("unused: true\n")

    def fake_load_config(path, config_cls):
        seen.append(("load", config_cls.__name__, str(path)))
        return SimpleNamespace(stage=config_cls.__name__)

    def fake_import_module(module_path):
        stage_name = module_path.rsplit(".", 1)[-1]

        class FakeModule:
            @staticmethod
            def run(config_obj, out_dir):
                seen.append(("run", stage_name, config_obj.stage, str(out_dir)))

        return FakeModule()

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    for stage in ("pairgen", "label", "derive", "audit", "reward", "gate"):
        out_dir = tmp_path / stage
        assert cli.main(["run", stage, "--config", str(cfg), "--out", str(out_dir)]) == 0

    assert seen == [
        ("load", "PairgenConfig", str(cfg)),
        ("run", "pairgen", "PairgenConfig", str(tmp_path / "pairgen")),
        ("load", "LabelConfig", str(cfg)),
        ("run", "label", "LabelConfig", str(tmp_path / "label")),
        ("load", "DeriveConfig", str(cfg)),
        ("run", "derive", "DeriveConfig", str(tmp_path / "derive")),
        ("load", "AuditConfig", str(cfg)),
        ("run", "audit", "AuditConfig", str(tmp_path / "audit")),
        ("load", "RewardTrainConfig", str(cfg)),
        ("run", "reward", "RewardTrainConfig", str(tmp_path / "reward")),
        ("load", "GateConfig", str(cfg)),
        ("run", "gate", "GateConfig", str(tmp_path / "gate")),
    ]
