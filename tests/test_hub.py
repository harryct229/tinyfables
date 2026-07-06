from tinyfables import hub
from tinyfables.cli import main


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))


def test_push_model_creates_repo_and_uploads_folder(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    api = FakeApi()
    repo = hub.push_model_to_hub(tmp_path, "user/tinyfables-13m-base", api=api)
    assert repo == "user/tinyfables-13m-base"
    names = [c[0] for c in api.calls]
    assert names == ["create_repo", "upload_folder"]
    create_kw = api.calls[0][1]
    assert create_kw["repo_id"] == "user/tinyfables-13m-base" and create_kw["private"] is True
    folder_kw = api.calls[1][1]
    assert "optimizer.pt" in folder_kw["ignore_patterns"]


def test_push_tokenizer_uploads_tokenizer_json(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")
    api = FakeApi()
    hub.push_tokenizer_to_hub(tmp_path, "user/tinyfables-tokenizer", private=False, api=api)
    up = [c for c in api.calls if c[0] == "upload_file"][0][1]
    assert up["path_in_repo"] == "tokenizer.json"
    assert api.calls[0][1]["private"] is False


def test_cli_push_dispatches(tmp_path, monkeypatch, capsys):
    recorded = {}

    def fake_push_model(path, repo, *, private=True, api=None):
        recorded["args"] = (str(path), repo, private)
        return repo

    monkeypatch.setattr(hub, "push_model_to_hub", fake_push_model)
    rc = main(["push", "--kind", "model", "--path", str(tmp_path), "--repo", "user/m"])
    assert rc == 0
    assert recorded["args"] == (str(tmp_path), "user/m", True)
    assert "pushed model" in capsys.readouterr().out
