import json

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.stage import sha256_file, write_manifest


def test_sha256_is_stable(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello fables")
    assert sha256_file(f) == sha256_file(f)
    g = tmp_path / "b.bin"
    g.write_bytes(b"hello fables!")
    assert sha256_file(f) != sha256_file(g)


def test_write_manifest(tmp_path):
    art = tmp_path / "tokenizer.json"
    art.write_text("{}")
    cfg = TokenizerConfig(source=SourceSpec(jsonl_path="x.jsonl"), vocab_size=64)
    path = write_manifest(tmp_path, "tokenizer", cfg, [art])
    m = json.loads(path.read_text())
    assert m["stage"] == "tokenizer"
    assert m["config"]["vocab_size"] == 64
    assert m["config"]["source"]["jsonl_path"] == "x.jsonl"
    assert list(m["artifacts"]) == ["tokenizer.json"]
    assert len(m["artifacts"]["tokenizer.json"]) == 64  # sha256 hex
    assert isinstance(m["created_unix"], int)
