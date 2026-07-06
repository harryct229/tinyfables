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
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    sub_art = sub_dir / "extra.bin"
    sub_art.write_bytes(b"x")
    cfg = TokenizerConfig(source=SourceSpec(jsonl_path="x.jsonl"), vocab_size=64)
    path = write_manifest(tmp_path, "tokenizer", cfg, [art, sub_art])
    m = json.loads(path.read_text())
    assert m["stage"] == "tokenizer"
    assert m["config"]["vocab_size"] == 64
    assert m["config"]["source"]["jsonl_path"] == "x.jsonl"
    assert set(m["artifacts"]) == {"tokenizer.json", "sub/extra.bin"}
    assert len(m["artifacts"]["tokenizer.json"]) == 64  # sha256 hex
    assert isinstance(m["created_unix"], int)
    assert m["inputs"] == {}
    assert isinstance(m["versions"], dict)


def test_write_manifest_records_inputs(tmp_path):
    art = tmp_path / "tokens.bin"
    art.write_bytes(b"abc")
    inp = tmp_path / "tokenizer.json"
    inp.write_text("{}")
    cfg = TokenizerConfig(source=SourceSpec(jsonl_path="x.jsonl"), vocab_size=64)
    path = write_manifest(tmp_path, "prep", cfg, [art], inputs={"tokenizer.json": inp})
    m = json.loads(path.read_text())
    assert m["inputs"] == {"tokenizer.json": sha256_file(inp)}
