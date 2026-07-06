import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl"


def test_fixture_corpus_shape():
    rows = [json.loads(l) for l in FIXTURE.read_text().splitlines() if l.strip()]
    assert len(rows) == 24
    for r in rows:
        assert set(r) == {"prompt", "fable"}
        assert r["prompt"].startswith("Create a fable based on the following elements")
        assert "- Main Character:" in r["prompt"]
        assert "**The Moral:**" in r["fable"]
