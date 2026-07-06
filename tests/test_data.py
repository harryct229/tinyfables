from pathlib import Path

import pytest

from tinyfables.config import SourceSpec
from tinyfables.data import read_rows

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def test_reads_all_rows_with_expected_keys():
    rows = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)
    assert len(rows) == 24
    assert all(set(r) == {"prompt", "fable"} for r in rows)


def test_shuffle_is_deterministic_and_seed_sensitive():
    a = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)
    b = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)
    c = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=1)
    assert a == b
    assert a != c  # 24 rows: astronomically unlikely to collide


def test_max_rows_truncates():
    rows = read_rows(SourceSpec(jsonl_path=FIXTURE, max_rows=5), seed=0)
    assert len(rows) == 5


@pytest.mark.network
def test_hf_streaming_source_smoke():
    rows = read_rows(
        SourceSpec(hf_dataset="klusai/ds-tf1-en-3m", hf_split="train", max_rows=3), seed=0
    )
    assert len(rows) == 3
    assert all(set(r) == {"prompt", "fable"} for r in rows)
