from tinyfables.report import write_eval_report

METRICS = {
    "perplexity": {"fable_token_loss": 1.4466, "perplexity": 4.249, "n_rows": 500},
    "generation": {
        "n_specs": 50,
        "distinct_1": 0.42,
        "distinct_2": 0.81,
        "repetition_4": 0.05,
        "length": {"mean": 248.0, "median": 249, "min": 210, "max": 260},
        "moral_delivery_rate": 0.6,
        "moral_threshold": 0.3,
    },
    "adherence_grid": {
        "canonical": {
            "character": 0.9,
            "setting": 0.5,
            "challenge": 0.2,
            "outcome": 0.3,
            "overall": 0.475,
            "moral_delivery": 0.62,
            "n": 50,
        },
        "seen-template": {
            "character": 0.88,
            "setting": 0.48,
            "challenge": 0.18,
            "outcome": 0.28,
            "overall": 0.455,
            "moral_delivery": 0.6,
            "n": 50,
        },
        "held-out-template": {
            "character": 0.8,
            "setting": 0.44,
            "challenge": 0.16,
            "outcome": 0.26,
            "overall": 0.415,
            "moral_delivery": 0.55,
            "n": 50,
        },
    },
    "provenance": {
        "checkpoint_sha": "abc123def4567890",
        "tokenizer_sha": "def789abc1234560",
        "seed": 0,
        "versions": {"torch": "2.11"},
    },
}


def test_write_eval_report_renders_tables(tmp_path):
    path = write_eval_report(tmp_path, METRICS)
    assert path == tmp_path / "eval_report.md"
    md = path.read_text()
    assert "4.249" in md
    assert "distinct-1" in md and "0.42" in md
    assert md.index("canonical") < md.index("seen-template") < md.index("held-out-template")
    assert "held-out-template" in md and "0.415" in md
    assert "abc123def4567890" in md
    assert "def789abc1234560" in md
