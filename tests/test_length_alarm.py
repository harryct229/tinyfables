import torch

from tinyfables.length_alarm import length_drift, probe_lengths
from tinyfables.model import GPT, GPTConfig


def test_length_drift_flags_beyond_threshold():
    calm = length_drift(250.0, 260.0, threshold=0.25)
    assert calm["alarm"] is False and abs(calm["drift_pct"] - 0.04) < 1e-9
    loud = length_drift(250.0, 100.0, threshold=0.25)
    assert loud["alarm"] is True and abs(loud["drift_pct"] - 0.6) < 1e-9
    assert length_drift(0.0, 10.0, threshold=0.25) == {
        "baseline_mean_words": 0.0,
        "current_mean_words": 10.0,
        "drift_pct": 0.0,
        "alarm": False,
    }


def test_probe_lengths_is_deterministic_given_seed(tmp_path):
    from tinyfables.config import SourceSpec, TokenizerConfig
    from tinyfables.stages import tokenizer as tokenizer_stage
    from pathlib import Path
    from tokenizers import Tokenizer

    fixture = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")
    tokenizer_stage.run(TokenizerConfig(source=SourceSpec(jsonl_path=fixture), vocab_size=512, seed=0), tmp_path / "tok")
    tok = Tokenizer.from_file(str(tmp_path / "tok" / "tokenizer.json"))
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=1, n_head=2, d_model=32, n_ctx=128)).eval()

    prompts = ["Create a fable about a fox.", "Create a fable about a crow."]
    a = probe_lengths(model, tok, prompts, max_new_tokens=12, temperature=0.9, seed=7, device="cpu")
    b = probe_lengths(model, tok, prompts, max_new_tokens=12, temperature=0.9, seed=7, device="cpu")
    assert a == b
    assert a["n_prompts"] == 2
    assert a["mean_new_tokens"] <= 12
