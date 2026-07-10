from pathlib import Path

from tokenizers import Tokenizer

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.constants import EOT, PAD
from tinyfables.hf_tokenizer import load_hf_tokenizer
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def _tok_dir(tmp_path):
    out = tmp_path / "tok"
    tokenizer_stage.run(TokenizerConfig(source=SourceSpec(jsonl_path=FIXTURE), vocab_size=512, seed=0), out)
    return out


def test_wrapper_matches_raw_tokenizer_and_sets_specials(tmp_path):
    tok_dir = _tok_dir(tmp_path)
    raw = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    hf = load_hf_tokenizer(tok_dir)
    text = "Create a fable about a shy octopus."
    assert hf(text, add_special_tokens=False)["input_ids"] == raw.encode(text).ids
    assert hf.pad_token == PAD and hf.pad_token_id == raw.token_to_id(PAD)
    assert hf.eos_token == EOT and hf.eos_token_id == raw.token_to_id(EOT)
    assert hf.bos_token_id is None
    assert hf.padding_side == "right"


def test_wrapper_left_padding_side(tmp_path):
    hf = load_hf_tokenizer(_tok_dir(tmp_path), padding_side="left")
    assert hf.padding_side == "left"
    batch = hf(["a b", "a b c d e"], padding=True, return_tensors="pt", add_special_tokens=False)
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["attention_mask"][0, 0].item() == 0  # short row left-padded
