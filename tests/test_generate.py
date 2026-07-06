from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.generate import encode_prompt, generate_fable
from tinyfables.model import GPT, GPTConfig
from tinyfables.prompts import FableSpec, render_canonical_prompt
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")

SPEC = FableSpec(
    character="a shy octopus",
    setting="a quiet tide pool",
    challenge="doubting oneself",
    outcome="a friend helps just in time",
    moral="courage grows by small steps",
)


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    out = tmp_path_factory.mktemp("tok")
    tokenizer_stage.run(TokenizerConfig(source=SourceSpec(jsonl_path=FIXTURE), vocab_size=512, seed=0), out)
    return Tokenizer.from_file(str(out / "tokenizer.json"))


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2, d_model=64, n_ctx=512)).eval()


def test_encode_prompt_matches_prep_separate_encoding(tok):
    # prep.py does tok.encode(prompt).ids for the prompt alone; the contract is
    # that generation encodes the prompt exactly the same way.
    prompt = render_canonical_prompt(SPEC)
    assert encode_prompt(tok, SPEC) == tok.encode(prompt).ids
    assert encode_prompt(tok, "Write a fable about a brave mouse.") == tok.encode(
        "Write a fable about a brave mouse."
    ).ids


def test_generate_from_fablespec_returns_text(model, tok):
    out = generate_fable(model, tok, SPEC, max_new_tokens=24, min_new_tokens=8, seed=0)
    assert isinstance(out, str) and len(out) > 0


def test_generate_from_free_text_returns_text(model, tok):
    out = generate_fable(model, tok, "Write a fable about a brave mouse.", max_new_tokens=24, min_new_tokens=8, seed=0)
    assert isinstance(out, str) and len(out) > 0


def test_generation_strips_at_endoftext(model, tok):
    from tinyfables.constants import EOT

    out = generate_fable(model, tok, SPEC, max_new_tokens=24, min_new_tokens=8, seed=0)
    assert EOT not in out  # the endoftext marker is never part of the returned fable
