from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.constants import EOT, PAD
from tinyfables.reward_data import (
    collate_preference_batch,
    encode_reward_text,
    load_preferences,
    select_train_subset,
)
from tinyfables.stages import tokenizer as tokenizer_stage


FIX = Path(__file__).parent / "fixtures"
PREFS = FIX / "preferences_replay.jsonl"
CORPUS = FIX / "tiny_corpus.jsonl"


def test_load_preferences_filters_split_and_keeps_order():
    train = load_preferences(PREFS, split="train")
    held = load_preferences(PREFS, split="held_out")

    assert [row.pair_id for row in train] == [
        "pair-000000",
        "pair-000001",
        "pair-000002",
        "pair-000003",
    ]
    assert [row.pair_id for row in held] == ["pair-000004", "pair-000005"]
    assert train[0].chosen.startswith("The octopus")
    assert train[0].rejected.startswith("The octopus hid")


def test_select_train_subset_is_seeded_and_clamps_to_available():
    train = load_preferences(PREFS, split="train")
    a = select_train_subset(train, requested_size=3, seed=123)
    b = select_train_subset(train, requested_size=3, seed=123)
    c = select_train_subset(train, requested_size=99, seed=123)

    assert [x.pair_id for x in a] == [x.pair_id for x in b]
    assert len(a) == 3
    assert len(c) == 4
    assert {x.pair_id for x in c} == {x.pair_id for x in train}


def test_encode_reward_text_appends_eot_and_respects_n_ctx(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(CORPUS)), vocab_size=512, seed=0),
        tok_dir,
    )
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))

    ids = encode_reward_text(tok, "Prompt text. ", "Fable text.", n_ctx=64)

    assert ids[-1] == tok.token_to_id(EOT)
    assert len(ids) <= 64
    assert ids == tok.encode("Prompt text. ").ids + tok.encode("Fable text.").ids + [tok.token_to_id(EOT)]


def test_collate_preference_batch_pads_and_builds_attention_masks(tmp_path):
    tok_dir = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=str(CORPUS)), vocab_size=512, seed=0),
        tok_dir,
    )
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    examples = load_preferences(PREFS, split="train")[:2]

    batch = collate_preference_batch(examples, tok, n_ctx=64, device="cpu")

    assert set(batch) == {
        "chosen_input_ids",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
    }
    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["rejected_input_ids"].shape == batch["chosen_input_ids"].shape
    assert batch["chosen_attention_mask"].dtype == torch.long
    pad_id = tok.token_to_id(PAD)
    padded_positions = batch["chosen_attention_mask"] == 0
    assert torch.all(batch["chosen_input_ids"][padded_positions] == pad_id)
