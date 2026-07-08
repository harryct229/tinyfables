import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from tinyfables.config import SourceSpec, TokenizerConfig
from tinyfables.constants import EOT
from tinyfables.data import read_rows
from tinyfables.model import GPT, GPTConfig
from tinyfables.perplexity import fable_token_perplexity
from tinyfables.stages import tokenizer as tokenizer_stage

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def _tok(tmp_path):
    out = tmp_path / "tok"
    tokenizer_stage.run(
        TokenizerConfig(source=SourceSpec(jsonl_path=FIXTURE), vocab_size=512, seed=0),
        out,
    )
    return Tokenizer.from_file(str(out / "tokenizer.json"))


def test_perplexity_is_exp_of_mean_loss_and_finite(tmp_path):
    tok = _tok(tmp_path)
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2, d_model=64, n_ctx=512)
    ).eval()
    rows = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)
    mean_loss, ppl, n_used = fable_token_perplexity(
        model, tok, rows, n_ctx=512, device="cpu", max_rows=3
    )
    assert math.isfinite(mean_loss) and math.isfinite(ppl)
    assert abs(ppl - math.exp(mean_loss)) < 1e-4
    assert ppl > 1.0
    assert n_used == 3


def test_perplexity_matches_manual_completion_only_loss(tmp_path):
    tok = _tok(tmp_path)
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2, d_model=64, n_ctx=512)
    ).eval()
    row = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)[0]
    mean_loss, _, n_used = fable_token_perplexity(
        model, tok, [row], n_ctx=512, device="cpu", max_rows=1
    )
    assert n_used == 1
    p = tok.encode(row["prompt"]).ids
    f = tok.encode(row["fable"]).ids + [tok.token_to_id(EOT)]
    ids = torch.tensor([p + f])
    labels = ids.clone()
    labels[0, : len(p)] = -100
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1, :]
    tgt = labels[0, 1:]
    keep = tgt != -100
    expected = F.cross_entropy(logits[keep], tgt[keep]).item()
    assert abs(mean_loss - expected) < 1e-4


def test_perplexity_reports_only_scored_rows_when_skipping_overlong(tmp_path):
    tok = _tok(tmp_path)
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2, d_model=64, n_ctx=512)
    ).eval()
    rows = read_rows(SourceSpec(jsonl_path=FIXTURE), seed=0)
    overlong = {"prompt": rows[0]["prompt"] * 20, "fable": rows[0]["fable"]}
    mean_loss, ppl, n_used = fable_token_perplexity(
        model, tok, [overlong, rows[0]], n_ctx=512, device="cpu", max_rows=2
    )
    assert n_used == 1
    assert math.isfinite(mean_loss) and math.isfinite(ppl)
