import torch

from tinyfables.model import GPT, GPTConfig


def _model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, n_ctx=32)).eval()


def test_no_mask_path_bit_identical_to_all_ones_mask():
    m = _model()
    ids = torch.randint(2, 64, (2, 7))
    with torch.no_grad():
        plain = m(ids).logits
        masked = m(ids, attention_mask=torch.ones_like(ids)).logits
    assert torch.equal(plain, masked)


def test_left_padded_forward_matches_unpadded_on_real_tokens():
    m = _model()
    ids = torch.randint(2, 64, (1, 6))
    pad = torch.zeros((1, 3), dtype=torch.long)  # TRL masked_fills pads to token id 0
    padded = torch.cat([pad, ids], dim=1)
    mask = torch.cat([torch.zeros_like(pad), torch.ones_like(ids)], dim=1)
    position_ids = mask.cumsum(1) - mask.long()
    with torch.no_grad():
        want = m(ids).logits
        got = m(padded, attention_mask=mask, position_ids=position_ids).logits[:, 3:, :]
    assert torch.allclose(want, got, atol=1e-5)


def test_output_hidden_states():
    m = _model()
    ids = torch.randint(2, 64, (2, 5))
    with torch.no_grad():
        out = m(ids, output_hidden_states=True)
    assert out.hidden_states is not None
    assert out.hidden_states[-1].shape == (2, 5, 32)
    # default stays None (CausalLMOutput contract unchanged for existing callers)
    assert m(ids).hidden_states is None


def test_generate_is_cache_proof():
    m = _model()
    ids = torch.randint(2, 64, (1, 5))
    with torch.no_grad():
        no_cache = m.generate(ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=0, eos_token_id=0)
        cached = m.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True, pad_token_id=0, eos_token_id=0)
    assert torch.equal(no_cache, cached)


def test_generate_handles_left_padded_batch():
    m = _model()
    a = torch.randint(2, 64, (1, 4))
    b = torch.randint(2, 64, (1, 7))
    pad_id = 1
    width = 7
    queries = torch.full((2, width), pad_id, dtype=torch.long)
    queries[0, width - 4:] = a
    queries[1, :] = b
    mask = queries != pad_id
    ids = torch.masked_fill(queries, ~mask, 0)
    with torch.no_grad():
        batch_out = m.generate(ids, attention_mask=mask, max_new_tokens=6, do_sample=False, pad_token_id=pad_id, eos_token_id=None)
        solo_out = m.generate(a, max_new_tokens=6, do_sample=False, pad_token_id=pad_id, eos_token_id=None)
    assert torch.equal(batch_out[0, width:], solo_out[0, 4:])
