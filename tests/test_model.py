import math

import pytest
import torch
import torch.nn.functional as F

from tinyfables.model import GPT, GPTConfig


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, n_ctx=32)).eval()


def test_real_config_param_count_and_tying():
    m = GPT(GPTConfig())  # design defaults: 6/384/6/1024, vocab 8192
    unique = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    assert 13_500_000 < unique < 14_500_000  # measured 14,186,496
    assert m.head.weight.data_ptr() == m.tok.weight.data_ptr()  # shared storage


def test_forward_shapes():
    m = tiny()
    ids = torch.randint(0, 64, (3, 10))
    out = m(input_ids=ids)
    assert out.logits.shape == (3, 10, 64)
    assert out.loss is None


def test_future_tokens_do_not_change_current_logits():
    m = tiny()
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        base = m(input_ids=ids).logits
        perturbed = ids.clone()
        perturbed[0, -1] = (perturbed[0, -1] + 1) % 64
        after = m(input_ids=perturbed).logits
    assert torch.allclose(base[0, :9], after[0, :9], atol=1e-6)   # prefix unchanged
    assert not torch.allclose(base[0, 9], after[0, 9], atol=1e-6)  # perturbation is real


def test_fp16_forward_is_shape_and_dtype_stable():
    m = tiny().half()
    ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        out = m(input_ids=ids).logits
    assert out.dtype == torch.float16
    assert out.shape == (2, 16, 64)
    assert torch.isfinite(out).all()


def test_masked_prompt_tokens_contribute_zero_loss():
    m = tiny()
    ids = torch.randint(0, 64, (2, 8))
    labels = ids.clone()
    labels[:, :4] = -100  # mask the "prompt" half
    with torch.no_grad():
        out = m(input_ids=ids, labels=labels)
        shift_logits = out.logits[:, :-1].reshape(-1, 64)
        shift_labels = labels[:, 1:].reshape(-1)
        keep = shift_labels != -100
        manual = F.cross_entropy(shift_logits[keep], shift_labels[keep])
    assert torch.allclose(out.loss, manual, atol=1e-6)  # masked targets add exactly nothing


def test_save_load_round_trip_preserves_logits_and_tie(tmp_path):
    m = tiny()
    ids = torch.randint(0, 64, (1, 8))
    m.save_pretrained(tmp_path / "ckpt")
    m2 = GPT.from_pretrained(tmp_path / "ckpt").eval()
    assert m2.head.weight.data_ptr() == m2.tok.weight.data_ptr()
    with torch.no_grad():
        assert torch.allclose(m(input_ids=ids).logits, m2(input_ids=ids).logits, atol=1e-5)
