import torch

from tinyfables.model import GPT, GPTConfig
from tinyfables.reward_model import RewardModel, save_reward_model
from tinyfables.trl_compat import ScoredModelAdapter


def _rm_dir(tmp_path):
    torch.manual_seed(0)
    backbone = GPT(GPTConfig(vocab_size=64, n_layer=1, n_head=2, d_model=32, n_ctx=32))
    rm = RewardModel(backbone)
    save_reward_model(rm, tmp_path / "rm")
    return tmp_path / "rm", rm


def test_adapter_satisfies_trl_reward_contract(tmp_path):
    rm_dir, rm = _rm_dir(tmp_path)
    adapter = ScoredModelAdapter.from_reward_model_dir(rm_dir)
    assert adapter.base_model_prefix == "backbone"
    backbone = getattr(adapter, adapter.base_model_prefix)

    ids = torch.randint(2, 64, (2, 6))
    mask = torch.ones_like(ids)
    mask[1, 4:] = 0  # right-padded row
    position_ids = mask.cumsum(1) - mask.long()
    with torch.no_grad():
        out = backbone(
            input_ids=torch.masked_fill(ids, ~mask.bool(), 0),
            attention_mask=mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = adapter.score(out.hidden_states[-1])
    assert logits.shape == (2, 6, 1)

    # pooled score at last non-pad == the original RewardModel's score
    with torch.no_grad():
        want = rm(torch.masked_fill(ids, ~mask.bool(), 0), mask)
    last = mask.sum(1) - 1
    got = logits[torch.arange(2), last].squeeze(-1)
    assert torch.allclose(want, got, atol=1e-5)
