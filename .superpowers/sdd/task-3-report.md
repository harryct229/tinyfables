# Task 3 Report: TRL score adapter (reward/value contract)

## Summary

Implemented `ScoredModelAdapter`, a thin wrapper around TinyFables `RewardModel` that satisfies TRL's PPOTrainer calling convention. The adapter exposes the GPT backbone and a scalar `score` head (nn.Linear) to allow TRL's framework to compute rewards by extracting hidden states and applying the score head directly.

## Implementation Details

### Files Created

1. **src/tinyfables/trl_compat.py** (54 lines)
   - `ScoredModelAdapter(nn.Module)` class
   - `base_model_prefix = "backbone"` to enable TRL's attribute lookup pattern
   - `.backbone` (GPT instance)
   - `.score` (nn.Linear(d_model, 1))
   - `from_reward_model_dir(rm_dir, device="cpu")` classmethod to load from existing RewardModel directory layout
   - `forward()` method that calls backbone with output_hidden_states=True and applies score head

2. **tests/test_trl_compat.py** (43 lines)
   - `_rm_dir()` helper to create a temporary RewardModel and save it
   - `test_adapter_satisfies_trl_reward_contract()` comprehensive test that:
     - Creates a ScoredModelAdapter from a saved RewardModel
     - Verifies base_model_prefix attribute
     - Calls backbone with masked and position_id inputs (TRL's calling pattern)
     - Verifies score shape is (batch, seq_len, 1)
     - **Critical**: Validates that scores at last-non-pad positions exactly match the original RewardModel's pooled scores

## TDD Process

### RED (Step 2)
```bash
$ .venv/bin/pytest tests/test_trl_compat.py -v
```

**Output:**
```
ImportError while importing test module '.../tests/test_trl_compat.py'.
...
ModuleNotFoundError: No module named 'tinyfables.trl_compat'
```

✓ Test fails as expected — module doesn't exist yet.

### GREEN (Step 4)
```bash
$ .venv/bin/pytest tests/test_trl_compat.py -v
```

**Output:**
```
tests/test_trl_compat.py::test_adapter_satisfies_trl_reward_contract PASSED [100%]
============================== 1 passed in 1.19s ===============================
```

✓ Test passes after implementation.

### Full Suite Verification
```bash
$ .venv/bin/pytest -q
```

**Output:**
```
........................................................................ [ 28%]
........................................................................ [ 57%]
........................................................................ [ 86%]
..................................                                       [100%]
250 passed, 1 deselected in 6.26s
```

✓ All 250 tests pass, no regressions.

## Design Decisions

1. **Direct state_dict loading**: Load `reward_head.pt` (which is an nn.Linear state dict) directly into `.score` rather than mapping keys. This works because both are `nn.Linear(d_model, 1)` and torch's `load_state_dict()` matches by parameter name.

2. **Forward signature**: The adapter's forward() accepts `**kwargs` to absorb any TRL-specific kwargs without raising errors, following defensive programming for framework integration.

3. **Return type**: Forward returns `torch.Tensor` (the raw scores), not CausalLMOutput, because TRL's pooling happens outside the model call.

4. **Device handling**: Both backbone and adapter are moved to device, and `torch.load()` uses `map_location=device` for safe device-agnostic loading.

## Self-Review Findings

**Strengths:**
- Minimal implementation: 54 lines of clean, focused code
- Perfect contract adherence: satisfies exactly what TRL's `get_reward` / `PolicyAndValueWrapper` expect
- Reuse: leverages existing RewardModel directory layout without modification
- Test validation: test validates both interface contract AND output correctness (scores match original RewardModel)

**No issues found:**
- No type mismatches
- No device/dtype edge cases
- No state management issues

## Integration Readiness

This adapter is ready for Task 5 (TRL's PPOTrainer integration), where it will be instantiated twice:
1. As `reward_model` for computing actual rewards
2. As `value_model` for value function training

The independent instance separation ensures clean reward/value computation paths.

## Commit

```
[issue-08-aligned-model e81fa90] feat(align): ScoredModelAdapter for TRL reward/value contract
 2 files changed, 97 insertions(+)
 create mode 100644 src/tinyfables/trl_compat.py
 create mode 100644 tests/test_trl_compat.py
```

## Concerns

None. Implementation is complete, tested, and ready for integration.
