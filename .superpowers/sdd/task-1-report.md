# Task 1 Report: `align` extra + HF tokenizer wrapper

## Summary

Successfully implemented the TRL tokenizer wrapper for issue 08. The implementation adds a PreTrainedTokenizerFast wrapper around the raw tokenizers.Tokenizer with lazy imports to keep the light import path clear of torch/transformers.

## What Was Implemented

1. **Added `align` optional dependency to `pyproject.toml`**: Added `align = ["trl>=1.8"]` to `[project.optional-dependencies]` to enable TRL trainer support.

2. **Created `src/tinyfables/hf_tokenizer.py`**: Implemented `load_hf_tokenizer(tokenizer_dir: str | Path, padding_side: str = "right") -> PreTrainedTokenizerFast` with:
   - Lazy imports of `tokenizers.Tokenizer` and `transformers.PreTrainedTokenizerFast` to preserve light import path
   - Loads raw tokenizer from `tokenizer.json` in the given directory
   - Sets `pad_token` to PAD (`<|pad|>`) and `eos_token` to EOT (`<|endoftext|>`)
   - Configurable `padding_side` with default "right"
   - No post-processor added; token IDs match raw tokenizer exactly

3. **Created `tests/test_hf_tokenizer.py`**: Two comprehensive tests:
   - `test_wrapper_matches_raw_tokenizer_and_sets_specials`: Verifies token ID matching, special token configuration, and default padding side
   - `test_wrapper_left_padding_side`: Tests left-padding configuration with batched input and attention mask verification

## Test Results

### RED phase (before implementation):
```
ModuleNotFoundError: No module named 'tinyfables.hf_tokenizer'
ERROR during collection
```

### GREEN phase (after implementation):
```
tests/test_hf_tokenizer.py::test_wrapper_matches_raw_tokenizer_and_sets_specials PASSED [ 50%]
tests/test_hf_tokenizer.py::test_wrapper_left_padding_side PASSED        [100%]

2 passed in 1.19s
```

### Full test suite:
```
244 passed, 1 deselected in 6.37s
```

## Light Import Verification

```
python -c "import sys, tinyfables.cli, tinyfables.hf_tokenizer; \
  assert 'torch' not in sys.modules and 'transformers' not in sys.modules; \
  print('light')"
→ light
```

Verified: torch and transformers are NOT loaded at import time. Both are lazily loaded only when `load_hf_tokenizer()` is called.

## Files Changed

| File | Changes |
|------|---------|
| `pyproject.toml` | Added `align = ["trl>=1.8"]` to `[project.optional-dependencies]` |
| `src/tinyfables/hf_tokenizer.py` | New module with `load_hf_tokenizer()` function |
| `tests/test_hf_tokenizer.py` | New test file with 2 tests |

## Commit

- **SHA**: e9bc7f1
- **Message**: `feat(align): trl extra + PreTrainedTokenizerFast wrapper`

## Self-Review Findings

✅ **Completeness**: All 7 steps from the brief completed exactly as specified
✅ **Code Quality**: Follows repo conventions (lazy imports, clear docstring, type hints)
✅ **Test Coverage**: Two tests verify core functionality (matching, specials, padding)
✅ **No Regressions**: Full test suite passes (244 tests)
✅ **Light Imports**: Verified torch/transformers stay out of sys.modules at import
✅ **YAGNI**: No unnecessary code; minimal, focused implementation
✅ **Integration Ready**: `load_hf_tokenizer` is the exact interface expected by Tasks 5–6

## Concerns

None. The implementation is straightforward and follows the brief precisely. The lazy import strategy is clean and effective.
