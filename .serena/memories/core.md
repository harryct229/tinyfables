# Core

TinyFables: from-scratch moral-fable GPT + RLAIF course project. Source under `src/tinyfables/`; configs under `configs/`; design/issue queue under `docs/`.

Read `mem:tech_stack` for runtime/dependency constraints, `mem:conventions` for stage/config contracts, `mem:suggested_commands` for commands, and `mem:task_completion` before finishing coding tasks.

Project invariants:
- Stage CLI: `python -m tinyfables run <stage> --config <yaml> --out <dir>`; generation via `python -m tinyfables generate ...`.
- Stage completion marker is `manifest.json`, written last by `tinyfables.stage.write_manifest` after artifacts.
- Stage registry stores module-path strings so heavy modules import only at dispatch.
- Dataset is single-band age B / 4-7 / ~250 words; `render_canonical_prompt` raises for other ages; parse requires exact real template.
- Prompt-family artifacts: `families.bin` uint8 codes 0 canonical / 1 seen-template / 2 held-out-template; held-out templates are eval-only.
- Issue 04 Base Model exists on Hub: `congthanh991/tinyfables-13m-base`; tokenizer: `congthanh991/tinyfables-tokenizer`.