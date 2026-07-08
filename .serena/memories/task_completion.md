# Task Completion

Before considering coding work done:
- Run focused pytest for changed behavior.
- Run full suite with `.venv/bin/pytest -q` before commit/merge unless task is docs-only and explicitly scoped otherwise.
- Ensure no unintended tracked/untracked files are staged; `.superpowers/`, `.venv/`, and `runs/` are ignored scratch.
- Commit with a concise subject referencing the issue when applicable.
- For SDD work: implementer writes report to `.superpowers/sdd/task-N-report.md`; controller generates review package from the task BASE to HEAD; reviewer must approve spec + quality before next task.
- At issue completion, update `docs/issues/README.md` only when acceptance criteria are actually satisfied.