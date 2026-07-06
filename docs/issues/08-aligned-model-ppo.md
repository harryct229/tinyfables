# 08 — Aligned Model: PPO with DPO fallback

**Triage**: `ready-for-agent`

## Parent

[TinyFables PRD](../prd.md)

## What to build

The alignment stage: PPO via TRL (policy, frozen Base reference, and Reward Model together on
one T4 at 13M scale), KL-anchored to the Base Model, with a per-iteration length-drift alarm —
length bias is the classic reward hack and the ~250-word fable format makes it detectable —
and reward/KL curves in the tracker. Alongside it, the documented fallback: a DPO stage
consuming the same derived preferences, so a PPO failure still ships an Aligned Model (the
week-5 decision). Output: the Aligned Model pushed to the Hub, plus a before/after sample set
over fixed FableSpecs.

## Acceptance criteria

- [ ] PPO refuses to start without a gate-pass record from issue 07
- [ ] Length alarm and reward/KL curves logged every iteration
- [ ] Toy-scale PPO smoke test joins the full toy chain (prep → … → PPO → eval)
- [ ] DPO stage runs on the same preferences artifact; exercised at toy scale
- [ ] An Aligned Model exists on the Hub — via PPO, or via DPO with the fallback decision recorded
- [ ] Before/after generations for fixed FableSpecs saved as an artifact

## Blocked by

- [07 — Reward Model + gates](./07-reward-model-gates.md)
