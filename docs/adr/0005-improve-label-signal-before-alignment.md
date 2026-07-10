# Improve the label signal (majority-vote relabel) before alignment; DPO ships if the gate still fails

The issue-07 alignment gate is NO-GO (labeler self-consistency 0.778 < 0.85; RM held-out
accuracy 0.590 < 0.65). Before choosing an optimization path, the issue-08 margin diagnostic
(`margins` stage, 2026-07-10) stratified the 188 held-out preferences by aggregate-score margin:

| margin | n | RM accuracy |
|---|---|---|
| [0.1, 0.3) | 26 | 0.615 |
| [0.3, 0.6) | 36 | 0.472 |
| [0.6, 1.0) | 46 | 0.587 |
| [1.0, inf) | 80 | 0.637 |

Overall 0.590 (reproduces the issue-07 record). The curve is **flat**: no monotone trend, the
mid-margin bucket is below chance, and even the widest-margin pairs stay under the 0.65 gate —
so tie-dilution alone does not explain the weak RM, and label noise is the prime suspect
(consistent with the 0.778 self-consistency). Decision: **one round of signal improvement
first** — majority-vote relabeling (3 independent `claude -p` passes per pair, prompt v3
tightened against near-tie rating inflation), voted preference derivation, RM retrain —
then re-gate honestly. PPO runs only if the re-gate passes.

## Consequences

- **Two hurdles, not one.** The gate requires BOTH voted-labeler self-consistency ≥ 0.85 AND
  retrained-RM held-out accuracy ≥ 0.65. Majority voting is measured as the instrument actually
  used: the audit computes self-consistency and position-flip rate over the *voted* preference
  (vote across the 3 caches per pair/phase), not over any single pass.
- **Pre-declared fallback (b).** If either hurdle fails after this one relabel round, no further
  signal-rescue rounds: DPO on the derived preferences ships the Aligned Model, the gate record
  stays honestly NO-GO for PPO, and the report narrative is "the pipeline gated itself off PPO;
  the pre-declared fallback shipped the Aligned Model." The relabel round then becomes report
  material (did voting move self-consistency / RM accuracy at all?).
- **Decision ids** (recorded in stage manifests via `adr_decision`): `ADR-0005a` = PPO after a
  passing re-gate; `ADR-0005b` = DPO fallback. `configs/ppo_full.yaml` / `configs/dpo_full.yaml`
  carry these ids.
- **Prompt v3** (`configs/labeler_prompt.yaml`, version bump forces fresh caches by design) adds
  one instruction against near-tie inflation; the rubric itself is unchanged. Relabeling runs on
  the maintainer's Mac (subscription auth) per the Operations note; the three caches are pushed
  to `congthanh991/tinyfables-preferences` like any artifact.
- **Derive/audit grow a voting mode** (`extra_labels`): majority winner = same non-tie preference
  from ≥2 of 3 caches; pairs with no majority are skipped and counted (`n_no_majority`); output
  ratings for a voted pair are the per-axis mean over the agreeing caches.

## Considered Options

- **DPO-primary immediately** — the diagnostic's flat curve actually favors this (a flat curve
  means the labels carry little usable RM signal), and it needs no reward model at optimization
  time. Rejected as the *first* move: one bounded relabel round directly attacks the measured
  bottleneck (labeler noise), is cheap (an unattended run on subscription auth), and preserves
  the designed PPO path; DPO remains the pre-declared fallback either way.
- **ADR-gated PPO on the 0.59 RM** — rejected: barely above coin-flip; optimizing it is
  optimizing noise, and it would require an explicit override of a gate this project built
  specifically to prevent that.
