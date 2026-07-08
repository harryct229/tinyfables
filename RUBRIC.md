# TinyFables Rubric

The anchored definition of "good" for the feedback stage. A human authored this;
the AI Labeler (Claude, ADR-0004) applies it at scale. Every fable in a Preference
Pair is rated on all four Axes below, each an integer **1–5**. The preference is
derived from the fixed **Aggregate Score = 0.4·moral + 0.3·adherence + 0.2·coherence
+ 0.1·prose** (ADR-0003); exact ties are skipped. Rate each fable on its own merits
against these anchors — not relative to the other fable in the pair.

Every fable was requested for children **ages 4–7** (dataset band B), around 250 words.

---

## Axis 1 — Moral delivery (weight 0.4)

Does the fable teach the **requested** Moral, clearly and inside the story?

- 1: No discernible moral, or the moral contradicts the requested one.
- 2: A moral is gestured at but muddled, tacked on, or a different lesson than requested.
- 3: A moral is present and on-topic but generic, or only loosely the requested one
  (e.g. "be brave" when "courage grows by small steps" was asked).
- 4: The requested moral is clearly delivered — either strongly dramatized by the plot
  OR stated plainly at the end, but not both; wording may differ from the request.
- 5: The requested moral is unmistakable — dramatized by the plot AND echoed in a clear
  closing line a 4–7-year-old would grasp.
  _Example (requested "courage grows by small steps"):_ "...and each small brave step
  made her braver. **Courage grows one small step at a time.**"

## Axis 2 — Spec adherence (weight 0.3)

Are the requested Elements (main character, setting, challenge, outcome) actually used
and faithful to the request? Rate faithful **use**, not literal string matching
(verbatim presence is measured separately and mechanically).

- 1: The fable ignores most of the spec, or is about something else entirely.
- 2: Two or more Elements are missing/changed, or the outcome contradicts the request.
- 3: One Element is missing or clearly changed (e.g. the requested lamb becomes a wolf),
  the rest honored.
- 4: All four Elements present; one is only lightly touched or slightly altered.
- 5: All four requested Elements appear and drive the story, faithful to the request.

## Axis 3 — Coherence (weight 0.2)

Is it a single, sensible story that holds together from beginning to end?

- 1: Incoherent word-salad, or it abandons the narrative.
- 2: Disjointed — events don't connect, or the story stalls or loops.
- 3: Followable but loose — some repetition, a dropped thread, or a soft ending.
- 4: Coherent with a minor bump (a small logic gap or an abrupt transition).
- 5: Clear beginning-middle-end; cause and effect track; no contradictions or non-sequiturs.

## Axis 4 — Prose & age-fit (weight 0.1)

Is the language fluent and right for ages 4–7?

- 1: Broken grammar, garbled, or wholly age-inappropriate.
- 2: Frequent grammar errors, or vocabulary too abstract/adult for 4–7.
- 3: Readable but plain or occasionally clunky/repetitive; a few too-hard words.
- 4: Fluent and age-appropriate with a rare awkward phrase or slightly advanced word.
- 5: Smooth, grammatical, simple concrete vocabulary; vivid but easy — a delight read aloud.
