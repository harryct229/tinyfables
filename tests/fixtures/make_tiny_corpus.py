"""Deterministically generate the committed 24-row toy corpus.
Run from the repo root:  python tests/fixtures/make_tiny_corpus.py"""

import itertools
import json
from pathlib import Path

CHARACTERS = ["a shy octopus", "a stubborn raccoon", "a persuasive firefly", "a patient tortoise"]
SETTINGS = ["a quiet tide pool", "a misty marsh", "a deep canyon"]
MORALS = ["courage grows by small steps", "timely help earns lasting loyalty"]

# The real ds-tf1-en-3m canonical prompt (single-band: age group B, 4-7 years,
# ~250 words), so the toy corpus exercises the same template the real model sees.
# Must stay byte-identical to prompts.render_canonical_prompt (guarded by
# test_full_spec_matches_fixture_canonical_prompt).
PROMPT_TMPL = (
    "Create a fable based on the following elements. Weave them naturally into a story:\n"
    "  - Main Character: {character}\n"
    "  - Setting: {setting}\n"
    "  - Challenge: doubting oneself\n"
    "  - Outcome: a friend helps just in time\n"
    "  - Teaching: {moral}\n"
    "The fable should:\n"
    "  - Be appropriate for age group B (4-7 years)\n"
    "  - Use simple vocabulary that 4-7 year olds can understand\n"
    "  - Use concrete rather than abstract language\n"
    "  - Begin with vivid scene-setting\n"
    "  - Not use names for the characters, instead use the trait and character\n"
    "  - Include meaningful but simple dialogue\n"
    "  - Show (don't tell) the character's growth\n"
    "  - End with a clear connection to the moral\n"
    "Keep the story concise but engaging, around 250 words."
)

FABLE_TMPL = (
    "Once, in {setting}, there lived {character} who doubted every step. "
    "One grey morning a storm rolled in, and the little one froze with fear. "
    "A friend arrived just in time, and together they found the way home. "
    "From that day on, they practiced one small brave thing each morning.\n\n"
    "**The Moral:** {moral}"
)


def main(out: Path | None = None) -> None:
    if out is None:
        out = Path(__file__).parent / "tiny_corpus.jsonl"
    rows = []
    for character, setting, moral in itertools.product(CHARACTERS, SETTINGS, MORALS):
        rows.append(
            {
                "prompt": PROMPT_TMPL.format(character=character, setting=setting, moral=moral),
                "fable": FABLE_TMPL.format(character=character, setting=setting, moral=moral),
            }
        )
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
