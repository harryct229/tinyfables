from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_rubric_exists_and_anchors_all_four_axes():
    text = (REPO / "RUBRIC.md").read_text()
    for axis in ("Moral delivery", "Spec adherence", "Coherence", "Prose"):
        assert axis in text, f"RUBRIC.md missing axis: {axis}"
    # every 1-5 anchor is present (each axis defines all five scale points)
    for point in ("1", "2", "3", "4", "5"):
        assert f"- {point}:" in text or f"**{point}**" in text
    # the fixed weights appear so a labeler/report reader sees them
    for w in ("0.4", "0.3", "0.2", "0.1"):
        assert w in text


def test_labeler_prompt_is_versioned_and_slotted():
    data = yaml.safe_load((REPO / "configs" / "labeler_prompt.yaml").read_text())
    assert isinstance(data["version"], int) and data["version"] >= 1
    tmpl = data["template"]
    assert "{rubric}" in tmpl and "{pairs}" in tmpl
    # instructs strict JSON with the four axes named
    for axis in ("moral", "adherence", "coherence", "prose"):
        assert axis in tmpl
    # literal JSON braces are escaped for str.format (no stray single braces
    # besides the two named slots) -> format with dummy values must not raise
    tmpl.format(rubric="R", pairs="P")
