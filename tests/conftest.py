"""Shared test fixtures/helpers.

`tests` is an intentional package, so test modules must use package-relative
imports such as `from .conftest import fake_labeler_runner` for shared helpers.
"""


def fake_labeler_runner(prompt, model):
    import hashlib
    import json

    from tinyfables.feedback import AXES

    ids = [ln.split("pair_id: ")[1].strip() for ln in prompt.splitlines() if "pair_id: " in ln]

    def rate(seedtext):
        h = int.from_bytes(hashlib.sha256(seedtext.encode("utf-8")).digest()[:8], "big")
        return {a: 1 + (h >> (3 * j)) % 5 for j, a in enumerate(AXES)}

    return json.dumps(
        {
            "labels": [
                {"pair_id": pid, "fable_a": rate(pid + "a"), "fable_b": rate(pid + "b")}
                for pid in ids
            ]
        }
    )
