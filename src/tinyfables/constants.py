"""Vocabulary constants. Special tokens are reserved at tokenizer creation
(ADR-0002) and can never be added later — do not change these strings."""

EOT = "<|endoftext|>"
PAD = "<|pad|>"
SPECIALS: list[str] = [EOT, PAD]  # trained first, so ids are 0 and 1
