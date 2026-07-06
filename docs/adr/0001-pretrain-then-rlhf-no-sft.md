# Pretrain from scratch, then RLHF directly — no pretrained base, no SFT stage

We train a ~13M-parameter GPT from random weights on the instruction-formatted fable corpus, then apply RLHF (reward model + PPO) directly to that Base Model. We rejected fine-tuning an existing pretrained model because "trained from the beginning" is the project's point, and we rejected a separate SFT stage because every pretraining example is already prompt→fable formatted — SFT exists to bridge raw-text pretraining to instruction-following, and our pretraining already lands there; a second pass over the same distribution would be more pretraining wearing a different name.

## Considered Options

- Fine-tune GPT-2/Qwen + RLHF — cheaper, better fluency, but abandons the from-scratch claim
- Three-stage InstructGPT pipeline (pretrain → SFT → RLHF) — canonical, but the SFT stage would duplicate pretraining here
- RLHF as the sole training method — unworkable: a random-weight policy generates gibberish, preferences over gibberish carry no learning signal

## Consequences

RLHF's KL anchor references the Base Model itself (there is no separate SFT checkpoint). The report gains a section: "why our pipeline has two stages, not three."
