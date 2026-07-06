# Custom byte-level BPE tokenizer, vocab 8192, trained on the fable corpus

We train our own 8k-vocab BPE instead of reusing GPT-2's 50,257-token vocabulary. At d_model=384 with tied embeddings, GPT-2's vocab costs 19.3M embedding parameters — larger than the entire 10.6M transformer — while 8k costs 3.1M (~24% of the model), and a domain-trained vocab compresses fable text better, fitting more story per 1024-token window. Changing the tokenizer later means retraining everything, so this is locked before the first pretraining run; special tokens (end-of-text, pad) are reserved now because they cannot be added afterwards.

## Considered Options

- GPT-2 tokenizer — zero effort, but >65% of parameters would be embeddings, mostly dead vocabulary (code/web tokens fables never use)
- 4k vocab — halves embedding cost again but lengthens sequences ~15–25%; right only for a <5M model
