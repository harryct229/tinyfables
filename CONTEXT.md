# Fable Generation

A from-scratch language-model system that writes short English moral fables on demand, trained on the klusai/ds-tf1-en-3m dataset and aligned with human feedback. Built as a university systems-design project: every component exists to be explainable.

## Language

### The domain

**Fable**:
A short (~250-word) story that weaves a main character, setting, and challenge into an outcome that delivers an explicit Moral.
_Avoid_: story, tale (a Fable always carries a Moral; generic stories don't)

**Moral**:
The explicit lesson a Fable teaches, stated in the text (e.g. "timely help earns lasting loyalty").
_Avoid_: teaching (the dataset's field name), lesson, message

**Element**:
One of the five ingredients of a Fable: main character, setting, challenge, outcome, Moral.

**FableSpec**:
The structured request a user makes: some or all Elements plus an Age Group. What the model must honor when it writes.
_Avoid_: prompt (ambiguous — see Canonical Prompt), input, request

**Age Group**:
The dataset's A–E audience bands (3-and-under through 16+) that set vocabulary and complexity.

### The interface

**Canonical Prompt**:
The dataset's single fixed instruction template ("Create a fable based on the following elements…") that renders a FableSpec as natural language.

**Paraphrased Prompt**:
An alternative natural-language rendering of the same FableSpec, produced by the augmentation stage. Element values are preserved verbatim; only the surrounding phrasing varies.

### The models

**Base Model**:
The transformer pretrained from random weights on fable data. Fluent but not yet aligned.

**Aligned Model**:
The Base Model after the feedback stage. The system's final product.

### The feedback

**Axis**:
One of the four dimensions a fable is judged on: Moral delivery, spec adherence, coherence, prose & age-fit.

**Rubric**:
The anchored definitions of each Axis and each 1–5 scale point, written before any labeling happens.

**Preference Pair**:
Two fables generated from the same FableSpec, each rated on all four Axes; the preferred one is derived from Aggregate Scores.
_Avoid_: comparison, sample pair

**Aggregate Score**:
The fixed weighted combination of a fable's four Axis ratings (Moral delivery weighted highest) that derives preferences.

**Calibration Set**:
A fixed set of pairs re-rated repeatedly during the labeling run to measure the AI Labeler's self-consistency.

**AI Labeler**:
Claude, pinned to one model version and one prompt, rating fables against the Rubric. The system's preference source.
_Avoid_: annotator, judge (the Judge is the separate eval-time model, a different family)

**Gold Set**:
A small human-labeled pair set, held ready as a fallback validator of the AI Labeler if literal human feedback is required.
