# tcpSemER

Time-constrained minimum-permutation semantic error rate for multi-speaker ASR,
from [Tawara et al., Interspeech 2026 (arXiv:2603.22709)](https://arxiv.org/abs/2603.22709).

tcpSemER keeps the speaker assignment and time-constrained alignment of
tcpWER ([MeetEval](https://github.com/fgnt/meeteval)) and replaces the exact
word match with the cosine distance between sentence embeddings of the aligned
reference and hypothesis utterances. A substitution that keeps the meaning
costs less than one that changes it.

## Install

```bash
pip install git+https://github.com/<org>/tcpSemER
# or, from a clone
pip install -e .
```

## Usage

Inputs are anything `meeteval-wer` reads: SegLST JSON, STM, CTM. A SegLST file
is a list of segments:

```json
[{"session_id": "S1", "speaker": "A", "start_time": 0.0, "end_time": 2.0, "words": "hello how are you"}]
```

Command line (same flag names as `meeteval-wer`):

```bash
tcpsemer -r examples/ref.json -h examples/hyp.json --collar 5 -v
# S1	tcpSemER=7.96%	errors=1.35	length=17
# tcpSemER: 7.96% [errors=1.35, length=17, sessions=1, collar=5.0]
```

Useful options:

| option | meaning |
|---|---|
| `--collar` | collar in seconds (default 5) |
| `--normalizer` | text normaliser as in `meeteval-wer`, e.g. `chime8` |
| `--uem`, `--regex` | restrict the scored region / sessions |
| `--model` | sentence-transformers model (default `sentence-transformers/all-MiniLM-L6-v2`) |
| `--max-gap` | reference gap in seconds that starts a new utterance (default 2) |
| `--per-reco-out` | per-session results as JSON |
| `--segments-out` | per-utterance distances as JSON lines |

Python:

```python
from tcpsemer import tcpsemer, combine

res = tcpsemer("ref.json", "hyp.json", collar=5, normalizer="chime8")
print(combine(res).error_rate)             # sum of errors / sum of reference words
print({k: v.error_rate for k, v in res.items()})
```

`model=` also takes a loaded `SentenceTransformer`, so one model can be reused
across calls. `return_segments=True` stores the per-utterance scores in
`result.segments`.

## Definition

For each session:

1. Find the speaker assignment that minimises tcpWER at the given collar.
2. For each reference speaker, align its words with those of the assigned
   hypothesis speaker using the time-constrained Levenshtein alignment of
   tcpWER (character-based pseudo word timings for the reference, point timings
   for the hypothesis).
3. Split the alignment into utterances wherever two consecutive reference
   words are more than 2 s apart (end time to end time). Matched and
   substituted hypothesis words go with their reference word.
4. A hypothesis word aligned as an insertion joins the utterance whose
   reference span, widened by the collar on both sides, contains the word's
   time (the nearest one if there are several). An insertion that falls in no
   widened span costs 1, as in tcpWER.
5. An utterance with reference words $r$ and hypothesis words $h$ costs
   $(1 - \cos(e(r), e(h)))\,|r|$, where $e$ is the sentence embedding. An
   utterance with no hypothesis words costs $|r|$.
6. A reference speaker with no assigned hypothesis speaker costs all its words.
   A hypothesis speaker with no reference speaker costs all its words.
7. tcpSemER is the total cost divided by the number of reference words.

The corpus score is the sum of costs over the sum of reference words
(`combine`), the same micro average used for tcpWER.

A session that is missing from the hypothesis counts as all deletions.
Hypothesis sessions with no reference are ignored, with a warning.

## Speed

Each distinct sentence is encoded once, in batches. DiPCo eval (5 sessions,
about 29k reference words) takes about 5 s on a MacBook.

## Compatibility

`tcpSemDist(reference, hypothesis, collar=5, regex=None, uem=None,
normalizer=None)` is kept as an alias for code written against
`compute_tcpsemdist.py`. Results have `.errors`, `.length`, `.error_rate`, and
`.references` (equal to `.length`).

## Citation

```bibtex
@inproceedings{tawara2026who,
  author    = {Tawara, Naohiro and Cornell, Samuele and Polok, Alexander and Delcroix, Marc and Burget, Luk{\'a}{\v{s}} and Watanabe, Shinji},
  title     = {{Who Spoke What When? Evaluating Spoken Language Models for Conversational ASR with Semantic and Overlap-Aware Metrics}},
  booktitle = {Proc. Interspeech},
  year      = {2026},
  note      = {arXiv:2603.22709}
}
```

Please also cite MeetEval, which provides the alignment and speaker assignment.
