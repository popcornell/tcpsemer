"""Time-constrained minimum-permutation semantic error rate (tcpSemER).

The computation per session is:

1. Find the speaker assignment that minimises tcpWER (MeetEval).
2. For every reference speaker, align its words with the words of the
   assigned hypothesis speaker using the time-constrained Levenshtein
   alignment of tcpWER.
3. Split the alignment into utterances wherever two consecutive reference
   words are more than ``max_gap`` seconds apart (end time to end time).
   Matched and substituted hypothesis words go with their reference word.
4. Each inserted hypothesis word (aligned to no reference word) joins the
   utterance whose span, widened by the collar on both sides, contains the
   word's time; the nearest one if there are several. An inserted word that
   falls in no widened span costs 1.
5. Each utterance costs ``(1 - cos(e_ref, e_hyp)) * n_ref`` where ``e`` are
   sentence embeddings and ``n_ref`` is the number of reference words. An
   utterance with no hypothesis words costs ``n_ref`` (all deletions).
6. Missed reference speakers cost all their words; hypothesis speakers that
   are not assigned to any reference speaker cost all their words.
7. The session error rate is the total cost divided by the number of
   reference words.

The corpus-level score is the sum of errors over the sum of reference words
(see :func:`combine`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from meeteval.io.seglst import SegLST
from meeteval.wer import time_constrained_minimum_permutation_word_error_rate
from meeteval.wer.wer.time_constrained import align as tc_align

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Speaker key for hypothesis streams that have no reference counterpart.
_UNASSIGNED = "__tcpsemer_unassigned__"


@dataclass
class SegmentScore:
    """One aligned utterance of a reference speaker."""
    speaker: str
    start_time: float
    end_time: float
    reference: str
    hypothesis: str
    reference_words: int
    distance: float  # 1 - cosine similarity, or 1.0 if there is no hypothesis
    errors: float    # distance * reference_words (or hypothesis words for pure insertions)


@dataclass
class SemERResult:
    errors: float
    length: int
    segments: Optional[List[SegmentScore]] = field(default=None, repr=False)

    @property
    def error_rate(self) -> float:
        return self.errors / self.length if self.length else float("nan")

    # Backward compatibility with compute_tcpsemdist.SemDistResult
    @property
    def references(self) -> int:
        return self.length


def combine(results: Dict[str, SemERResult]) -> SemERResult:
    """Micro average over sessions: sum of errors / sum of reference words."""
    return SemERResult(
        errors=sum(r.errors for r in results.values()),
        length=sum(r.length for r in results.values()),
    )


def _n_words(segments) -> int:
    return sum(len(s["words"].split()) for s in segments)


def _to_float(sl: SegLST) -> SegLST:
    # meeteval loads timestamps as Decimal; the time-constrained code mixes
    # them with float collars, which raises TypeError.
    return sl.map(lambda s: {**s, "start_time": float(s["start_time"]),
                             "end_time": float(s["end_time"])})


def _load(reference, hypothesis, regex, uem, normalizer):
    from meeteval.wer.api import _load_texts
    reference, hypothesis = _load_texts(
        reference, hypothesis, regex=regex, uem=uem, normalizer=normalizer)
    return _to_float(reference), _to_float(hypothesis)


def _word_time(w) -> float:
    return 0.5 * (float(w["start_time"]) + float(w["end_time"]))


def _split_utterances(alignment, max_gap: float, collar: float):
    """Group an alignment (list of (ref_word, hyp_word) seglst pairs) into
    utterances.

    Utterances are split at gaps > max_gap between consecutive reference
    words. Inserted hypothesis words are attached by time: to the utterance
    whose reference span, widened by ``collar``, contains them (the nearest
    if several). Returns ``(utterances, unattached_insertions)``; each
    utterance is a list of ``(ref_word, hyp_word)`` pairs in alignment order,
    and ``unattached_insertions`` is a list of runs of hypothesis words.
    """
    utts, stack, prev_end = [], [], None
    insertions = []  # (alignment index, hyp word)
    for i, (ref, hyp) in enumerate(alignment):
        if ref is None:
            insertions.append((i, hyp))
            continue
        end = float(ref["end_time"])
        if prev_end is not None and end - prev_end > max_gap and stack:
            utts.append(stack)
            stack = []
        prev_end = end
        stack.append((i, ref, hyp))
    if stack:
        utts.append(stack)

    spans = [(min(float(r["start_time"]) for _, r, _ in u),
              max(float(r["end_time"]) for _, r, _ in u)) for u in utts]
    unattached = []
    for i, hyp in insertions:
        t = _word_time(hyp)
        best, best_d = None, None
        for k, (st, et) in enumerate(spans):
            if st - collar <= t <= et + collar:
                d = max(st - t, 0.0, t - et)
                if best is None or d < best_d:
                    best, best_d = k, d
        if best is None:
            if unattached and unattached[-1][-1][0] == i - 1:
                unattached[-1].append((i, hyp))
            else:
                unattached.append([(i, hyp)])
        else:
            utts[best].append((i, None, hyp))

    utts = [[(r, h) for _, r, h in sorted(u, key=lambda x: x[0])] for u in utts]
    unattached = [[h for _, h in run] for run in unattached]
    return utts, unattached


class _Embedder:
    """Sentence embedder with a cache, so repeated utterances ("yeah", "okay")
    are encoded once and all distinct sentences are encoded in batches."""

    def __init__(self, model, device=None, batch_size=64):
        if isinstance(model, str):
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model, device=device)
        self.model = model
        self.batch_size = batch_size
        self.cache = {}

    def add(self, sentences):
        new = list(dict.fromkeys(s for s in sentences if s not in self.cache))
        if new:
            emb = self.model.encode(new, batch_size=self.batch_size,
                                    convert_to_tensor=True,
                                    show_progress_bar=False)
            self.cache.update(zip(new, emb))

    def distance(self, a: str, b: str) -> float:
        import torch
        sim = torch.nn.functional.cosine_similarity(
            self.cache[a].unsqueeze(0), self.cache[b].unsqueeze(0)).item()
        return 1.0 - sim


def tcpsemer(
        reference,
        hypothesis,
        *,
        collar: float = 5.0,
        regex: Optional[str] = None,
        uem=None,
        normalizer: Optional[str] = None,
        model: Union[str, object] = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_gap: float = 2.0,
        return_segments: bool = False,
) -> Dict[str, SemERResult]:
    """Compute tcpSemER per session.

    Args:
        reference, hypothesis: anything ``meeteval`` can load (SegLST/STM/CTM
            file paths, a list of segment dicts, a ``SegLST``).
        collar: collar in seconds for the time-constrained alignment.
        regex: only keep sessions whose id matches this regex.
        uem: UEM file or object, as in ``meeteval-wer``.
        normalizer: text normaliser name, as in ``meeteval-wer``
            (e.g. ``"chime8"``, ``"lower,rm(.?!,)"``).
        model: sentence-transformers model name or an already loaded
            ``SentenceTransformer``.
        device: device for the embedding model.
        batch_size: encoding batch size.
        max_gap: a reference gap longer than this (seconds) starts a new
            utterance.
        return_segments: store per-utterance scores in ``result.segments``.

    Returns:
        ``{session_id: SemERResult}``. Use :func:`combine` for the overall
        score.
    """
    reference, hypothesis = _load(reference, hypothesis, regex, uem, normalizer)
    ref_by_sess = reference.groupby("session_id")
    hyp_by_sess = hypothesis.groupby("session_id")

    extra = set(hyp_by_sess) - set(ref_by_sess)
    if extra:
        logger.warning("Ignoring %d hypothesis session(s) without reference: %s",
                       len(extra), sorted(extra))

    # Pass 1: alignments. Pass 2 (below): embeddings in one batched call.
    plans = {}
    for sess, c_ref in ref_by_sess.items():
        c_hyp = hyp_by_sess.get(sess, SegLST([]))
        ref_spk = c_ref.groupby("speaker")
        length = sum(_n_words(v) for v in ref_spk.values())
        fixed_errors = 0.0
        utts = []  # (speaker, [(ref_word, hyp_word), ...])
        inserted = []  # (speaker, [hyp_word, ...]) outside every utterance

        if len(c_hyp) == 0:
            plans[sess] = (length, float(length), [], [
                (spk, _n_words(segs)) for spk, segs in ref_spk.items()], [])
            continue

        result = time_constrained_minimum_permutation_word_error_rate(
            c_ref, c_hyp, collar=collar)
        assign = {h: r for r, h in result.assignment if h is not None}
        c_hyp = SegLST([{**s, "speaker": assign.get(s["speaker"]) or _UNASSIGNED}
                        for s in c_hyp])
        hyp_spk = c_hyp.groupby("speaker")

        for spk, segs in hyp_spk.items():
            if spk not in ref_spk:
                fixed_errors += _n_words(segs)  # insertions

        missed = []
        for spk, segs in ref_spk.items():
            if spk not in hyp_spk:
                fixed_errors += _n_words(segs)  # deletions
                missed.append((spk, _n_words(segs)))
                continue
            alignment = tc_align(segs, hyp_spk[spk], collar=collar, style="seglst")
            spk_utts, unattached = _split_utterances(alignment, max_gap, collar)
            for u in spk_utts:
                utts.append((spk, u))
            for run in unattached:
                fixed_errors += len(run)  # insertions outside every utterance
                inserted.append((spk, run))
        plans[sess] = (length, fixed_errors, utts, missed, inserted)

    embedder = None
    pending = []
    for _, _, utts, _, _ in plans.values():
        for _, u in utts:
            r = [x["words"] for x, _ in u if x is not None]
            h = [y["words"] for _, y in u if y is not None]
            if r and h:
                pending += [" ".join(r), " ".join(h)]
    if pending:
        embedder = _Embedder(model, device=device, batch_size=batch_size)
        embedder.add(pending)

    out = {}
    for sess, (length, errors, utts, missed, inserted) in plans.items():
        segments = [] if return_segments else None
        for spk, u in utts:
            r_words = [x for x, _ in u if x is not None]
            h_words = [y for _, y in u if y is not None]
            r = " ".join(x["words"] for x in r_words)
            h = " ".join(y["words"] for y in h_words)
            if r_words and not h_words:
                dist, cost = 1.0, float(len(r_words))
            elif h_words and not r_words:
                dist, cost = 1.0, float(len(h_words))
            else:
                dist = embedder.distance(r, h)
                cost = dist * len(r_words)
            errors += cost
            if return_segments:
                times = [float(w["start_time"]) for w in r_words or h_words] + \
                        [float(w["end_time"]) for w in r_words or h_words]
                segments.append(SegmentScore(
                    speaker=spk, start_time=min(times), end_time=max(times),
                    reference=r, hypothesis=h, reference_words=len(r_words),
                    distance=dist, errors=cost))
        if return_segments:
            # Insertions outside every utterance (already counted in `errors`).
            for spk, run in inserted:
                segments.append(SegmentScore(
                    speaker=spk,
                    start_time=min(float(w["start_time"]) for w in run),
                    end_time=max(float(w["end_time"]) for w in run),
                    reference="", hypothesis=" ".join(w["words"] for w in run),
                    reference_words=0, distance=1.0, errors=float(len(run))))
            # Missed reference speakers (already counted in `errors`).
            for spk, n in missed:
                segments.append(SegmentScore(
                    speaker=spk, start_time=float("nan"), end_time=float("nan"),
                    reference="<missed speaker>", hypothesis="",
                    reference_words=n, distance=1.0, errors=float(n)))
        out[sess] = SemERResult(errors=errors, length=length, segments=segments)
    return out


def tcpSemDist(reference, hypothesis, collar=5, regex=None, uem=None,
               normalizer=None, **kwargs):
    """Alias with the signature of the original ``compute_tcpsemdist``."""
    return tcpsemer(reference, hypothesis, collar=collar, regex=regex,
                    uem=uem, normalizer=normalizer, **kwargs)
