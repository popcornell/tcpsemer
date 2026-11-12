"""The Rust kernels must reproduce MeetEval and SciPy exactly, ties included."""
import random

import pytest

_rust = pytest.importorskip("tcpsemer._rust")


def random_stream(rng, n, vocab, sort=True):
    words = [rng.randrange(vocab) for _ in range(n)]
    t, timing = rng.uniform(-3, 2), []
    for _ in range(n):
        if rng.random() < 0.3:
            st = t  # repeated start times
        else:
            st = t + rng.choice([0.0, 0.25, 0.5, rng.uniform(0, 2)])
        et = st + rng.choice([0.0, 0.5, rng.uniform(0, 3)])
        timing.append((st, et))
        t = st
    if not sort:
        rng.shuffle(timing)
    return words, timing


@pytest.mark.parametrize("seed", range(300))
def test_distance_and_alignment_match_meeteval(seed):
    from meeteval.wer.matching.cy_levenshtein import (
        time_constrained_levenshtein_distance,
        time_constrained_levenshtein_distance_with_alignment)
    rng = random.Random(seed)
    r, rt = random_stream(rng, rng.randrange(0, 40), rng.choice([2, 5, 50]), sort=seed % 7 != 0)
    h, ht = random_stream(rng, rng.randrange(0, 40), rng.choice([2, 5, 50]), sort=seed % 5 != 0)
    if rng.random() < 0.5:  # collar
        c = rng.choice([0.5, 5.0])
        ht = [(a - c, b + c) for a, b in ht]
    assert _rust.tc_distance(r, h, rt, ht) == time_constrained_levenshtein_distance(r, h, rt, ht)
    expected = time_constrained_levenshtein_distance_with_alignment(r, h, rt, ht)["alignment"]
    assert _rust.tc_align(r, h, rt, ht) == [tuple(x) for x in expected]


@pytest.mark.parametrize("seed", range(300))
def test_assignment_matches_scipy(seed):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    rng = random.Random(seed)
    n = rng.randrange(1, 12)
    hi = rng.choice([1, 2, 3, 100])  # small ranges give many ties
    cost = [float(rng.randrange(hi + 1)) for _ in range(n * n)]
    _, cols = linear_sum_assignment(np.array(cost).reshape(n, n))
    assert _rust.linear_sum_assignment(n, cost) == list(cols)
