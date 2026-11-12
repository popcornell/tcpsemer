import pytest

from tcpsemer import combine, tcpsemer


def seg(sess, spk, st, et, words):
    return {"session_id": sess, "speaker": spk, "start_time": st,
            "end_time": et, "words": words}


REF = [
    seg("S1", "A", 0.0, 2.0, "hello how are you doing today"),
    seg("S1", "B", 2.5, 4.0, "i am fine thank you"),
    seg("S1", "A", 8.0, 10.0, "shall we start the meeting now"),
]


class FakeModel:
    """Deterministic bag-of-characters embedder, so tests need no download."""

    def encode(self, sentences, **kw):
        import torch
        out = torch.zeros(len(sentences), 27)
        for i, s in enumerate(sentences):
            for c in s:
                out[i, (ord(c) - 97) if c.isalpha() else 26] += 1
        return out


def score(ref, hyp, **kw):
    return tcpsemer(ref, hyp, model=FakeModel(), **kw)


def test_perfect_hypothesis_with_renamed_speakers():
    hyp = [{**s, "speaker": {"A": "x", "B": "y"}[s["speaker"]]} for s in REF]
    r = score(REF, hyp)["S1"]
    assert r.length == 17
    assert r.errors == pytest.approx(0.0, abs=1e-5)


def test_missed_speaker_costs_all_its_words():
    hyp = [s for s in REF if s["speaker"] == "A"]
    r = score(REF, hyp)["S1"]
    assert r.errors == pytest.approx(5.0, abs=1e-5)


def test_extra_speaker_costs_all_its_words():
    hyp = REF + [seg("S1", "C", 5.0, 6.0, "one two three")]
    r = score(REF, hyp)["S1"]
    assert r.errors == pytest.approx(3.0, abs=1e-5)


def test_hypothesis_outside_collar_is_deletion_plus_insertion():
    # B's words shifted by 100 s: B's reference is deleted, the shifted words
    # are insertions, as in tcpWER.
    hyp = [REF[0], REF[2], seg("S1", "B", 102.5, 104.0, "i am fine thank you")]
    r = score(REF, hyp, collar=5, return_segments=True)["S1"]
    assert r.errors == pytest.approx(10.0, abs=1e-5)
    assert sum(s.errors for s in r.segments) == pytest.approx(r.errors)


def test_insertion_within_collar_joins_utterance():
    # "really" is inserted 3 s after the utterance ends: within the 5 s collar
    # it is scored semantically, outside a 1 s collar it costs 1.
    hyp = REF + [seg("S1", "B", 7.0, 7.0, "really")]
    near = score(REF, hyp, collar=5, return_segments=True)["S1"]
    far = score(REF, hyp, collar=1, return_segments=True)["S1"]
    assert 0 < near.errors < 1
    assert far.errors == pytest.approx(1.0, abs=1e-5)
    b = [s for s in near.segments if s.speaker == "B"]
    assert len(b) == 1 and b[0].hypothesis == "i am fine thank you really"


def test_missing_session_is_all_deletions():
    ref = REF + [seg("S2", "A", 0.0, 1.0, "a b c")]
    res = score(ref, REF)
    assert res["S2"].errors == 3 and res["S2"].length == 3
    assert combine(res).length == 20


def test_segments_sum_to_errors():
    hyp = [REF[0], seg("S1", "B", 2.5, 4.0, "i am fine thanks")]
    r = score(REF, hyp, return_segments=True)["S1"]
    assert sum(s.errors for s in r.segments) == pytest.approx(r.errors)
    assert sum(s.reference_words for s in r.segments) == r.length


def test_gap_splits_utterances():
    hyp = [{**s, "speaker": "x" if s["speaker"] == "A" else "y"} for s in REF]
    r = score(REF, hyp, return_segments=True)["S1"]
    assert len([s for s in r.segments if s.speaker == "A"]) == 2
    r = score(REF, hyp, return_segments=True, max_gap=100)["S1"]
    assert len([s for s in r.segments if s.speaker == "A"]) == 1
