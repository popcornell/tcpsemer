//! Time-constrained Levenshtein distance and alignment.
//!
//! Line-by-line ports of `meeteval/wer/matching/levenshtein.h` (MeetEval
//! 0.4.3) with unit costs (del = ins = sub = 1, cor = 0). Unsigned arithmetic
//! wraps as in C++, so results are identical, including tie-breaking in the
//! backtrace.

pub type Timing = (f64, f64);

/// Sentinel for "no word" in an alignment.
pub const EPS: usize = usize::MAX;

#[inline]
fn overlaps(a: Timing, b: Timing) -> bool {
    a.0 < b.1 && a.1 > b.0
}

#[inline]
fn cmax(a: f64, b: f64) -> f64 {
    // std::max(a, b) == (a < b) ? b : a
    if a < b {
        b
    } else {
        a
    }
}

/// `_timings_sorted` from cy_levenshtein.pyx. Note that `last_begin` starts
/// at 0, so a negative first begin time counts as unsorted.
pub fn timings_sorted(t: &[Timing]) -> bool {
    let mut last_begin = 0.0f64;
    for &(b, _) in t {
        if b < last_begin {
            return false;
        }
        last_begin = b;
    }
    true
}

/// `time_constrained_levenshtein_distance_unoptimized_`
fn distance_unoptimized(r: &[u32], h: &[u32], rt: &[Timing], ht: &[Timing]) -> u32 {
    let mut row: Vec<u32> = vec![0; h.len() + 1];
    for i in 1..row.len() {
        row[i] = row[i - 1].wrapping_add(1);
    }
    for ri in 0..r.len() {
        let rs = r[ri];
        let riv = rt[ri];
        let mut diagonal = row[0];
        row[0] = row[0].wrapping_add(1);
        for hi in 1..h.len() + 1 {
            let hs = h[hi - 1];
            let hiv = ht[hi - 1];
            let up = row[hi];
            let ins_or_del = row[hi - 1].wrapping_add(1).min(up.wrapping_add(1));
            if overlaps(riv, hiv) {
                row[hi] = ins_or_del.min(diagonal.wrapping_add(if hs == rs { 0 } else { 1 }));
            } else {
                row[hi] = ins_or_del;
            }
            diagonal = up;
        }
    }
    row[row.len() - 1]
}

/// `time_constrained_levenshtein_distance_v2_`
fn distance_v2(r: &[u32], h: &[u32], rt: &[Timing], ht: &[Timing]) -> u32 {
    const INSDEL: u32 = 2;
    let mut row: Vec<u32> = vec![u32::MAX; h.len() + 1];
    row[0] = 0;
    let mut ref_end_time = 0.0f64;
    let mut hyp_start = 0usize;
    let mut hi = 0usize;
    for ri in 0..r.len() {
        let rs = r[ri];
        let riv = rt[ri];
        ref_end_time = cmax(riv.1, ref_end_time);
        let mut diagonal = row[hyp_start];
        row[hyp_start] = row[hyp_start].wrapping_add(1);
        let mut allow_shift = true;
        hi = hyp_start;
        while hi < h.len() {
            let hs = h[hi];
            let hiv = ht[hi];
            if allow_shift {
                if riv.0 > hiv.1 {
                    hyp_start += 1;
                } else {
                    allow_shift = false;
                }
            }
            let up = row[hi + 1].min(diagonal.wrapping_add(1));
            let sub = if overlaps(riv, hiv) {
                if hs == rs { 0 } else { 1 }
            } else {
                INSDEL
            };
            row[hi + 1] = row[hi]
                .wrapping_add(1)
                .min(up.wrapping_add(1))
                .min(diagonal.wrapping_add(sub));
            diagonal = up;
            if ref_end_time < hiv.0 {
                break;
            }
            hi += 1;
        }
    }
    while hi < h.len() {
        row[hi + 1] = row[hi].wrapping_add(1);
        hi += 1;
    }
    row[h.len()]
}

/// `time_constrained_levenshtein_distance` (cy_levenshtein.pyx), prune="auto".
pub fn distance(r: &[u32], h: &[u32], rt: &[Timing], ht: &[Timing]) -> u32 {
    if r.is_empty() {
        return h.len() as u32;
    }
    if h.is_empty() {
        return r.len() as u32;
    }
    if timings_sorted(rt) && timings_sorted(ht) {
        distance_v2(r, h, rt, ht)
    } else {
        distance_unoptimized(r, h, rt, ht)
    }
}

/// `time_constrained_levenshtein_distance_with_alignment` (prune="auto").
/// Returns (ref index | EPS, hyp index | EPS) pairs in temporal order.
pub fn align(r: &[u32], h: &[u32], rt: &[Timing], ht: &[Timing]) -> Result<Vec<(usize, usize)>, String> {
    let prune = timings_sorted(rt) && timings_sorted(ht);
    if r.is_empty() {
        return Ok((0..h.len()).map(|i| (EPS, i)).collect());
    }
    if h.is_empty() {
        return Ok((0..r.len()).map(|i| (i, EPS)).collect());
    }
    let (nr, nh) = (r.len(), h.len());
    let max_value: u32 = u32::MAX - 1;
    if (max_value as u64) <= (nr + nh) as u64 {
        return Err("costs too large".into());
    }
    let w = nh + 1;
    let mut m: Vec<u32> = vec![max_value; (nr + 1) * w];
    m[0] = 0;
    for i in 1..w {
        m[i] = m[i - 1].wrapping_add(1);
    }
    let mut start = 1usize;
    let mut ref_end_time = rt[0].1;
    for ri in 0..nr {
        let rs = r[ri];
        let (prev, row) = m.split_at_mut((ri + 1) * w);
        let prev = &prev[ri * w..];
        let row = &mut row[..w];
        let riv = rt[ri];
        ref_end_time = cmax(riv.1, ref_end_time);
        if prune {
            while start < nh + 1 {
                if ht[start - 1].1 > riv.0 {
                    break;
                }
                start += 1;
            }
        }
        row[start - 1] = prev[start - 1].wrapping_add(1);
        let mut hi = start;
        while hi < nh + 1 {
            let hs = h[hi - 1];
            let hiv = ht[hi - 1];
            if prune && hiv.0 > ref_end_time {
                break;
            }
            let ins_or_del = row[hi - 1].wrapping_add(1).min(prev[hi].wrapping_add(1));
            if overlaps(riv, hiv) {
                row[hi] = ins_or_del.min(prev[hi - 1].wrapping_add(if hs == rs { 0 } else { 1 }));
            } else {
                row[hi] = ins_or_del;
            }
            hi += 1;
        }
        if prune {
            if ri < nr - 1 {
                while hi < nh + 1 {
                    row[hi] = row[hi - 1].wrapping_add(1);
                    if ht[hi - 1].0 > rt[ri + 1].1 {
                        break;
                    }
                    hi += 1;
                }
            } else {
                while hi < nh + 1 {
                    row[hi] = row[hi - 1].wrapping_add(1);
                    hi += 1;
                }
            }
        }
    }

    // Backtracking
    let mut hi = nh;
    let mut ri = nr;
    let mut out = Vec::with_capacity(nr + nh);
    while ri > 0 || hi > 0 {
        if ri == 0 {
            hi -= 1;
            out.push((EPS, hi));
        } else if hi == 0 {
            ri -= 1;
            out.push((ri, EPS));
        } else {
            let cost_ins = m[ri * w + hi - 1].wrapping_add(1);
            let cost_del = m[(ri - 1) * w + hi].wrapping_add(1);
            if overlaps(ht[hi - 1], rt[ri - 1]) {
                let c = m[(ri - 1) * w + hi - 1]
                    .wrapping_add(if r[ri - 1] == h[hi - 1] { 0 } else { 1 });
                if c < cost_del.min(cost_ins) {
                    ri -= 1;
                    hi -= 1;
                    out.push((ri, hi));
                    continue;
                }
            }
            if cost_del < cost_ins {
                ri -= 1;
                out.push((ri, EPS));
            } else {
                hi -= 1;
                out.push((EPS, hi));
            }
        }
    }
    out.reverse();
    Ok(out)
}
