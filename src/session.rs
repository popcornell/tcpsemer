//! Per-session tcpSemER plan: speaker assignment, alignment and utterance
//! grouping. Mirrors `tcpsemer/core.py` (`_plan_session_python`) and the
//! MeetEval preprocessing it relies on.

use std::collections::HashMap;

use crate::levenshtein::{self, Timing, EPS};
use crate::lsap;

pub struct Seg {
    pub speaker: String,
    pub start: f64,
    pub end: f64,
    pub words: String,
}

pub struct Utt {
    pub speaker: String,
    pub reference: String,
    pub hypothesis: String,
    pub n_ref: usize,
    pub n_hyp: usize,
    pub start: f64,
    pub end: f64,
}

pub struct Run {
    pub speaker: String,
    pub hypothesis: String,
    pub n: usize,
    pub start: f64,
    pub end: f64,
}

pub struct Plan {
    pub length: u64,
    pub fixed_errors: u64,
    pub utts: Vec<Utt>,
    pub unattached: Vec<Run>,
    pub missed: Vec<(String, u64)>,
}

/// Python's `str.isspace` set (Rust's `char::is_whitespace` lacks U+001C..U+001F).
#[inline]
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.split()` without arguments.
fn py_split(s: &str) -> Vec<&str> {
    s.split(is_py_space).filter(|w| !w.is_empty()).collect()
}

fn n_words(segs: &[&Seg]) -> u64 {
    segs.iter().map(|s| py_split(&s.words).len() as u64).sum()
}

/// meeteval `character_based`
fn character_based(s: f64, e: f64, words: &[&str]) -> Vec<Timing> {
    if words.len() == 1 {
        return vec![(s, e)];
    }
    let lens: Vec<i64> = words.iter().map(|w| w.chars().count() as i64).collect();
    let total: i64 = lens.iter().sum();
    let cl = (e - s) / (total as f64);
    let mut out = Vec::with_capacity(words.len());
    let mut start = 0i64;
    for l in lens {
        let end = start + l;
        out.push((s + cl * (start as f64), s + cl * (end as f64)));
        start = end;
    }
    out
}

struct Word<'a> {
    text: &'a str,
    sym: u32,
    start: f64, // pseudo word timing, without collar
    end: f64,
    t: Timing, // timing used in the DP (with collar for the hypothesis)
}

struct Interner<'a>(HashMap<&'a str, u32>);

impl<'a> Interner<'a> {
    fn get(&mut self, w: &'a str) -> u32 {
        let n = self.0.len() as u32;
        *self.0.entry(w).or_insert(n)
    }
}

fn check_valid(segs: &[&Seg], name: &str) -> Result<(), String> {
    for s in segs {
        if s.end < s.start {
            return Err(format!(
                "The end time of an interval must be larger than the start time. \
                 Found start_time={}, end_time={}, speaker={:?} in {}",
                s.start, s.end, s.speaker, name
            ));
        }
    }
    Ok(())
}

fn stable_sorted<'a>(segs: &[&'a Seg]) -> Vec<&'a Seg> {
    let mut v = segs.to_vec();
    v.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Words of already sorted segments, empty words removed.
fn words_of<'a>(
    segs: &[&'a Seg],
    points: bool,
    collar: f64,
    interner: &mut Interner<'a>,
) -> Vec<Word<'a>> {
    let mut out = Vec::new();
    for &s in segs {
        let ws = py_split(&s.words);
        if ws.is_empty() {
            continue; // the dummy '' word is removed again later
        }
        for (w, (st, et)) in ws.iter().zip(character_based(s.start, s.end, &ws)) {
            let (st, et) = if points {
                let p = (et + st) / 2.0;
                (p, p)
            } else {
                (st, et)
            };
            out.push(Word { text: w, sym: interner.get(w), start: st, end: et, t: (st - collar, et + collar) });
        }
    }
    out
}

/// Group by speaker in first-appearance order.
fn group<'a, K: Eq + std::hash::Hash + Clone>(
    segs: impl Iterator<Item = (K, &'a Seg)>,
) -> Vec<(K, Vec<&'a Seg>)> {
    let mut idx: HashMap<K, usize> = HashMap::new();
    let mut out: Vec<(K, Vec<&'a Seg>)> = Vec::new();
    for (k, s) in segs {
        match idx.get(&k) {
            Some(&i) => out[i].1.push(s),
            None => {
                idx.insert(k.clone(), out.len());
                out.push((k, vec![s]));
            }
        }
    }
    out
}

fn syms(w: &[Word]) -> Vec<u32> {
    w.iter().map(|x| x.sym).collect()
}
fn timings(w: &[Word]) -> Vec<Timing> {
    w.iter().map(|x| x.t).collect()
}

/// tcpWER speaker assignment: hypothesis speaker -> reference speaker (None
/// if the hypothesis speaker is assigned to a padding row).
fn assignment<'a>(
    r: &[&'a Seg],
    h: &[&'a Seg],
    collar: f64,
    interner: &mut Interner<'a>,
) -> Result<HashMap<&'a str, Option<&'a str>>, String> {
    check_valid(r, "reference")?;
    check_valid(h, "hypothesis")?;
    let r = stable_sorted(r);
    let h = stable_sorted(h);
    let rg = group(r.iter().map(|s| (s.speaker.as_str(), *s)));
    let hg = group(h.iter().map(|s| (s.speaker.as_str(), *s)));
    let (nr, nh) = (rg.len(), hg.len());
    if nr.max(nh) > 20 {
        return Err(format!(
            "Found a total of {} speakers in the input (Reference: {}, Hypothesis: {}). \
             MeetEval refuses to score more than 20 speakers per session.",
            nr.max(nh), nr, nh
        ));
    }
    let hyp_collar = if collar > 0.0 { collar } else { 0.0 };
    let rw: Vec<Vec<Word>> = rg.iter().map(|(_, s)| words_of(s, false, 0.0, interner)).collect();
    let hw: Vec<Vec<Word>> = hg.iter().map(|(_, s)| words_of(s, true, hyp_collar, interner)).collect();
    let rs: Vec<(Vec<u32>, Vec<Timing>)> = rw.iter().map(|w| (syms(w), timings(w))).collect();
    let hs: Vec<(Vec<u32>, Vec<Timing>)> = hw.iter().map(|w| (syms(w), timings(w))).collect();
    let empty: (Vec<u32>, Vec<Timing>) = (vec![], vec![]);
    let n = nr.max(nh);
    let mut cost = vec![0.0f64; n * n];
    for i in 0..n {
        let a = rs.get(i).unwrap_or(&empty);
        for j in 0..n {
            let b = hs.get(j).unwrap_or(&empty);
            cost[i * n + j] = levenshtein::distance(&a.0, &b.0, &a.1, &b.1) as f64;
        }
    }
    let col4row = lsap::linear_sum_assignment(n, &cost)?;
    let mut out = HashMap::new();
    for (ri, &c) in col4row.iter().enumerate() {
        if c < nh {
            out.insert(hg[c].0, if ri < nr { Some(rg[ri].0) } else { None });
        }
    }
    Ok(out)
}

fn join(words: &[&Word]) -> String {
    let mut s = String::new();
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(w.text);
    }
    s
}

/// Align one reference speaker with its hypothesis stream and group the
/// alignment into utterances (see `_split_utterances` in core.py).
fn utterances<'a>(
    speaker: &str,
    r: &[&'a Seg],
    h: &[&'a Seg],
    collar: f64,
    max_gap: f64,
    interner: &mut Interner<'a>,
) -> Result<(Vec<Utt>, Vec<Run>), String> {
    check_valid(r, "reference")?;
    check_valid(h, "hypothesis")?;
    let rw = words_of(&stable_sorted(r), false, 0.0, interner);
    let hw = words_of(&stable_sorted(h), true, collar, interner);
    let alignment = levenshtein::align(&syms(&rw), &syms(&hw), &timings(&rw), &timings(&hw))?;

    // (alignment index, ref word, hyp word)
    let mut utts: Vec<Vec<(usize, Option<&Word>, Option<&Word>)>> = Vec::new();
    let mut stack: Vec<(usize, Option<&Word>, Option<&Word>)> = Vec::new();
    let mut insertions: Vec<(usize, &Word)> = Vec::new();
    let mut prev_end: Option<f64> = None;
    for (i, &(a, b)) in alignment.iter().enumerate() {
        let hw_ = if b == EPS { None } else { Some(&hw[b]) };
        if a == EPS {
            insertions.push((i, hw_.unwrap()));
            continue;
        }
        let rw_ = &rw[a];
        let end = rw_.end;
        if let Some(p) = prev_end {
            if end - p > max_gap && !stack.is_empty() {
                utts.push(std::mem::take(&mut stack));
            }
        }
        prev_end = Some(end);
        stack.push((i, Some(rw_), hw_));
    }
    if !stack.is_empty() {
        utts.push(stack);
    }

    let spans: Vec<(f64, f64)> = utts
        .iter()
        .map(|u| {
            let st = u.iter().map(|x| x.1.unwrap().start).fold(f64::INFINITY, f64::min);
            let et = u.iter().map(|x| x.1.unwrap().end).fold(f64::NEG_INFINITY, f64::max);
            (st, et)
        })
        .collect();
    let mut unattached: Vec<Vec<(usize, &Word)>> = Vec::new();
    for (i, w) in insertions {
        let t = 0.5 * (w.start + w.end);
        let mut best: Option<usize> = None;
        let mut best_d = 0.0f64;
        for (k, &(st, et)) in spans.iter().enumerate() {
            if st - collar <= t && t <= et + collar {
                // Python: max(st - t, 0.0, t - et)
                let mut d = st - t;
                if 0.0 > d {
                    d = 0.0;
                }
                if t - et > d {
                    d = t - et;
                }
                if best.is_none() || d < best_d {
                    best = Some(k);
                    best_d = d;
                }
            }
        }
        match best {
            None => match unattached.last_mut() {
                Some(run) if run.last().unwrap().0 + 1 == i => run.push((i, w)),
                _ => unattached.push(vec![(i, w)]),
            },
            Some(k) => utts[k].push((i, None, Some(w))),
        }
    }

    let mut out = Vec::with_capacity(utts.len());
    for mut u in utts {
        u.sort_by_key(|x| x.0);
        let r_words: Vec<&Word> = u.iter().filter_map(|x| x.1).collect();
        let h_words: Vec<&Word> = u.iter().filter_map(|x| x.2).collect();
        let st = r_words.iter().map(|w| w.start).fold(f64::INFINITY, f64::min);
        let et = r_words.iter().map(|w| w.end).fold(f64::NEG_INFINITY, f64::max);
        out.push(Utt {
            speaker: speaker.to_string(),
            reference: join(&r_words),
            hypothesis: join(&h_words),
            n_ref: r_words.len(),
            n_hyp: h_words.len(),
            start: st,
            end: et,
        });
    }
    let runs = unattached
        .into_iter()
        .map(|run| {
            let ws: Vec<&Word> = run.iter().map(|x| x.1).collect();
            Run {
                speaker: speaker.to_string(),
                hypothesis: join(&ws),
                n: ws.len(),
                start: ws.iter().map(|w| w.start).fold(f64::INFINITY, f64::min),
                end: ws.iter().map(|w| w.end).fold(f64::NEG_INFINITY, f64::max),
            }
        })
        .collect();
    Ok((out, runs))
}

pub fn plan_session(r: &[Seg], h: &[Seg], collar: f64, max_gap: f64) -> Result<Plan, String> {
    let rv: Vec<&Seg> = r.iter().collect();
    let hv: Vec<&Seg> = h.iter().collect();
    let ref_groups = group(rv.iter().map(|s| (s.speaker.as_str(), *s)));
    let length: u64 = ref_groups.iter().map(|(_, s)| n_words(s)).sum();
    if hv.is_empty() {
        return Ok(Plan {
            length,
            fixed_errors: length,
            utts: vec![],
            unattached: vec![],
            missed: ref_groups.iter().map(|(k, s)| (k.to_string(), n_words(s))).collect(),
        });
    }
    let mut interner = Interner(HashMap::new());
    let assign = assignment(&rv, &hv, collar, &mut interner)?;
    // Hypothesis speakers relabelled to their reference speaker; None for
    // unassigned ones (Python: `assign.get(spk) or _UNASSIGNED`).
    let hyp_groups = group(hv.iter().map(|s| {
        let k = assign.get(s.speaker.as_str()).copied().flatten().filter(|k| !k.is_empty());
        (k, *s)
    }));
    let ref_names: std::collections::HashSet<&str> = ref_groups.iter().map(|(k, _)| *k).collect();
    let hyp_by_ref: HashMap<&str, &Vec<&Seg>> =
        hyp_groups.iter().filter_map(|(k, s)| k.map(|k| (k, s))).collect();

    let mut fixed: u64 = 0;
    for (k, segs) in &hyp_groups {
        if k.map_or(true, |k| !ref_names.contains(k)) {
            fixed += n_words(segs);
        }
    }
    let mut utts = Vec::new();
    let mut unattached = Vec::new();
    let mut missed = Vec::new();
    for (spk, segs) in &ref_groups {
        match hyp_by_ref.get(spk) {
            None => {
                let n = n_words(segs);
                fixed += n;
                missed.push((spk.to_string(), n));
            }
            Some(hsegs) => {
                let (u, runs) = utterances(spk, segs, hsegs, collar, max_gap, &mut interner)?;
                fixed += runs.iter().map(|r| r.n as u64).sum::<u64>();
                utts.extend(u);
                unattached.extend(runs);
            }
        }
    }
    Ok(Plan { length, fixed_errors: fixed, utts, unattached, missed })
}
