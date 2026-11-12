//! Rust core of tcpSemER: speaker assignment, time-constrained alignment and
//! utterance grouping for many sessions in parallel. Text loading,
//! normalisation and sentence embeddings stay in Python.

pub mod levenshtein;
pub mod lsap;
pub mod session;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use session::{plan_session, Plan, Seg};

type PySeg = (String, f64, f64, String);
type PyUtt = (String, String, String, usize, usize, f64, f64);
type PyRun = (String, String, usize, f64, f64);
type PyPlan = (u64, u64, Vec<PyUtt>, Vec<PyRun>, Vec<(String, u64)>);

fn to_segs(v: Vec<PySeg>) -> Vec<Seg> {
    v.into_iter()
        .map(|(speaker, start, end, words)| Seg { speaker, start, end, words })
        .collect()
}

fn to_py(p: Plan) -> PyPlan {
    (
        p.length,
        p.fixed_errors,
        p.utts
            .into_iter()
            .map(|u| (u.speaker, u.reference, u.hypothesis, u.n_ref, u.n_hyp, u.start, u.end))
            .collect(),
        p.unattached
            .into_iter()
            .map(|r| (r.speaker, r.hypothesis, r.n, r.start, r.end))
            .collect(),
        p.missed,
    )
}

/// plan_sessions(sessions, collar, max_gap) -> list of plans
///
/// `sessions` is a list of `(reference, hypothesis)`, each a list of
/// `(speaker, start_time, end_time, words)`. Each plan is
/// `(length, fixed_errors, utterances, unattached_insertions, missed_speakers)`.
#[pyfunction]
fn plan_sessions(
    py: Python<'_>,
    sessions: Vec<(Vec<PySeg>, Vec<PySeg>)>,
    collar: f64,
    max_gap: f64,
) -> PyResult<Vec<PyPlan>> {
    let sessions: Vec<(Vec<Seg>, Vec<Seg>)> =
        sessions.into_iter().map(|(r, h)| (to_segs(r), to_segs(h))).collect();
    let plans: Result<Vec<Plan>, String> = py.allow_threads(|| {
        sessions
            .par_iter()
            .map(|(r, h)| plan_session(r, h, collar, max_gap))
            .collect()
    });
    plans.map(|v| v.into_iter().map(to_py).collect()).map_err(PyValueError::new_err)
}

/// Exposed for testing against scipy.optimize.linear_sum_assignment.
#[pyfunction]
fn linear_sum_assignment(n: usize, cost: Vec<f64>) -> PyResult<Vec<usize>> {
    lsap::linear_sum_assignment(n, &cost).map_err(PyValueError::new_err)
}

/// Exposed for testing against meeteval's time_constrained_levenshtein_distance.
#[pyfunction]
fn tc_distance(r: Vec<u32>, h: Vec<u32>, rt: Vec<(f64, f64)>, ht: Vec<(f64, f64)>) -> u32 {
    levenshtein::distance(&r, &h, &rt, &ht)
}

/// Exposed for testing against meeteval's alignment.
#[pyfunction]
fn tc_align(
    r: Vec<u32>,
    h: Vec<u32>,
    rt: Vec<(f64, f64)>,
    ht: Vec<(f64, f64)>,
) -> PyResult<Vec<(Option<usize>, Option<usize>)>> {
    let a = levenshtein::align(&r, &h, &rt, &ht).map_err(PyValueError::new_err)?;
    let f = |x: usize| if x == levenshtein::EPS { None } else { Some(x) };
    Ok(a.into_iter().map(|(x, y)| (f(x), f(y))).collect())
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(plan_sessions, m)?)?;
    m.add_function(wrap_pyfunction!(linear_sum_assignment, m)?)?;
    m.add_function(wrap_pyfunction!(tc_distance, m)?)?;
    m.add_function(wrap_pyfunction!(tc_align, m)?)?;
    Ok(())
}
