//! Linear sum assignment for square cost matrices.
//!
//! Line-by-line port of `scipy/optimize/rectangular_lsap/rectangular_lsap.cpp`
//! (SciPy 1.15.3, shortest augmenting path, Crouse 2016), so ties are broken
//! exactly as `scipy.optimize.linear_sum_assignment` breaks them.

fn augmenting_path(
    nc: usize,
    cost: &[f64],
    u: &[f64],
    v: &[f64],
    path: &mut [isize],
    row4col: &[isize],
    spc: &mut [f64],
    mut i: usize,
    sr: &mut [bool],
    sc: &mut [bool],
    remaining: &mut [usize],
) -> Option<(usize, f64)> {
    let mut min_val = 0.0f64;
    let mut num_remaining = nc;
    for it in 0..nc {
        remaining[it] = nc - it - 1;
    }
    sr.iter_mut().for_each(|x| *x = false);
    sc.iter_mut().for_each(|x| *x = false);
    spc.iter_mut().for_each(|x| *x = f64::INFINITY);
    let mut sink: isize = -1;
    while sink == -1 {
        let mut index: isize = -1;
        let mut lowest = f64::INFINITY;
        sr[i] = true;
        for it in 0..num_remaining {
            let j = remaining[it];
            let r = min_val + cost[i * nc + j] - u[i] - v[j];
            if r < spc[j] {
                path[j] = i as isize;
                spc[j] = r;
            }
            if spc[j] < lowest || (spc[j] == lowest && row4col[j] == -1) {
                lowest = spc[j];
                index = it as isize;
            }
        }
        min_val = lowest;
        if min_val == f64::INFINITY {
            return None;
        }
        let index = index as usize;
        let j = remaining[index];
        if row4col[j] == -1 {
            sink = j as isize;
        } else {
            i = row4col[j] as usize;
        }
        sc[j] = true;
        num_remaining -= 1;
        remaining[index] = remaining[num_remaining];
    }
    Some((sink as usize, min_val))
}

/// Returns `col4row` for an `n x n` row-major cost matrix.
pub fn linear_sum_assignment(n: usize, cost: &[f64]) -> Result<Vec<usize>, String> {
    if n == 0 {
        return Ok(vec![]);
    }
    if cost.iter().any(|&c| c != c || c == f64::NEG_INFINITY) {
        return Err("matrix contains invalid numeric entries".into());
    }
    let mut u = vec![0.0f64; n];
    let mut v = vec![0.0f64; n];
    let mut spc = vec![0.0f64; n];
    let mut path = vec![-1isize; n];
    let mut col4row = vec![-1isize; n];
    let mut row4col = vec![-1isize; n];
    let mut sr = vec![false; n];
    let mut sc = vec![false; n];
    let mut remaining = vec![0usize; n];
    for cur_row in 0..n {
        let (sink, min_val) = augmenting_path(
            n, cost, &u, &v, &mut path, &row4col, &mut spc, cur_row, &mut sr, &mut sc,
            &mut remaining,
        )
        .ok_or_else(|| "cost matrix is infeasible".to_string())?;
        u[cur_row] += min_val;
        for i in 0..n {
            if sr[i] && i != cur_row {
                u[i] += min_val - spc[col4row[i] as usize];
            }
        }
        for j in 0..n {
            if sc[j] {
                v[j] -= min_val - spc[j];
            }
        }
        let mut j = sink as isize;
        loop {
            let i = path[j as usize];
            row4col[j as usize] = i;
            std::mem::swap(&mut col4row[i as usize], &mut j);
            if i as usize == cur_row {
                break;
            }
        }
    }
    Ok(col4row.into_iter().map(|c| c as usize).collect())
}
