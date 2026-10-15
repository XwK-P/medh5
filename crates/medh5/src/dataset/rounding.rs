//! Integer allocations for `medh5.split/2`: how many groups each partition
//! takes in all, and how many from each stratum.
//!
//! Totals are the largest remainder of `N·r`; each stratum's counts are a
//! *controlled rounding* of `n_s·r_p`: every cell is the floor or the
//! ceiling, every row sums to its stratum and every column to its total.
//! Such a rounding always exists (it is an integer flow), and which cells
//! round up is decided by a fixed order, so the answer depends on nothing but
//! the counts and the shares.

use crate::{Error, Result};

/// Products closer than this to an integer are that integer: `0.7 * 10` is
/// `7.000000000000001` in floating point, and it is 7 groups.
const EPS: f64 = 1e-9;

/// `x` split into a whole part and a fractional part in `[0, 1)`, with a
/// product within [`EPS`] of an integer taken as that integer.
fn split_at_integer(x: f64) -> (u64, f64) {
    let nearest = x.round();
    if (x - nearest).abs() <= EPS {
        return (nearest.max(0.0) as u64, 0.0);
    }
    let floor = x.floor();
    (floor.max(0.0) as u64, x - floor)
}

/// `n` items shared as `shares` (summing to 1): the floor of each `n·r`,
/// and the remainder to the largest fractional parts, earlier entries first
/// on a tie.
pub(crate) fn largest_remainder(n: u64, shares: &[f64]) -> Vec<u64> {
    let parts: Vec<(u64, f64)> = shares.iter().map(|r| split_at_integer(n as f64 * r)).collect();
    let mut out: Vec<u64> = parts.iter().map(|(whole, _)| *whole).collect();
    let given: u64 = out.iter().sum();
    let mut order: Vec<usize> = (0..shares.len()).collect();
    order.sort_by(|a, b| parts[*b].1.total_cmp(&parts[*a].1).then(a.cmp(b)));
    for i in order.into_iter().take(n.saturating_sub(given) as usize) {
        out[i] += 1;
    }
    out
}

/// Each row's `rows[s]` items shared as `shares`, as integers: every cell
/// `rows[s]·shares[p]` rounded down or up, every row summing to `rows[s]` and
/// every column `p` to `totals[p]`.
///
/// Which cells round up is chosen greedily, keeping a rounding possible at
/// every step: first the cells whose floor is 0 (rounding them up gives a
/// stratum a partition it would otherwise lack), then the rest; within each,
/// smaller rows first (the strata that most risk an empty partition), later
/// columns first (`test` before `val` before `train`: a stratum absent from
/// evaluation is unmeasured), larger fractional parts first, then row order.
pub(crate) fn controlled(rows: &[u64], shares: &[f64], totals: &[u64]) -> Result<Vec<Vec<u64>>> {
    let (n_rows, n_cols) = (rows.len(), shares.len());
    let mut floor = vec![vec![0u64; n_cols]; n_rows];
    let mut frac = vec![vec![0f64; n_cols]; n_rows];
    for (s, n) in rows.iter().enumerate() {
        for (p, r) in shares.iter().enumerate() {
            let (whole, part) = split_at_integer(*n as f64 * r);
            floor[s][p] = whole;
            frac[s][p] = part;
        }
    }
    let row_need: Vec<i64> =
        (0..n_rows).map(|s| rows[s] as i64 - floor[s].iter().map(|f| *f as i64).sum::<i64>()).collect();
    let col_need: Vec<i64> =
        (0..n_cols).map(|p| totals[p] as i64 - (0..n_rows).map(|s| floor[s][p] as i64).sum::<i64>()).collect();
    let impossible = || {
        Error::invalid(format!(
            "no controlled rounding of strata {rows:?} into totals {totals:?}: the totals do not round the shares"
        ))
    };
    if row_need.iter().chain(&col_need).any(|n| *n < 0) || row_need.iter().sum::<i64>() != col_need.iter().sum::<i64>()
    {
        return Err(impossible());
    }
    let allowed = |s: usize, p: usize| frac[s][p] > 0.0;
    let mut up = initial(n_rows, n_cols, &row_need, &col_need, &allowed).ok_or_else(impossible)?;

    // `Some(true)`: rounded up for good; `Some(false)`: rounded down for good.
    let mut fixed = vec![vec![None::<bool>; n_cols]; n_rows];
    let mut cells: Vec<(usize, usize)> =
        (0..n_rows).flat_map(|s| (0..n_cols).map(move |p| (s, p))).filter(|(s, p)| allowed(*s, *p)).collect();
    cells.sort_by(|a, b| {
        (floor[a.0][a.1] > 0)
            .cmp(&(floor[b.0][b.1] > 0))
            .then(rows[a.0].cmp(&rows[b.0]))
            .then(b.1.cmp(&a.1))
            .then(frac[b.0][b.1].total_cmp(&frac[a.0][a.1]))
            .then(a.0.cmp(&b.0))
    });
    for (s, p) in cells {
        if up[s][p] {
            fixed[s][p] = Some(true);
            continue;
        }
        match exchange(&up, &fixed, &allowed, s, p) {
            Some(path) => {
                up[s][p] = true;
                for (r, c, value) in path {
                    up[r][c] = value;
                }
                fixed[s][p] = Some(true);
            }
            None => fixed[s][p] = Some(false),
        }
    }
    Ok((0..n_rows).map(|s| (0..n_cols).map(|p| floor[s][p] + u64::from(up[s][p])).collect()).collect())
}

/// Some 0/1 choice of allowed cells with the row and column sums asked for:
/// a maximum flow, found by augmenting paths.
fn initial(
    n_rows: usize,
    n_cols: usize,
    row_need: &[i64],
    col_need: &[i64],
    allowed: &dyn Fn(usize, usize) -> bool,
) -> Option<Vec<Vec<bool>>> {
    let mut up = vec![vec![false; n_cols]; n_rows];
    let mut row_left = row_need.to_vec();
    let mut col_left = col_need.to_vec();
    for s in 0..n_rows {
        while row_left[s] > 0 {
            // Breadth first from row `s` over the residual graph: a row
            // reaches a column through a cell not yet chosen, a column a row
            // through a cell chosen (to give it up); it ends at a column with
            // room.
            let mut col_from: Vec<Option<usize>> = vec![None; n_cols];
            let mut row_from: Vec<Option<usize>> = vec![None; n_rows];
            let mut row_seen = vec![false; n_rows];
            row_seen[s] = true;
            let mut queue = std::collections::VecDeque::from([s]);
            let mut end = None;
            'search: while let Some(r) = queue.pop_front() {
                for c in 0..n_cols {
                    if col_from[c].is_some() || up[r][c] || !allowed(r, c) {
                        continue;
                    }
                    col_from[c] = Some(r);
                    if col_left[c] > 0 {
                        end = Some(c);
                        break 'search;
                    }
                    for r2 in 0..n_rows {
                        if up[r2][c] && !row_seen[r2] {
                            row_seen[r2] = true;
                            row_from[r2] = Some(c);
                            queue.push_back(r2);
                        }
                    }
                }
            }
            let mut c = end?;
            col_left[c] -= 1;
            row_left[s] -= 1;
            loop {
                let r = col_from[c].expect("on the path");
                up[r][c] = true;
                if r == s {
                    break;
                }
                let previous = row_from[r].expect("on the path");
                up[r][previous] = false;
                c = previous;
            }
        }
    }
    col_left.iter().all(|n| *n == 0).then_some(up)
}

/// Cells to change so that cell `(s, p)` can round up with every sum kept:
/// an alternating cycle through it that changes no fixed cell, as
/// `(row, column, rounded up)`.
fn exchange(
    up: &[Vec<bool>],
    fixed: &[Vec<Option<bool>>],
    allowed: &dyn Fn(usize, usize) -> bool,
    s: usize,
    p: usize,
) -> Option<Vec<(usize, usize, bool)>> {
    let (n_rows, n_cols) = (up.len(), up.first().map_or(0, Vec::len));
    // Column `p` has one too many: give up a chosen cell in it, which leaves
    // its row one short; that row takes an unchosen cell, and so on, until a
    // chosen cell of row `s` --- one too many since `(s, p)` --- is given up.
    let mut row_from: Vec<Option<usize>> = vec![None; n_rows];
    let mut col_from: Vec<Option<usize>> = vec![None; n_cols];
    let mut col_seen = vec![false; n_cols];
    col_seen[p] = true;
    let mut queue = std::collections::VecDeque::from([p]);
    while let Some(c) = queue.pop_front() {
        for r in 0..n_rows {
            if !up[r][c] || fixed[r][c] == Some(true) {
                continue;
            }
            if r == s {
                let mut path = vec![(s, c, false)];
                let mut column = c;
                while column != p {
                    let row = col_from[column].expect("on the path");
                    path.push((row, column, true));
                    column = row_from[row].expect("on the path");
                    path.push((row, column, false));
                }
                return Some(path);
            }
            if row_from[r].is_some() {
                continue;
            }
            row_from[r] = Some(c);
            for c2 in 0..n_cols {
                if col_seen[c2] || up[r][c2] || !allowed(r, c2) || fixed[r][c2] == Some(false) {
                    continue;
                }
                col_seen[c2] = true;
                col_from[c2] = Some(r);
                queue.push_back(c2);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sums(cells: &[Vec<u64>]) -> (Vec<u64>, Vec<u64>) {
        let rows = cells.iter().map(|r| r.iter().sum()).collect();
        let cols = (0..cells[0].len()).map(|p| cells.iter().map(|r| r[p]).sum()).collect();
        (rows, cols)
    }

    #[test]
    fn largest_remainder_meets_the_total() {
        assert_eq!(largest_remainder(115, &[0.7, 0.15, 0.15]), [81, 17, 17]);
        assert_eq!(largest_remainder(10, &[0.7, 0.15, 0.15]), [7, 2, 1]);
        assert_eq!(largest_remainder(7, &[0.2; 5]), [2, 2, 1, 1, 1]);
    }

    /// Controlled rounding is deterministic, rounds every cell to its floor
    /// or ceiling, and meets both margins --- and gives each rare stratum a
    /// test group.
    #[test]
    fn s12_3_controlled_rounding_is_deterministic() {
        let shares = [0.7, 0.15, 0.15];
        let rows = [60, 30, 10, 3, 5, 7];
        let totals = largest_remainder(rows.iter().sum(), &shares);
        let cells = controlled(&rows, &shares, &totals).unwrap();
        assert_eq!(cells, controlled(&rows, &shares, &totals).unwrap());
        let (row_sums, col_sums) = sums(&cells);
        assert_eq!(row_sums, rows);
        assert_eq!(col_sums, totals);
        for (s, n) in rows.iter().enumerate() {
            for (p, r) in shares.iter().enumerate() {
                let exact = *n as f64 * r;
                assert!((cells[s][p] as f64 - exact).abs() < 1.0, "{s},{p}: {} for {exact}", cells[s][p]);
            }
        }
        for rare in 3..6 {
            assert!(cells[rare][2] >= 1, "stratum of {} has no test group: {:?}", rows[rare], cells[rare]);
        }
    }

    #[test]
    fn whole_products_are_whole() {
        let cells = controlled(&[10, 20], &[0.7, 0.3], &[21, 9]).unwrap();
        assert_eq!(cells, [[7, 3], [14, 6]]);
    }
}
