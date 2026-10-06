//! Finding the transform between two frames, or two timepoints (spec §10).
//!
//! Resolution walks the **frame graph** rather than matching names.  Inverses
//! are used where they can be evaluated and never invented, a transform and
//! the stored inverse its `inverse_id` names are one route (§10.1), and two
//! equally short routes are refused rather than one picked (§10.2).

use std::collections::{BTreeSet, HashSet};

use indexmap::IndexMap;

use super::model::{can_invert, stored_inverse, Transform};
use crate::geometry::grid::Grid;
use crate::json::repr_str;
use crate::{Error, Result};

/// Distinct routes to keep per frame --- two is enough to prove a tie.
const AMBIGUITY_EVIDENCE: usize = 2;

/// Ids of transforms that are one half of a stored inverse pair.
fn paired(transforms: &IndexMap<String, Transform>) -> Result<HashSet<String>> {
    let mut out = HashSet::new();
    for (id, transform) in transforms {
        if let Some(other) = stored_inverse(transform)? {
            if transforms.contains_key(&other.transform_id) {
                out.insert(id.clone());
                out.insert(other.transform_id.clone());
            }
        }
    }
    Ok(out)
}

/// Frame -> `[(neighbour, step)]`, including inverses where they are usable.
fn edges(transforms: &IndexMap<String, Transform>) -> Result<IndexMap<String, Vec<(String, Transform)>>> {
    let paired = paired(transforms)?;
    let mut out: IndexMap<String, Vec<(String, Transform)>> = IndexMap::new();
    for (id, transform) in transforms {
        out.entry(transform.from_frame().to_string())
            .or_default()
            .push((transform.to_frame().to_string(), transform.clone()));
        out.entry(transform.to_frame().to_string()).or_default();
        if paired.contains(id) {
            continue;
        }
        if can_invert(transform)? {
            out.get_mut(transform.to_frame())
                .expect("inserted above")
                .push((transform.from_frame().to_string(), Transform::inverse_of(transform.clone())));
        }
    }
    Ok(out)
}

/// Frame -> frames one hop away, as [`resolve_between`] would walk them.
pub fn frame_graph(transforms: &IndexMap<String, Transform>) -> Result<IndexMap<String, Vec<String>>> {
    let mut graph: IndexMap<String, Vec<String>> = IndexMap::new();
    for transform in transforms.values() {
        graph.entry(transform.from_frame().to_string()).or_default().push(transform.to_frame().to_string());
        graph.entry(transform.to_frame().to_string()).or_default();
        if can_invert(transform)? {
            graph.get_mut(transform.to_frame()).expect("inserted above").push(transform.from_frame().to_string());
        }
    }
    Ok(graph)
}

/// The shortest transform path between two frames, or `None`.
///
/// A single hop returns the transform itself, several a chain.  Two distinct
/// minimal-length routes are refused (E501): nothing in the file says which
/// is authoritative.
pub fn resolve_between(
    transforms: &IndexMap<String, Transform>,
    from_frame: &str,
    to_frame: &str,
) -> Result<Option<Transform>> {
    if from_frame == to_frame {
        return Ok(None);
    }
    let graph = edges(transforms)?;
    if !graph.contains_key(from_frame) {
        return Ok(None);
    }
    let mut frontier: IndexMap<String, Vec<Vec<Transform>>> = IndexMap::new();
    frontier.insert(from_frame.to_string(), vec![Vec::new()]);
    let mut seen: HashSet<String> = HashSet::from([from_frame.to_string()]);
    while !frontier.is_empty() {
        let mut arrivals: Vec<Vec<Transform>> = Vec::new();
        let mut next: IndexMap<String, Vec<Vec<Transform>>> = IndexMap::new();
        for (frame, paths) in &frontier {
            for (neighbour, step) in graph.get(frame).map(Vec::as_slice).unwrap_or(&[]) {
                for path in paths {
                    let mut extended = path.clone();
                    extended.push(step.clone());
                    if neighbour == to_frame {
                        arrivals.push(extended);
                        continue;
                    }
                    if seen.contains(neighbour) {
                        continue;
                    }
                    let routes = next.entry(neighbour.clone()).or_default();
                    if routes.len() < AMBIGUITY_EVIDENCE {
                        routes.push(extended);
                    }
                }
            }
        }
        if !arrivals.is_empty() {
            reject_ambiguous(&arrivals, from_frame, to_frame)?;
            let mut best = arrivals.swap_remove(0);
            return Ok(Some(if best.len() == 1 { best.remove(0) } else { Transform::chain(best)? }));
        }
        seen.extend(next.keys().cloned());
        frontier = next;
    }
    Ok(None)
}

fn reject_ambiguous(arrivals: &[Vec<Transform>], from_frame: &str, to_frame: &str) -> Result<()> {
    let routes: BTreeSet<String> = arrivals
        .iter()
        .map(|path| path.iter().map(|s| s.transform_id.as_str()).collect::<Vec<_>>().join(" -> "))
        .collect();
    if routes.len() < 2 {
        return Ok(());
    }
    let named: Vec<String> = routes.into_iter().collect();
    Err(Error::coded(
        "E501",
        format!(
            "{} equally short transform paths relate frame {} to {}: {}. They need not agree, and nothing in the \
             file says which one is authoritative, so picking one here would assert an alignment no one chose \
             (§10.2). Select the transform you want by id from `sample.transforms`.",
            named.len(),
            repr_str(from_frame),
            repr_str(to_frame),
            named.join("; ")
        ),
    ))
}

/// Every frame of reference used by a timepoint's grids.
pub fn frames_of_timepoint(grids: &IndexMap<String, Grid>, timepoint: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for grid in grids.values() {
        if grid.timepoint.as_deref() == Some(timepoint) {
            if let Some(frame) = grid.frame_uid.as_deref().filter(|f| !f.is_empty()) {
                if !out.iter().any(|f| f == frame) {
                    out.push(frame.to_string());
                }
            }
        }
    }
    out
}
