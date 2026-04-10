"""Reproducible train/val/test splits with optional stratification and grouping.

All splitting logic uses only numpy + stdlib. Supports three common
patterns, any two of which can be combined:

- Ratio split with a named partition (``ratios={"train": 0.7, ...}``)
- K-fold cross-validation (``k_folds=5``)
- Stratification by image-level label (``stratify_by="label"``)
- Grouping by a dotted path into ``extra`` (``group_by="extra.patient_id"``)
  — all records sharing the same group value land in the same partition.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from medh5.dataset.index import Dataset, DatasetRecord
from medh5.exceptions import MEDH5ValidationError


def _resolve_key(record: DatasetRecord, key: str) -> Any:
    """Look up a dotted path on a record (supports ``extra.foo.bar``)."""
    if "." not in key:
        return getattr(record, key, None)
    head, _, tail = key.partition(".")
    value: Any = getattr(record, head, None)
    for part in tail.split("."):
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
    return value


def _group_records(
    ds: Dataset, group_by: str | None
) -> tuple[list[list[int]], list[Any]]:
    """Group record indices by group key. Ungrouped => each record its own group."""
    if group_by is None:
        return [[i] for i in range(len(ds))], [None] * len(ds)

    groups: dict[Any, list[int]] = {}
    keys_order: list[Any] = []
    for i, rec in enumerate(ds.records):
        key = _resolve_key(rec, group_by)
        # Unhashable fallback
        try:
            hash(key)
        except TypeError:
            key = repr(key)
        if key not in groups:
            groups[key] = []
            keys_order.append(key)
        groups[key].append(i)
    return [groups[k] for k in keys_order], keys_order


def _stratum_key(rec: DatasetRecord, stratify_by: str | None) -> Any:
    if stratify_by is None:
        return None
    value = _resolve_key(rec, stratify_by)
    try:
        hash(value)
    except TypeError:
        value = repr(value)
    return value


def _group_stratum(
    ds: Dataset, group_indices: list[int], stratify_by: str | None
) -> Any:
    """Return the stratum for a group (first record's stratum value)."""
    if stratify_by is None:
        return None
    return _stratum_key(ds.records[group_indices[0]], stratify_by)


def _ratio_split(
    ratios: dict[str, float],
    groups: list[list[int]],
    strata: list[Any],
    seed: int,
) -> dict[str, list[int]]:
    """Distribute groups into named partitions according to *ratios*.

    Groups within the same stratum are distributed proportionally; any
    rounding remainder lands in the largest partition. Deterministic
    given *seed*.
    """
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise MEDH5ValidationError(
            f"ratios must sum to 1.0, got {total}: {ratios}"
        )

    rng = np.random.default_rng(seed)
    partition_names = list(ratios.keys())
    out: dict[str, list[int]] = {name: [] for name in partition_names}

    # Bucket groups by stratum
    by_stratum: dict[Any, list[int]] = {}
    for g_idx, s in enumerate(strata):
        by_stratum.setdefault(s, []).append(g_idx)

    for stratum_groups in by_stratum.values():
        shuffled = list(stratum_groups)
        rng.shuffle(shuffled)
        n = len(shuffled)

        # Compute integer targets that sum to n using largest-remainder.
        raw = np.array([ratios[name] * n for name in partition_names])
        floored = np.floor(raw).astype(int)
        remainder = n - int(floored.sum())
        if remainder > 0:
            order = np.argsort(-(raw - floored))
            for j in range(remainder):
                floored[order[j % len(partition_names)]] += 1

        cursor = 0
        for name, take in zip(partition_names, floored, strict=True):
            for g_idx in shuffled[cursor : cursor + int(take)]:
                out[name].extend(groups[g_idx])
            cursor += int(take)

    # Preserve original record order within each partition
    for name in partition_names:
        out[name].sort()
    return out


def _kfold_split(
    k: int,
    groups: list[list[int]],
    strata: list[Any],
    seed: int,
) -> list[dict[str, list[int]]]:
    """Build K (train, val) folds over group indices, stratum-preserving."""
    if k < 2:
        raise MEDH5ValidationError(f"k_folds must be >= 2, got {k}")

    rng = np.random.default_rng(seed)
    group_fold = np.full(len(groups), -1, dtype=int)

    by_stratum: dict[Any, list[int]] = {}
    for g_idx, s in enumerate(strata):
        by_stratum.setdefault(s, []).append(g_idx)

    for stratum_groups in by_stratum.values():
        shuffled = list(stratum_groups)
        rng.shuffle(shuffled)
        for i, g_idx in enumerate(shuffled):
            group_fold[g_idx] = i % k

    folds: list[dict[str, list[int]]] = []
    for fold in range(k):
        train: list[int] = []
        val: list[int] = []
        for g_idx, members in enumerate(groups):
            target = val if group_fold[g_idx] == fold else train
            target.extend(members)
        train.sort()
        val.sort()
        folds.append({"train": train, "val": val})
    return folds


def make_splits(
    ds: Dataset,
    *,
    ratios: dict[str, float] | None = None,
    k_folds: int | None = None,
    stratify_by: str | None = None,
    group_by: str | None = None,
    seed: int = 0,
) -> dict[str, Dataset] | list[dict[str, Dataset]]:
    """Split *ds* into named partitions or k-fold folds.

    Exactly one of ``ratios`` or ``k_folds`` must be given.

    Parameters
    ----------
    ds : Dataset
        Input dataset to split.
    ratios : dict[str, float], optional
        Named partition sizes that must sum to ``1.0`` (e.g.
        ``{"train": 0.7, "val": 0.15, "test": 0.15}``).
    k_folds : int, optional
        Number of cross-validation folds (``>= 2``). Returns a list of
        ``{"train": Dataset, "val": Dataset}`` dicts.
    stratify_by : str, optional
        Dotted path into a record whose value defines the stratum (e.g.
        ``"label"``, ``"extra.site"``). Groups with the same value are
        balanced across partitions.
    group_by : str, optional
        Dotted path whose value groups records together — all records
        sharing the same value land in the same partition.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict[str, Dataset] or list[dict[str, Dataset]]
        For ratio splits, a dict keyed by partition name. For k-fold,
        a list of ``{"train", "val"}`` dicts of length ``k_folds``.

    Raises
    ------
    MEDH5ValidationError
        On invalid ratios / k_folds / combination.
    """
    if (ratios is None) == (k_folds is None):
        raise MEDH5ValidationError(
            "Exactly one of 'ratios' or 'k_folds' must be provided."
        )

    groups, _group_keys = _group_records(ds, group_by)
    strata = [_group_stratum(ds, g, stratify_by) for g in groups]

    if ratios is not None:
        index_map = _ratio_split(ratios, groups, strata, seed)
        return {
            name: Dataset([ds.records[i] for i in idxs])
            for name, idxs in index_map.items()
        }

    assert k_folds is not None
    fold_maps = _kfold_split(k_folds, groups, strata, seed)
    return [
        {name: Dataset([ds.records[i] for i in idxs]) for name, idxs in fm.items()}
        for fm in fold_maps
    ]
