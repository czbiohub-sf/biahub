"""Planning a registration: which positions to estimate on, and how many transforms.

Before registering, `survey_positions` says which positions are usable and how their
structure lies, and `select_positions` picks where to estimate and what to hold out.
After estimating on a sample, `transform_spread` says whether the dataset needs one
transform, one per position or per timepoint, and `pool_transforms` makes one shared
transform from per-position estimates (their medoid).
"""

from __future__ import annotations

import numpy as np

from skimage import filters

from biahub.cli.parsing import position_key
from biahub.core.nodes.solvers import medoid as _medoid
from biahub.core.transform import Transform
from biahub.registration.methods.landmark.nuclei_matching import detect_nuclei
from biahub.utils.ngff import is_unwritten

# --- how many transforms ------------------------------------------------------------------
# Estimates on a sample of positions and timepoints, compared: does the plate need one
# transform, one per position, one per timepoint, or one per position and timepoint?


def _widest_pair(
    transforms: dict, shape_zyx: tuple, voxel_um: tuple, gap_key: str = "max_um"
) -> tuple[float, tuple]:
    """Return the largest `gap_key` gap between any two of `transforms` (by key), and the pair."""
    keys = list(transforms)
    best, pair = 0.0, ()
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            gap = transforms[keys[a]].displacement_to(transforms[keys[b]], shape_zyx, voxel_um)
            if gap[gap_key] > best or not pair:
                best, pair = gap[gap_key], (keys[a], keys[b])
    return best, pair


def transform_spread(
    estimates: dict,
    shape_zyx: tuple,
    voxel_um: tuple,
    tolerance_um: float = 0.5,
) -> dict:
    """Say how many transforms a dataset needs, from estimates on a sample.

    `estimates` maps (position, t) to a forward `Transform`. The spread across positions
    is the largest gap (`Transform.displacement_to`, max over the volume, µm) between two
    positions at one timepoint; across timepoints, between two timepoints at one position.
    yx and z are judged apart against `tolerance_um` (default 0.5 µm, about a good
    registration's held-out residual; pass the dataset's own): `choice_yx` and `choice_z`
    are "one", "per position", "per timepoint" or "per position and timepoint", and
    `choice` is what both need (e.g. yx one, z per position: per position). An axis with
    one sample shows no spread (None). Measured over one grid (`shape_zyx`, `voxel_um`):
    pass the moving volume's shape with the reference's voxel size.
    """
    positions = sorted({p for p, _ in estimates})
    times = sorted({t for _, t in estimates})
    by_t = [(t, {p: estimates[p, t] for p in positions if (p, t) in estimates}) for t in times]
    by_p = [
        (p, {(p, t): estimates[p, t] for t in times if (p, t) in estimates}) for p in positions
    ]

    def widest(groups, gap_key="max_um"):
        best, worst, where = None, (), None
        for label, group in groups:
            if len(group) < 2:
                continue
            gap, pair = _widest_pair(group, shape_zyx, voxel_um, gap_key)
            if best is None or gap > best:
                best, worst, where = gap, pair, label
        return best, worst, where

    def over(gap):
        return gap is not None and gap > tolerance_um

    def choice(by_position, by_time):
        return {
            (False, False): "one",
            (True, False): "per position",
            (False, True): "per timepoint",
            (True, True): "per position and timepoint",
        }[by_position, by_time]

    across_p, worst_p, worst_p_t = widest(by_t)
    across_t, worst_t, _ = widest(by_p)
    axes = {
        axis: (widest(by_t, f"max_{axis}_um")[0], widest(by_p, f"max_{axis}_um")[0])
        for axis in ("yx", "z")
    }
    choices = {axis: choice(over(p), over(t)) for axis, (p, t) in axes.items()}
    return {
        "choice": choice(
            any(over(p) for p, _ in axes.values()), any(over(t) for _, t in axes.values())
        ),
        "choice_yx": choices["yx"],
        "choice_z": choices["z"],
        "tolerance_um": tolerance_um,
        "across_positions_um": across_p,
        "across_timepoints_um": across_t,
        **{f"across_positions_{axis}_um": p for axis, (p, _) in axes.items()},
        **{f"across_timepoints_{axis}_um": t for axis, (_, t) in axes.items()},
        "worst_positions": list(worst_p),
        "worst_positions_t": worst_p_t,
        "worst_timepoints": [list(k) for k in worst_t],
        "n_positions": len(positions),
        "n_timepoints": len(times),
    }


def pool_transforms(
    estimates: dict,
    shape_zyx: tuple,
    voxel_um: tuple,
    z: str = "medoid",
    tolerance_um: float = 0.5,
) -> dict:
    """Return one shared transform from per-position estimates (position -> `Transform`).

    The medoid: the estimate with the smallest summed gap (`displacement_to`, max over
    the volume) to all the others -- a real estimate, not an average of matrices. With
    `z="median"`, its z is moved so the volume's centre lands at the median depth of all
    estimates (when yx agrees across positions and z does not). Returns the `transform`,
    the `medoid` position, the `z_shift_um` applied, `gap_to_medoid_um` (the median
    over the other estimates of their largest yx and z gap to the medoid: robust to one
    noisy estimate, unlike the widest pair) and the estimates' `spread`
    (`transform_spread`, the widest pair).
    """
    if z not in ("medoid", "median"):
        raise ValueError(f"z must be 'medoid' or 'median', not {z!r}")
    keys = sorted(estimates)
    if not keys:
        raise ValueError("no estimates to pool")
    gaps = np.array(
        [
            [
                0.0
                if a == b
                else estimates[a].displacement_to(estimates[b], shape_zyx, voxel_um)["max_um"]
                for b in keys
            ]
            for a in keys
        ]
    )
    medoid = keys[_medoid(gaps)]
    matrix = np.array(estimates[medoid].matrix, dtype=float)
    z_shift = 0.0
    if z == "median":
        centre = (np.asarray(shape_zyx, dtype=float) - 1) / 2
        depths = [estimates[k].apply_points(centre[None])[0, 0] for k in keys]
        z_shift = float(np.median(depths) - estimates[medoid].apply_points(centre[None])[0, 0])
        matrix[0, 3] += z_shift
    spread = transform_spread(
        {(k, 0): estimates[k] for k in keys}, shape_zyx, voxel_um, tolerance_um
    )
    gaps = [
        estimates[medoid].displacement_to(estimates[k], shape_zyx, voxel_um)
        for k in keys
        if k != medoid
    ]
    return {
        "transform": Transform(matrix),
        "medoid": medoid,
        "gap_to_medoid_um": {
            axis: float(np.median([g[f"max_{axis}_um"] for g in gaps])) if gaps else None
            for axis in ("yx", "z")
        },
        "z_shift_um": z_shift * voxel_um[0],
        "spread": spread,
    }


# --- survey before registering ------------------------------------------------------------


def survey_positions(
    position_dirpaths: list,
    channels: list[str],
    t: int = 0,
    nuclei: bool = False,
    cells: int = 4,
) -> list[dict]:
    """Return one row per position: whether it is usable, and how its structure lies.

    `empty`: never written (no stored chunks, metadata only). From the first channel at
    `t`: `nan_fraction`; `foreground`, the share of the max projection above Otsu; and
    `spread`, the share of a `cells` x `cells` grid holding foreground (structure all over
    the field constrains scale and rotation; a corner does not). `peak_z` per channel:
    the slice of highest mean signal. `n_nuclei` when `nuclei` (detect_nuclei).
    """
    from iohub import open_ome_zarr

    rows = []
    for path in position_dirpaths:
        row = {"position": position_key(path), "empty": False}
        with open_ome_zarr(path, mode="r") as position:
            if is_unwritten(position):
                rows.append({**row, "empty": True})
                continue
            index = [position.channel_names.index(c) for c in channels]
            volumes = [np.asarray(position.data[t, i], dtype=np.float32) for i in index]
            voxel_um = tuple(position.scale[-3:])
        first = volumes[0]
        row["nan_fraction"] = float(np.isnan(first).mean())
        row["peak_z"] = {
            c: int(np.nanargmax(np.nanmean(v, axis=(1, 2)))) if np.isfinite(v).any() else None
            for c, v in zip(channels, volumes, strict=True)
        }
        mip = np.nanmax(first, axis=0) if np.isfinite(first).any() else None
        if mip is None or np.nanstd(mip) == 0:
            rows.append({**row, "foreground": 0.0, "spread": 0.0})
            continue
        mip = np.nan_to_num(mip, nan=float(np.nanmin(mip)))
        mask = mip > filters.threshold_otsu(mip)
        ny, nx = mask.shape
        filled = [
            mask[
                i * ny // cells : (i + 1) * ny // cells,
                j * nx // cells : (j + 1) * nx // cells,
            ].mean()
            > 0.05
            for i in range(cells)
            for j in range(cells)
        ]
        row["foreground"] = float(mask.mean())
        row["spread"] = float(np.mean(filled))
        if nuclei:
            row["n_nuclei"] = int(len(detect_nuclei(first, voxel_um)))
        rows.append(row)
    return rows


def select_positions(survey_rows: list[dict], n_held_out: int = 3) -> tuple[str, list[str]]:
    """Return (the position to estimate on, held-out positions) from `survey_positions`.

    Usable positions are written and under half NaN. The estimate position has the most
    spread-out structure (what constrains scale and rotation); held-out ones are the next
    most spread, from other wells first, so the evaluation sees how the transform carries
    across the plate.
    """
    usable = [
        r for r in survey_rows if not r.get("empty") and (r.get("nan_fraction") or 0) < 0.5
    ]
    if not usable:
        raise ValueError("no usable position: every one is empty or mostly NaN")
    ranked = sorted(usable, key=lambda r: -(r.get("spread") or 0))
    estimate = ranked[0]["position"]

    def well(position):
        return position.rsplit("/", 1)[0]

    others = [r["position"] for r in ranked[1:] if well(r["position"]) != well(estimate)]
    same = [r["position"] for r in ranked[1:] if well(r["position"]) == well(estimate)]
    return estimate, (others + same)[:n_held_out]
