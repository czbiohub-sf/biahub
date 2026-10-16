"""Registration evaluation, layer 1: geometry -- plausibility and residual statistics in µm."""

import numpy as np
import pytest

from biahub.core.transform import Transform
from biahub.registration.evaluate import decompose, displacement_stats, residual_affine

VOX = (0.2, 0.103, 0.103)  # z, y, x in µm (DragonFly)


def _similarity(scale_zyx, angle_deg, shift_zyx):
    a = np.deg2rad(angle_deg)
    m = np.eye(4)
    m[1:3, 1:3] = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    m[:3, :3] = m[:3, :3] @ np.diag(scale_zyx)
    m[:3, 3] = shift_zyx
    return Transform(m)


def test_decompose_recovers_scale_rotation_and_translation():
    t = _similarity((1.0, 0.95, 0.95), 0.5, (6.0, 50.0, 37.0))
    d = decompose(t, VOX, VOX)
    np.testing.assert_allclose(d["scale_zyx"], [1.0, 0.95, 0.95], atol=1e-9)
    assert d["rotation_yx_deg"] == pytest.approx(0.5, abs=1e-9)
    assert d["rotation_deg"] == pytest.approx(0.5, abs=1e-9)
    np.testing.assert_allclose(d["shear"], 0.0, atol=1e-9)
    np.testing.assert_allclose(d["translation_um"], [6.0 * 0.2, 50.0 * 0.103, 37.0 * 0.103])


def test_decompose_reads_scale_in_physical_units():
    # Equal voxel matrices mean different physical scales when the voxel sizes differ: a
    # moving voxel of 0.116 µm mapped 1:1 onto a reference voxel of 0.1494 µm enlarges.
    d = decompose(Transform.identity(3), (0.174, 0.116, 0.116), (0.174, 0.1494, 0.1494))
    np.testing.assert_allclose(d["scale_zyx"], [1.0, 0.1494 / 0.116, 0.1494 / 0.116])


def test_displacement_stats_in_microns():
    diff = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -2.0], [1.0, 2.0, 2.0]])
    s = displacement_stats(diff, VOX)
    assert s["n"] == 4
    um = np.linalg.norm(diff * VOX, axis=1)
    assert s["median_um"] == pytest.approx(np.median(um))
    assert s["p90_um"] == pytest.approx(np.percentile(um, 90))
    np.testing.assert_allclose(s["median_zyx_um"], np.median(diff * VOX, axis=0))
    np.testing.assert_allclose(s["rms_zyx_um"], np.sqrt(((diff * VOX) ** 2).mean(axis=0)))
    assert displacement_stats(np.zeros((0, 3)), VOX)["n"] == 0


def test_residual_affine_recovers_a_scale_error_and_a_rotation():
    # Residuals that grow outward are a scale error; a rotation leaves a curl. Both must be
    # recovered from noisy matched displacements, and a constant shift must not read as either.
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(0, 60, 400), rng.uniform(0, 2048, 400), rng.uniform(0, 2048, 400)]
    error = _similarity((1.0, 1.002, 1.002), 0.1, (0.5, 3.0, -2.0))
    moved = error.apply_points(pts)
    diff = moved - pts + rng.normal(0, 0.3, pts.shape)
    r = residual_affine(pts, diff, VOX)
    np.testing.assert_allclose(r["scale_zyx"][1:], [1.002, 1.002], atol=2e-4)
    assert r["rotation_yx_deg"] == pytest.approx(0.1, abs=0.01)
    assert r["n"] == 400

    shift_only = residual_affine(pts, np.tile([0.5, 3.0, -2.0], (400, 1)), VOX)
    np.testing.assert_allclose(shift_only["scale_zyx"], 1.0, atol=1e-9)
    assert shift_only["rotation_yx_deg"] == pytest.approx(0.0, abs=1e-9)


def test_residual_affine_needs_enough_points():
    r = residual_affine(np.zeros((3, 3)), np.zeros((3, 3)), VOX)
    assert r["n"] == 3 and r["scale_zyx"] is None


# --- layer 2: point-based residuals -----------------------------------------------------

from biahub.registration.evaluate import (  # noqa: E402
    Landmarks,
    detect_beads,
    detect_nuclei,
    point_residuals,
)


def test_clicked_pairs_are_matched_by_their_ids_not_by_distance():
    # A person clicked point i in both channels: the correspondence is known, even where
    # another point is closer.
    ref = Landmarks(
        np.array([[10.0, 100, 100], [10, 110, 100], [20, 500, 500]]),
        VOX,
        "manual",
        pair_ids=[0, 1, 2],
    )
    mov = Landmarks(ref.points_vox + [0.0, 3.0, 0.0], VOX, "manual", pair_ids=[0, 1, 2])
    r = point_residuals(Transform.identity(3), mov, ref)
    assert r["n_matched"] == 3
    np.testing.assert_allclose(r["stats"]["median_zyx_um"], [0.0, -3 * 0.103, 0.0])


def test_detected_points_match_mutual_nearest_within_the_radius():
    rng = np.random.default_rng(1)
    ref_pts = np.c_[
        rng.uniform(5, 55, 80), rng.uniform(50, 2000, 80), rng.uniform(50, 2000, 80)
    ]
    truth = _similarity((1.0, 0.96, 0.96), 0.2, (2.0, 40.0, 30.0))  # forward: mov -> ref
    mov_pts = truth.invert().apply_points(ref_pts) + rng.normal(0, 0.2, ref_pts.shape)
    mov_pts = np.vstack([mov_pts, [[30.0, 1000.0, 1000.0]]])  # an unmatched extra bead
    ref_l, mov_l = Landmarks(ref_pts, VOX, "beads"), Landmarks(mov_pts, VOX, "beads")

    good = point_residuals(truth, mov_l, ref_l)
    assert good["n_matched"] >= 78 and good["n_mov"] == 81
    assert good["stats"]["median_um"] < 0.1
    off = point_residuals(Transform.identity(3), mov_l, ref_l)
    assert off["n_matched"] < 10  # far beyond the 1 µm radius: nothing pairs up


def test_beads_are_located_to_a_fraction_of_a_voxel():
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(2)
    # a grid wider than the detector's minimum 3D spacing (16 voxels), with sub-voxel offsets
    grid = np.array(
        [(z, y, x) for z in (10, 30) for y in (24, 64, 104) for x in (32, 96)], dtype=float
    )
    centres = grid + rng.uniform(-0.5, 0.5, grid.shape)
    zz, yy, xx = np.meshgrid(np.arange(40), np.arange(128), np.arange(128), indexing="ij")
    vol = np.zeros((40, 128, 128), np.float32)
    for c in centres:
        vol += 1000 * np.exp(
            -(((zz - c[0]) / 2.0) ** 2 + ((yy - c[1]) / 1.5) ** 2 + ((xx - c[2]) / 1.5) ** 2)
        )
    vol = gaussian_filter(vol, 0.3) + rng.normal(0, 5, vol.shape).astype(np.float32)
    found = detect_beads(vol, VOX, threshold_abs=200)
    d, _ = (
        __import__("scipy.spatial", fromlist=["cKDTree"])
        .cKDTree(found.points_vox)
        .query(centres)
    )
    assert len(found.points_vox) == 12 and np.max(d) < 0.25  # sub-voxel


def test_nuclei_centroids_skip_objects_cut_by_the_frame():
    zz, yy, xx = np.meshgrid(np.arange(30), np.arange(200), np.arange(200), indexing="ij")
    vol = np.full((30, 200, 200), 100.0, np.float32)
    inside = [(15.0, 60.0, 60.0), (12.0, 140.0, 120.0)]
    for cz, cy, cx in inside + [(15.0, 100.0, 2.0)]:  # the last one touches the frame
        vol += 800 * (
            ((zz - cz) / 6) ** 2 + ((yy - cy) / 18) ** 2 + ((xx - cx) / 18) ** 2 <= 1
        )
    found = detect_nuclei(vol, (0.2, 0.103, 0.103), min_area_um2=5, max_area_um2=300)
    assert len(found.points_vox) == 2
    d = np.linalg.norm(
        np.sort(found.points_vox, axis=0) - np.sort(np.array(inside), axis=0), axis=1
    )
    assert np.max(d) < 1.0
