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
