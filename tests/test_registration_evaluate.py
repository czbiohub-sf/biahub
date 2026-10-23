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
    moved = error.apply_points(pts)  # the reference positions; pts are where they map from
    diff = moved - pts + rng.normal(0, 0.3, pts.shape)
    r = residual_affine(moved, diff, VOX)
    np.testing.assert_allclose(r["scale_zyx"][1:], [1.002, 1.002], atol=2e-4)
    assert r["rotation_yx_deg"] == pytest.approx(0.1, abs=0.01)
    assert r["n"] == 400

    shift_only = residual_affine(pts, np.tile([0.5, 3.0, -2.0], (400, 1)), VOX)
    np.testing.assert_allclose(shift_only["scale_zyx"], 1.0, atol=1e-9)
    assert shift_only["rotation_yx_deg"] == pytest.approx(0.0, abs=1e-9)


def test_residual_affine_needs_enough_points():
    # DragonFly: 8 nuclei gave a residual scale of 0.86 (noise), 13-16 sensible values
    rng = np.random.default_rng(0)
    for n in (3, 8):
        r = residual_affine(rng.random((n, 3)) * 100, rng.random((n, 3)), VOX)
        assert r["n"] == n and r["scale_zyx"] is None


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


# --- layer 3: image-based residuals -----------------------------------------------------

from scipy.ndimage import affine_transform as _affine  # noqa: E402
from scipy.ndimage import shift as _ndi_shift  # noqa: E402

from biahub.registration.evaluate import tile_residuals, z_residuals  # noqa: E402

VOX3 = (0.2, 0.103, 0.103)


def _cells(seed=0, shape=(24, 256, 256)):
    """Blobs of several sizes on a background: structure everywhere, as cells."""
    from tests.test_registration_estimators import _synthetic_blob_volume

    rng = np.random.default_rng(seed)
    return _synthetic_blob_volume(rng, shape, n_blobs=120, sigma=3.0) + 100.0


def test_tile_residuals_measure_a_known_shift_with_the_point_sign_convention():
    ref = _cells()
    shift_zyx = (0.0, 3.0, -2.0)
    mov = _ndi_shift(ref, shift_zyx, order=1, mode="nearest")  # content at p + shift
    exact = Transform.from_translation([-s for s in shift_zyx])
    off = tile_residuals(Transform.identity(3), mov, ref, VOX3, tile_px=64)
    on = tile_residuals(exact, mov, ref, VOX3, tile_px=64)
    assert off["stats"]["n"] >= 6
    # reference minus mapped moving, as point_residuals: -shift
    np.testing.assert_allclose(
        off["stats"]["median_zyx_um"][1:], [-3 * 0.103, 2 * 0.103], atol=0.05
    )
    assert on["stats"]["median_um"] < 0.05


def test_tile_residuals_expose_a_scale_error():
    ref = _cells(1)
    c = (np.array(ref.shape) - 1) / 2
    scale = np.diag([1.0, 1.01, 1.01])
    # mov content sits at c + 1.01 (p - c): sample ref there by the inverse map
    inv = np.linalg.inv(scale)
    mov = _affine(ref, inv, offset=c - inv @ c, order=1, mode="nearest")
    r = tile_residuals(Transform.identity(3), mov, ref, VOX3, tile_px=64)
    np.testing.assert_allclose(r["affine"]["scale_zyx"][1:], [1 / 1.01, 1 / 1.01], atol=2e-3)


def test_z_residuals_recover_a_z_shift_in_every_crop():
    ref = _cells(2, shape=(40, 256, 256))
    mov = _ndi_shift(ref, (2.0, 0.0, 0.0), order=1, mode="nearest")
    r = z_residuals(Transform.identity(3), mov, ref, VOX3, crop_px=96)
    dz = [v[0] for v in r["crops_um"].values() if v is not None]
    assert len(dz) == 5
    np.testing.assert_allclose(dz, -2 * 0.2, atol=0.05)


def test_edges_register_inverted_contrast_where_intensities_do_not():
    # Cross-modal stand-in: the moving channel shows the same structures dark on bright.
    ref = _cells(3)
    mov = 2 * ref.mean() - _ndi_shift(ref, (0.0, 2.0, 0.0), order=1, mode="nearest")
    edges = tile_residuals(Transform.identity(3), mov, ref, VOX3, tile_px=64)
    intensity = tile_residuals(
        Transform.identity(3), mov, ref, VOX3, tile_px=64, compare="intensity"
    )
    assert edges["stats"]["n"] >= 6
    np.testing.assert_allclose(edges["stats"]["median_zyx_um"][1], -2 * 0.103, atol=0.05)
    assert intensity["stats"]["n"] < edges["stats"]["n"]


def _random_affine(rng, ndim=3):
    matrix = np.eye(ndim + 1)
    angle = np.deg2rad(rng.uniform(-10, 10))
    matrix[-3:-1, -3:-1] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    matrix[:ndim, :ndim] = matrix[:ndim, :ndim] @ (
        np.eye(ndim) + rng.uniform(-0.05, 0.05, (ndim, ndim))
    )
    matrix[:ndim, ndim] = rng.uniform(-6, 6, ndim)
    return Transform(matrix)


@pytest.mark.parametrize("ndim", [3, 2])
def test_coverage_from_geometry_matches_warping_a_volume_of_ones(ndim):
    # Coverage used to be a second full warp (a volume of ones; ~190 s on a 2048 x 2048
    # stack); it is where the inverse-mapped reference grid falls inside the moving volume,
    # with the edge each backend uses (ANTs half a voxel beyond the centres, scipy at them).
    from biahub.registration.evaluate import coverage

    rng = np.random.default_rng(4)
    for trial in range(12):
        transform = _random_affine(rng, ndim)
        if trial % 3 == 0:  # half-voxel shifts put sample points exactly on the edges
            transform = Transform.from_translation(np.round(rng.uniform(-6, 6, ndim) * 2) / 2)
        mov_shape = tuple(int(s) for s in rng.integers(16, 40, ndim))
        ref_shape = tuple(int(s) for s in rng.integers(16, 40, ndim))
        ones = transform.apply(
            np.ones(mov_shape, np.float32), reference=np.zeros(ref_shape, np.float32)
        )
        np.testing.assert_array_equal(coverage(transform, mov_shape, ref_shape), ones > 0.5)


def test_tiles_and_z_share_one_prepared_pair(monkeypatch):
    # Each function prepared (Sobel + warp, ~240 s on a 2048 x 2048 stack) on its own; a
    # pair prepared once serves both, with the same results.
    from biahub.registration.evaluate import prepare_pair

    ref = _cells(5, shape=(32, 256, 256))
    mov = _ndi_shift(ref, (1.0, 2.0, -1.0), order=1, mode="nearest")
    transform = Transform.from_translation([-0.5, -1.0, 0.5])
    alone = (
        tile_residuals(transform, mov, ref, VOX3, tile_px=64),
        z_residuals(transform, mov, ref, VOX3, crop_px=96),
    )
    warps = []
    real = Transform.apply
    monkeypatch.setattr(
        Transform, "apply", lambda self, *a, **k: warps.append(1) or real(self, *a, **k)
    )
    pair = prepare_pair(transform, mov, ref)
    shared = (
        tile_residuals(transform, mov, ref, VOX3, tile_px=64, pair=pair),
        z_residuals(transform, mov, ref, VOX3, crop_px=96, pair=pair),
    )
    assert len(warps) == 1
    assert shared == alone
    with pytest.raises(ValueError, match="prepared for 'edges'"):
        tile_residuals(transform, mov, ref, VOX3, compare="intensity", pair=pair)


# --- layer 4: one record per position and timepoint ---------------------------------------


def _clicked(points_ref, shift_zyx, voxel_um):
    """Clicked pairs: the same structures in both channels, the moving ones at p + shift."""
    from biahub.registration.evaluate import Landmarks

    ids = list(range(len(points_ref)))
    mov = Landmarks(np.asarray(points_ref) + shift_zyx, voxel_um, "manual", ids)
    ref = Landmarks(points_ref, voxel_um, "manual", ids)
    return mov, ref


def test_evaluate_reads_every_layer_against_the_unregistered_pair():
    import json

    from biahub.registration.evaluate import evaluate

    ref = _cells(6, shape=(32, 256, 256))
    shift_zyx = (2.0, 3.0, -2.0)
    mov = _ndi_shift(ref, shift_zyx, order=1, mode="nearest")
    exact = Transform.from_translation([-s for s in shift_zyx])
    grid = np.stack(np.meshgrid([8, 16, 24], [40, 128, 216], [40, 128, 216], indexing="ij"))
    landmarks = _clicked(grid.reshape(3, -1).T.astype(float), shift_zyx, VOX3)

    record = evaluate(exact, mov, ref, VOX3, VOX3, landmarks=landmarks, tile_px=64, crop_px=96)
    assert record.geometry["translation_um"] == pytest.approx([-0.4, -0.309, 0.206])
    assert record.points["stats"]["median_um"] < 1e-6
    assert record.tiles["stats"]["median_um"] < 0.05
    assert (record.z_from, record.z_um) == ("points (manual)", pytest.approx(0.0, abs=1e-6))
    # doing nothing leaves the full shift, in every measure
    shift_um = np.linalg.norm(np.multiply(shift_zyx, VOX3))
    assert record.baseline["points"]["stats"]["median_um"] == pytest.approx(shift_um)
    assert record.baseline["z_um"] == pytest.approx(-2 * 0.2)
    assert record.baseline["tiles"]["stats"]["median_um"] > 0.3
    json.dumps(record.to_dict())  # a record is plain data, ready to store


def test_without_landmarks_z_comes_from_the_crops_and_says_so():
    from biahub.registration.evaluate import evaluate

    ref = _cells(7, shape=(40, 256, 256))
    mov = _ndi_shift(ref, (2.0, 0.0, 0.0), order=1, mode="nearest")
    record = evaluate(Transform.identity(3), mov, ref, VOX3, VOX3, tile_px=64, crop_px=96)
    assert record.points is None
    assert record.z_from == "z crops (edges)"
    assert record.z_um == pytest.approx(-2 * 0.2, abs=0.05)


def test_evaluate_refuses_an_unknown_landmark_source():
    from biahub.registration.evaluate import evaluate

    ref = _cells(8)
    with pytest.raises(ValueError, match="landmarks"):
        evaluate(Transform.identity(3), ref, ref, VOX3, VOX3, landmarks="cells", images=False)


def test_an_identity_pair_of_equal_shapes_skips_the_warp(monkeypatch):
    # The baseline's identity warp returns the volume unchanged (~200 s on a 2048 x 2048
    # stack for nothing); a pair of equal shapes uses the volume as it is.
    from biahub.registration.evaluate import prepare_pair

    ref = _cells(9)
    mov = _ndi_shift(ref, (0.0, 2.0, 1.0), order=1, mode="nearest")
    warped = prepare_pair(Transform.identity(3), mov, ref).warped
    calls = []
    monkeypatch.setattr(Transform, "apply", lambda self, *a, **k: calls.append(1))
    pair = prepare_pair(Transform.identity(3), mov, ref)
    assert calls == []
    np.testing.assert_array_equal(pair.warped, warped)
    assert pair.has_data.all()


# --- layer 5: views ------------------------------------------------------------------------


def test_records_keep_each_residual_where_it_was_measured():
    # The overlays draw each residual at its place, so the records keep them (in µm, on
    # the reference grid), not only their statistics.
    from biahub.registration.evaluate import point_residuals

    shift_zyx = (0.0, 3.0, -2.0)
    grid = np.stack(np.meshgrid([8, 16], [40, 128, 216], [40, 128, 216], indexing="ij"))
    mov_marks, ref_marks = _clicked(grid.reshape(3, -1).T.astype(float), shift_zyx, VOX3)
    points = point_residuals(Transform.identity(3), mov_marks, ref_marks)
    um = np.asarray(VOX3)
    np.testing.assert_allclose(points["residuals"]["position_um"], ref_marks.points_vox * um)
    np.testing.assert_allclose(
        points["residuals"]["diff_um"], -np.multiply(shift_zyx, um)[None].repeat(18, 0)
    )

    ref = _cells()
    mov = _ndi_shift(ref, shift_zyx, order=1, mode="nearest")
    tiles = tile_residuals(Transform.identity(3), mov, ref, VOX3, tile_px=64)
    assert len(tiles["residuals"]["position_um"]) == tiles["n_tiles"]
    np.testing.assert_allclose(
        np.median(tiles["residuals"]["diff_um"], axis=0), tiles["stats"]["median_zyx_um"]
    )


def test_evaluate_keeps_small_projections_for_the_views_but_does_not_store_them():
    from biahub.registration.evaluate import evaluate

    ref = _cells(10, shape=(24, 600, 600))
    shift_zyx = (0.0, 6.0, -4.0)
    mov = _ndi_shift(ref, shift_zyx, order=1, mode="nearest")
    exact = Transform.from_translation([-s for s in shift_zyx])
    record = evaluate(exact, mov, ref, VOX3, VOX3, images=False)

    views = record.projections
    assert max(views["reference"].shape) <= 512 and views["downsample"] == 2
    assert views["reference"].shape == views["registered"].shape == views["unregistered"].shape

    def corr(a, b):
        inner = (slice(8, -8), slice(8, -8))  # away from the shifted-in border
        return np.corrcoef(a[inner].ravel(), b[inner].ravel())[0, 1]

    assert corr(views["registered"], views["reference"]) > 0.99
    assert corr(views["unregistered"], views["reference"]) < corr(
        views["registered"], views["reference"]
    )
    assert "projections" not in record.to_dict()


def _record(images=True, landmarks=True):
    from biahub.registration.evaluate import evaluate

    ref = _cells(11, shape=(24, 256, 256))
    shift_zyx = (1.0, 3.0, -2.0)
    mov = _ndi_shift(ref, shift_zyx, order=1, mode="nearest")
    grid = np.stack(np.meshgrid([8, 16], [40, 128, 216], [40, 128, 216], indexing="ij"))
    marks = (
        _clicked(grid.reshape(3, -1).T.astype(float), shift_zyx, VOX3) if landmarks else None
    )
    exact = Transform.from_translation([-s for s in shift_zyx])
    return evaluate(
        exact, mov, ref, VOX3, VOX3, landmarks=marks, images=images, tile_px=64, crop_px=96
    )


@pytest.mark.parametrize("images, landmarks", [(True, True), (True, False), (False, True)])
def test_the_evaluation_figure_shows_both_pairs_and_the_z_residual(
    tmp_path, images, landmarks
):
    from biahub.registration.evaluate_view import plot_evaluation

    path = tmp_path / "evaluation.png"
    figure = plot_evaluation(_record(images, landmarks), path=path, title="A/1/0 t=0")
    titles = [ax.get_title() for ax in figure.axes]
    assert any("unregistered" in t for t in titles) and any("registered" in t for t in titles)
    assert any("z" in t for t in titles)
    assert path.stat().st_size > 10_000  # an image was written


def test_a_stored_record_has_no_projections_to_plot():
    from biahub.registration.evaluate import Evaluation
    from biahub.registration.evaluate_view import plot_evaluation

    stored = Evaluation(**_record(images=False).to_dict())
    with pytest.raises(ValueError, match="projections"):
        plot_evaluation(stored)


def test_the_video_holds_the_projections_then_scans_through_z(tmp_path):
    import imageio_ffmpeg

    from biahub.registration.evaluate_view import HOLD_FRAMES, animate_evaluation

    ref = _cells(12, shape=(20, 160, 160))
    mov = _ndi_shift(ref, (0.0, 3.0, -2.0), order=1, mode="nearest")
    transforms = {
        "unregistered": Transform.identity(3),
        "registered": Transform.from_translation([0.0, -3.0, 2.0]),
    }
    path = animate_evaluation(mov, ref, transforms, tmp_path / "check.mp4", zoom_px=64)
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    n_frames = sum(1 for _ in reader)
    assert n_frames == HOLD_FRAMES + ref.shape[0]
    assert meta["size"][0] % 16 == 0 and meta["size"][1] % 16 == 0  # as the encoder wants


def test_the_zoom_warps_only_its_corner_and_matches_the_full_warp():
    from biahub.registration.evaluate_view import _corner_pair

    ref = _cells(13, shape=(20, 160, 160))
    mov = _ndi_shift(ref, (0.0, 3.0, -2.0), order=1, mode="nearest")
    transform = Transform.from_translation([0.0, -3.0, 2.0])
    warped, ref_crop = _corner_pair(transform, mov, ref, size=64)
    full = transform.apply(mov, reference=ref)
    np.testing.assert_allclose(warped, full[:, :64, :64], atol=1e-3)
    np.testing.assert_array_equal(ref_crop, ref[:, :64, :64])


# --- layer 6: the visual check in napari -------------------------------------------------


@pytest.mark.parametrize("with_phase", [False, True])
def test_the_visual_check_shows_the_reference_and_the_moving_volume_before_and_after(
    monkeypatch, with_phase
):
    # Kept simple on purpose: the person looks, closes the window, and the skill asks
    # whether it is good.
    pytest.importorskip("napari")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from biahub.registration.evaluate_view import _build_check

    ref = _cells(14)
    mov = _ndi_shift(ref, (0.0, 3.0, -2.0), order=1, mode="nearest")
    phase = ref.max() - ref if with_phase else None
    transform = Transform.from_translation([0.0, -3.0, 2.0])
    viewer = _build_check(mov, ref, transform, VOX3, phase=phase, show=False)
    try:
        names = [layer.name for layer in viewer.layers]
        expected = ["reference", "moving (original)", "moving (registered)"]
        assert names == (["phase"] + expected if with_phase else expected)
        assert viewer.layers["moving (registered)"].visible
        assert not viewer.layers["moving (original)"].visible
        if with_phase:  # auto contrast: its own percentiles, not min to max
            shown = viewer.layers["phase"]
            low, high = np.percentile(shown.data, [0.5, 99.5])
            assert shown.contrast_limits == pytest.approx([low, high])
    finally:
        viewer.close()


def test_z_from_too_few_points_falls_back_to_the_crops_and_says_why():
    # One matched nucleus gave z = -1.98 µm on a sparse DragonFly FOV: noise, not a measure.
    from biahub.registration.evaluate import evaluate

    ref = _cells(16, shape=(40, 256, 256))
    mov = _ndi_shift(ref, (2.0, 0.0, 0.0), order=1, mode="nearest")
    few = _clicked(np.array([[20.0, 60.0, 60.0], [20.0, 190.0, 190.0]]), (2.0, 0.0, 0.0), VOX3)
    record = evaluate(
        Transform.identity(3), mov, ref, VOX3, VOX3, landmarks=few, tile_px=64, crop_px=96
    )
    assert record.points["n_matched"] == 2
    assert record.z_from == "z crops (edges); 2 matched points are too few"
    assert record.z_um == pytest.approx(-2 * 0.2, abs=0.05)


# --- review fixes ----------------------------------------------------------------------


def test_a_plain_shift_on_points_at_one_depth_is_not_read_as_scale():
    # Tile positions all sit at one z: the intercept and the z column were the same
    # direction, and the fit moved part of a shift into a z gradient (a false scale).
    yx = np.stack(np.meshgrid(np.linspace(100, 1900, 7), np.linspace(100, 1900, 7)), -1)
    points = np.c_[np.full(49, 33.0), yx.reshape(-1, 2)]
    shift = np.array([0.0, 2.5, -1.5])
    r = residual_affine(points, np.tile(shift, (49, 1)), VOX)
    np.testing.assert_allclose(r["scale_zyx"], [1.0, 1.0, 1.0], atol=1e-9)
    assert r["rotation_yx_deg"] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(r["shift_um"], shift * VOX, atol=1e-9)


@pytest.mark.parametrize("tile_px, shape", [(63, (24, 256, 256)), (256, (24, 160, 160))])
def test_tiles_fit_the_image_whatever_their_size(tile_px, shape):
    ref = _cells(17, shape=shape)
    mov = _ndi_shift(ref, (0.0, 2.0, -1.0), order=1, mode="nearest")
    r = tile_residuals(Transform.identity(3), mov, ref, VOX3, tile_px=tile_px, grid=3)
    assert r["n_tiles"] > 0


def test_a_nucleus_with_a_flat_depth_profile_is_skipped_not_a_nan():
    vol = np.zeros((12, 200, 200), np.float32)
    yy, xx = np.ogrid[:200, :200]
    disk = lambda cy, cx: ((yy - cy) ** 2 + (xx - cx) ** 2) <= 30**2  # noqa: E731
    vol[:, disk(60, 60)] = 100.0  # the same in every slice: no depth to measure
    profile = np.exp(-0.5 * ((np.arange(12) - 6) / 2.0) ** 2)[:, None]
    vol[:, disk(140, 140)] += 100.0 * profile
    marks = detect_nuclei(vol, VOX3)
    assert np.isfinite(marks.points_vox).all()
    assert len(marks.points_vox) == 1


def test_arrows_use_each_axis_voxel_size():
    from biahub.registration.evaluate_view import _arrows

    views = {"ref_um": (0.2, 0.1, 0.2), "downsample": 1}
    measure = {"residuals": {"position_um": [[1.0, 5.0, 10.0]], "diff_um": [[0.0, 0.1, 0.2]]}}
    _where, step_px = _arrows(measure, views)
    np.testing.assert_allclose(step_px, [[1.0, 1.0]])  # one voxel in y and one in x


def _disk_nuclei(n=20, seed=1, shape=(16, 768, 768)):
    """Flat nucleus-sized disks (radius 30 px), as DAPI or VS nuclei."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, np.float32)
    yy, xx = np.ogrid[: shape[1], : shape[2]]
    for cy, cx in rng.uniform(40, shape[1] - 40, (n, 2)):
        vol[4:12][:, ((yy - cy) ** 2 + (xx - cx) ** 2) <= 30**2] += 100
    noise = rng.normal(0, 1, shape).astype(np.float32)
    return gaussian_filter(vol, (1, 3, 3)) + noise + 10


def test_tiles_recover_a_known_shift_in_full():
    # The window both tiles are multiplied by pulls the correlation peak toward zero shift
    # (95% of the shift on nucleus-like disks); a second pass on the remainder removes it.
    ref = _disk_nuclei()
    shift_px = 6.0
    mov = _ndi_shift(ref, (0.0, shift_px, -shift_px), order=1, mode="nearest")
    r = tile_residuals(Transform.identity(3), mov, ref, VOX3, tile_px=256, grid=5)
    recovered = -np.median(np.asarray(r["residuals"]["diff_um"])[:, 1:], axis=0) / VOX3[1]
    np.testing.assert_allclose(recovered, [shift_px, -shift_px], rtol=0.01)


def test_the_residual_affine_is_the_map_left_on_the_mapped_points():
    # Clicked pairs with the moving points at 0.8 x the reference, under identity: what is
    # left is a scale of 1.25 acting on the mapped points (fitting at the reference
    # positions read 1.2), and a pure rotation leaves no scale.
    from biahub.registration.evaluate import point_residuals

    grid = np.stack(
        np.meshgrid([10.0, 30.0], np.linspace(100, 1900, 5), np.linspace(100, 1900, 5))
    )
    ref_pts = grid.reshape(3, -1).T
    centre = np.array([20.0, 1000.0, 1000.0])
    ids = list(range(len(ref_pts)))
    for moving, scale, angle in (
        (centre + 0.8 * (ref_pts - centre), 1.25, 0.0),
        (
            _similarity((1.0, 1.0, 1.0), -2.0, (0, 0, 0)).apply_points(ref_pts - centre)
            + centre,
            1.0,
            2.0,
        ),
    ):
        mov = Landmarks(moving, VOX, "manual", ids)
        ref = Landmarks(ref_pts, VOX, "manual", ids)
        affine = point_residuals(Transform.identity(3), mov, ref)["affine"]
        np.testing.assert_allclose(affine["scale_zyx"][1:], [scale, scale], atol=1e-6)
        assert affine["rotation_yx_deg"] == pytest.approx(angle, abs=1e-6)


def test_one_tile_that_fits_the_image_counts_once():
    # A 160 x 160 field with 256 px tiles is one tile: measuring it 49 times passed the
    # minimum for a residual affine with a single place measured.
    ref = _cells(19, shape=(24, 160, 160))
    mov = _ndi_shift(ref, (0.0, 2.0, -1.0), order=1, mode="nearest")
    r = tile_residuals(Transform.identity(3), mov, ref, VOX3, tile_px=256, grid=7)
    assert r["n_tiles"] == 1
    assert r["affine"]["scale_zyx"] is None


def test_the_shift_limit_holds_for_the_refined_shift():
    # The limit was checked before the second pass, which lengthens the shift: no tile
    # kept may be further off than the limit.
    ref = _disk_nuclei()
    mov = _ndi_shift(ref, (0.0, 6.0, -6.0), order=1, mode="nearest")
    limit = 5.9
    r = tile_residuals(
        Transform.identity(3), mov, ref, VOX3, tile_px=256, grid=5, max_shift_px=limit
    )
    kept_px = np.abs(np.asarray(r["residuals"]["diff_um"]).reshape(-1, 3)[:, 1:]) / VOX3[1]
    assert kept_px.max(initial=0.0) <= limit + 1e-9


def test_corner_crops_of_the_same_place_give_no_tilt():
    # When the image is no bigger than a crop, every corner is the whole volume: the z
    # spread across them is not a tilt measured.
    ref = _cells(20, shape=(40, 256, 256))
    mov = _ndi_shift(ref, (2.0, 0.0, 0.0), order=1, mode="nearest")
    r = z_residuals(Transform.identity(3), mov, ref, VOX3, crop_px=768)
    assert r["tilt_dz_um"] is None
