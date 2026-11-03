"""biahub.divisions: features of called divisions, a classifier, and cutting rejected divisions."""

import numpy as np
import pandas as pd


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def _dividing_movie(T=8, size=64):
    """Track 1 for 4 frames, then daughters 2 and 3; labels painted with track ids."""
    labels = np.zeros((T, size, size), dtype=np.int32)
    rows = []
    for t in range(4):
        labels[t][_disk((size, size), 32, 32, 8)] = 1
        rows.append((1, t, 32.0, 32.0, -1))
    for t in range(4, T):
        d = 9 + t - 4  # daughters 12 px wide, centres >= 18 px apart: no overlap
        labels[t][_disk((size, size), 32, 32 - d, 6)] = 2
        labels[t][_disk((size, size), 32, 32 + d, 6)] = 3
        rows += [(2, t, 32.0, 32.0 - d, 1), (3, t, 32.0, 32.0 + d, 1)]
    tracks = pd.DataFrame(rows, columns=["track_id", "t", "y", "x", "parent_track_id"])
    nuc = (labels > 0).astype(np.float32)
    return labels, tracks, nuc


def test_division_events_lists_each_called_division():
    from biahub.divisions import division_events

    _, tracks, _ = _dividing_movie()
    ev = division_events(tracks)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert (e.parent, {e.d1, e.d2}, e.t) == (1, {2, 3}, 4)


def test_division_features_are_in_physical_units_and_scale_free():
    from biahub.divisions import FEATURES, division_features

    labels, tracks, nuc = _dividing_movie()
    texture = np.ones_like(nuc)
    fg = np.ones_like(nuc)
    a = division_features(labels, tracks, nuc, texture, fg, pixel_size=0.1)
    b = division_features(labels, tracks, nuc, texture, fg, pixel_size=0.2)
    assert list(a.columns[: len(FEATURES)]) == FEATURES and len(a) == 1
    assert "t_frac" not in a.columns  # time in the movie is a confounder, not biology
    assert a.daughter_area_ratio.iloc[0] == 1.0  # equal daughters
    # distances in um scale with the pixel size; ratios do not
    assert b.daughter_dist_t0.iloc[0] == 2 * a.daughter_dist_t0.iloc[0]
    assert b.daughter_area_ratio.iloc[0] == a.daughter_area_ratio.iloc[0]


def test_classifier_scores_divisions():
    from biahub.divisions import FEATURES, fit_division_classifier, score_divisions

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, len(FEATURES))), columns=FEATURES)
    y = X["daughter_area_ratio"] > 0  # a learnable rule
    model = fit_division_classifier(X, y)
    p = score_divisions(model, X)
    assert p.shape == (80,) and ((p >= 0) & (p <= 1)).all()
    assert (p[y.to_numpy()] > 0.5).mean() > 0.9


def test_cut_divisions_makes_the_daughters_roots():
    from biahub.divisions import cut_divisions

    _, tracks, _ = _dividing_movie()
    cut = cut_divisions(tracks, rejected_parents={1})
    assert (cut.parent_track_id == -1).all()
    assert cut_divisions(tracks, rejected_parents=set()).equals(tracks)


def test_phase_texture_and_foreground():
    from biahub.divisions import phase_texture_and_foreground

    zyx = np.zeros((10, 64, 64), dtype=np.float32)
    zyx[:, 16:48, 16:48] = np.random.default_rng(0).normal(size=(10, 32, 32))  # textured cell
    texture, fg = phase_texture_and_foreground(zyx, min_size_px=100)
    assert texture.shape == fg.shape == (64, 64)
    assert fg[32, 32] and not fg[2, 2]
