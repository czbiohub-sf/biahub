"""Score the divisions a tracker calls and cut the ones judged false (draft; see issue #447).

A division is a mother track with two daughter tracks. Each called division is described by
features of the mother before the split and of the daughters at the split, taken from the
tracked labels (value = track id), the virtual-stained nuclei and the phase texture (log of the
per-pixel standard deviation of Phase3D over the focus window). A random forest trained on
annotated divisions (real vs false) gives each division a score; divisions scoring below a
threshold are cut: their daughters become new tracks.

On hold until more tracking annotations exist: the prototype was trained on 4 annotated FOVs.
Not wired into ``biahub track`` yet (where it runs is an open decision in #447).
"""

import numpy as np
import pandas as pd

# Features, in the order the model expects. Areas enter only as ratios and distances are in
# um, so the features do not depend on the pixel size. "Time in the movie" was dropped: on the
# prototype it ranked fifth, which reflects when these movies divide, not what a division is.
FEATURES = [
    "mother_area_change",  # mother area just before the split / 3-5 frames earlier (rounding)
    "mother_vs_change",  # same ratio for the mother's VS brightness
    "mother_texture_change",  # rise of the mother's phase texture before the split
    "mother_track_len",  # frames
    "daughter_area_ratio",  # smaller / larger daughter
    "daughters_vs_mother_area",  # daughters' total area / mother's area before the split
    "daughter_vs_ratio",  # dimmer / brighter daughter (VS)
    "daughter_texture_min",  # phase texture of the less textured daughter
    "daughter_phase_support_min",  # fraction of the daughter on phase foreground (min of two)
    "daughter_life_min",  # frames
    "daughter_life_max",  # frames
    "daughter_dist_t0",  # um between the daughters at the split
    "daughter_dist_change",  # um the daughters separate over the next 2 frames
]


def division_events(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per called division (mothers with exactly two daughters).

    Columns: ``t`` (daughters' first frame), ``y``/``x`` (mother's last position),
    ``parent``, ``d1``, ``d2``.
    """
    first = tracks.groupby("track_id").agg(
        t0=("t", "min"), parent=("parent_track_id", "first")
    )
    last = tracks.sort_values("t").groupby("track_id").last()
    rows = [
        {
            "t": int(d.t0.min()),
            "y": float(last.loc[p, "y"]),
            "x": float(last.loc[p, "x"]),
            "parent": int(p),
            "d1": int(d.index[0]),
            "d2": int(d.index[1]),
        }
        for p, d in first[first.parent > 0].groupby("parent")
        if len(d) == 2 and p in last.index
    ]
    return pd.DataFrame(rows, columns=["t", "y", "x", "parent", "d1", "d2"])


def _stats(labels_t: np.ndarray, track_id: int, *images: np.ndarray) -> list[float]:
    """Area (px) and the mean of each image over one object; NaN if the object is absent."""
    mask = labels_t == track_id
    if not mask.any():
        return [np.nan] * (1 + len(images))
    return [float(mask.sum())] + [float(img[mask].mean()) for img in images]


def _ratio(a: float, b: float) -> float:
    return a / b if b and np.isfinite(b) else np.nan


def division_features(
    labels: np.ndarray,
    tracks: pd.DataFrame,
    nuclei: np.ndarray,
    texture: np.ndarray,
    foreground: np.ndarray,
    pixel_size: float,
) -> pd.DataFrame:
    """Features (:data:`FEATURES`) of every called division, plus its ``t, y, x, parent``.

    ``labels``, ``nuclei``, ``texture`` and ``foreground`` are (T, Y, X): the tracked labels,
    the VS nuclei, the phase texture and the phase foreground mask (0/1). ``pixel_size`` in um.
    """
    events = division_events(tracks)
    life = tracks.groupby("track_id").size()
    pos = tracks.set_index(["track_id", "t"])[["y", "x"]]
    T = labels.shape[0]
    rows = []
    for e in events.itertuples():
        t0, parent, da, db = e.t, e.parent, e.d1, e.d2
        m_now = (
            _stats(labels[t0 - 1], parent, nuclei[t0 - 1], texture[t0 - 1])
            if t0 >= 1
            else [np.nan] * 3
        )
        early = [
            _stats(labels[t], parent, nuclei[t], texture[t])
            for t in range(max(0, t0 - 5), max(0, t0 - 2))
        ]
        m_early = np.nanmean(np.array(early), axis=0) if early else np.full(3, np.nan)
        d1 = _stats(labels[t0], da, nuclei[t0], texture[t0], foreground[t0])
        d2 = _stats(labels[t0], db, nuclei[t0], texture[t0], foreground[t0])

        def dist(t, da=da, db=db):
            try:
                a, b = pos.loc[(da, t)], pos.loc[(db, t)]
            except KeyError:
                return np.nan
            return float(np.hypot(a.y - b.y, a.x - b.x)) * pixel_size

        small, big = sorted([d1[0], d2[0]])
        rows.append(
            {
                "mother_area_change": _ratio(m_now[0], m_early[0]),
                "mother_vs_change": _ratio(m_now[1], m_early[1]),
                "mother_texture_change": m_now[2] - m_early[2],
                "mother_track_len": float(life.get(parent, 0)),
                "daughter_area_ratio": _ratio(small, big),
                "daughters_vs_mother_area": _ratio(d1[0] + d2[0], m_now[0]),
                "daughter_vs_ratio": _ratio(min(d1[1], d2[1]), max(d1[1], d2[1])),
                "daughter_texture_min": min(d1[2], d2[2]),
                "daughter_phase_support_min": min(d1[3], d2[3]),
                "daughter_life_min": float(min(life.get(da, 0), life.get(db, 0))),
                "daughter_life_max": float(max(life.get(da, 0), life.get(db, 0))),
                "daughter_dist_t0": dist(t0),
                "daughter_dist_change": dist(min(t0 + 2, T - 1)) - dist(t0),
                "t": t0,
                "y": e.y,
                "x": e.x,
                "parent": parent,
            }
        )
    return pd.DataFrame(rows, columns=[*FEATURES, "t", "y", "x", "parent"])


def fit_division_classifier(features: pd.DataFrame, is_real, random_state: int = 0):
    """Random forest on :data:`FEATURES` (missing values -> training medians).

    The settings are the prototype's: 500 trees, min 2 samples per leaf, balanced classes.
    """
    from sklearn.ensemble import RandomForestClassifier

    X = features[FEATURES]
    medians = X.median()
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
    ).fit(X.fillna(medians), np.asarray(is_real, dtype=bool))
    model.feature_medians_ = medians
    return model


def score_divisions(model, features: pd.DataFrame) -> np.ndarray:
    """Probability that each division is real."""
    if len(features) == 0:
        return np.zeros(0)
    X = features[FEATURES].fillna(model.feature_medians_)
    return model.predict_proba(X)[:, list(model.classes_).index(True)]


def cut_divisions(tracks: pd.DataFrame, rejected_parents: set[int]) -> pd.DataFrame:
    """Unlink the daughters of every rejected division: they become root tracks (parent -1).

    The labels need no change: tracks keep their ids.
    """
    if not rejected_parents:
        return tracks
    tracks = tracks.copy()
    tracks.loc[tracks.parent_track_id.isin(list(rejected_parents)), "parent_track_id"] = -1
    return tracks


def phase_texture_and_foreground(
    zyx: np.ndarray, close_radius: int = 10, min_size_px: int = 8000
) -> tuple[np.ndarray, np.ndarray]:
    """Phase texture and foreground of one timepoint from a Phase3D (Z, Y, X) window.

    Texture: log1p of the per-pixel standard deviation over Z (minus its minimum). Foreground:
    Otsu threshold of the texture, closed with a disk, components below ``min_size_px``
    dropped, holes filled (as shrimPy's FOV selection). ``min_size_px`` 8000 is for
    0.1133 um/px; scale it with the pixel size.
    """
    from scipy.ndimage import binary_fill_holes
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    from skimage.morphology import binary_closing, disk

    std = np.asarray(zyx, dtype=np.float32).std(axis=0)
    texture = np.log1p(std - std.min()).astype(np.float32)
    if texture.min() == texture.max():
        return texture, np.zeros(texture.shape, dtype=bool)
    fg = binary_closing(texture > threshold_otsu(texture), disk(close_radius))
    lab = label(fg)
    keep = np.bincount(lab.ravel()) >= min_size_px
    keep[0] = False
    return texture, binary_fill_holes(keep[lab])
