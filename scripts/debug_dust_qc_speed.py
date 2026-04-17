"""Fast dust QC benchmark: compare approaches that all use Gaussian blur."""

import time
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, label
from iohub.ngff import open_ome_zarr
from dynacell_geometry import make_circular_mask
import matplotlib.pyplot as plt

# --- Config ---
root_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
dataset = "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
lf_zarr = (
    root_path / dataset / "1-preprocess" / "label-free" / "0-reconstruct" / f"{dataset}.zarr"
)
lf_mask_radius = 0.98
test_z_planes = [0, 25, 50, 75, 104]


def score_dust(residual: np.ndarray, circ_mask: np.ndarray, n_std: float = 3.0) -> dict:
    valid = residual[circ_mask]
    sigma = float(np.std(valid))
    threshold = n_std * sigma
    dust_mask = (np.abs(residual) > threshold) & circ_mask
    dust_fraction = float(dust_mask.sum()) / circ_mask.sum()
    _, n_spots = label(dust_mask)
    return {
        "dust_fraction": dust_fraction,
        "dust_score": min(1.0, dust_fraction * 100),
        "n_spots": n_spots,
    }


def detect_dust(frames: list[np.ndarray], circ_mask: np.ndarray,
                agg: str = "median", bg_method: str = "subtract",
                sigma_bg: float = 30.0) -> dict:
    """Generic fast dust detection.

    agg: "median" or "mean" to aggregate frames.
    bg_method: "subtract" (additive) or "divide" (flat-field).
    """
    t0 = time.time()
    stack = np.stack(frames, axis=0)
    static = np.median(stack, axis=0) if agg == "median" else np.mean(stack, axis=0)

    smooth = gaussian_filter(static, sigma=sigma_bg)

    if bg_method == "divide":
        smooth[smooth == 0] = 1.0
        residual = (static / smooth - 1.0) * smooth.mean()
    else:
        residual = static - smooth

    residual[~circ_mask] = 0.0
    elapsed = time.time() - t0

    result = score_dust(residual, circ_mask)
    result["time_s"] = elapsed
    result["static_img"] = static
    result["residual"] = residual
    return result


if __name__ == "__main__":
    # Discover FOVs
    from glob import glob as gglob
    all_fov_paths = sorted(gglob(str(lf_zarr / "*" / "*" / "*")))
    all_fov_keys = ["/".join(Path(p).parts[-3:]) for p in all_fov_paths]
    print(f"Found {len(all_fov_keys)} FOVs")

    # Read shape
    with open_ome_zarr(lf_zarr / all_fov_keys[0]) as ds:
        T, C, Z, Y, X = ds.data.shape
    print(f"Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    circ_mask = make_circular_mask(Y, X, lf_mask_radius)

    # --- Read frames ---
    # Strategy A: 3 timepoints from 1 FOV (fastest read)
    fov_1 = all_fov_keys[0]
    t_idx = [0, T // 2, T - 1]
    print(f"\nReading 3 frames from 1 FOV ({fov_1})...")
    t0 = time.time()
    frames_1fov = {}
    with open_ome_zarr(lf_zarr / fov_1) as ds:
        arr = ds.data.dask_array()
        for z in test_z_planes:
            frames_1fov[z] = [np.asarray(arr[t, 0, z]).astype(np.float32) for t in t_idx]
    print(f"  {time.time() - t0:.1f}s")

    # Strategy B: 1 timepoint from 6 FOVs (better sample diversity)
    n_multi = 6
    fov_indices = np.linspace(0, len(all_fov_keys) - 1, n_multi, dtype=int)
    fov_multi = [all_fov_keys[i] for i in fov_indices]
    t_mid = T // 2
    print(f"Reading 1 frame from {n_multi} FOVs ({fov_multi})...")
    t0 = time.time()
    frames_multi = {z: [] for z in test_z_planes}
    for fk in fov_multi:
        with open_ome_zarr(lf_zarr / fk) as ds:
            arr = ds.data.dask_array()
            for z in test_z_planes:
                frames_multi[z].append(np.asarray(arr[t_mid, 0, z]).astype(np.float32))
    print(f"  {time.time() - t0:.1f}s")

    # --- Test all combinations ---
    configs = [
        ("1 FOV, 3t, median, subtract", lambda z: detect_dust(frames_1fov[z], circ_mask, "median", "subtract")),
        ("1 FOV, 3t, mean, subtract",   lambda z: detect_dust(frames_1fov[z], circ_mask, "mean", "subtract")),
        ("1 FOV, 3t, median, divide",   lambda z: detect_dust(frames_1fov[z], circ_mask, "median", "divide")),
        ("6 FOV, 1t, median, subtract", lambda z: detect_dust(frames_multi[z], circ_mask, "median", "subtract")),
        ("6 FOV, 1t, mean, subtract",   lambda z: detect_dust(frames_multi[z], circ_mask, "mean", "subtract")),
        ("6 FOV, 1t, median, divide",   lambda z: detect_dust(frames_multi[z], circ_mask, "median", "divide")),
    ]

    print("\n" + "=" * 80)
    all_results = {z: [] for z in test_z_planes}

    for z in test_z_planes:
        print(f"\n--- Z={z} ---")
        for name, fn in configs:
            r = fn(z)
            r["method"] = name
            print(f"  {name:35s}  {r['time_s']:.3f}s  score={r['dust_score']:.4f}  "
                  f"frac={r['dust_fraction']:.6f}  spots={r['n_spots']}")
            all_results[z].append(r)

    # --- Speed summary ---
    print("\n" + "=" * 80)
    print("SPEED (estimated for all 105 Z planes, processing only):")
    for i, (name, _) in enumerate(configs):
        times = [all_results[z][i]["time_s"] for z in test_z_planes]
        print(f"  {name:35s}  {np.mean(times)*105:.1f}s total")

    # --- Plot comparison for Z with highest dust ---
    scores_z = {z: max(r["dust_score"] for r in all_results[z]) for z in test_z_planes}
    worst_z = max(scores_z, key=scores_z.get)
    n_methods = len(configs)

    fig, axes = plt.subplots(2, n_methods, figsize=(4 * n_methods, 8))
    for i, r in enumerate(all_results[worst_z]):
        im = r["static_img"]
        vmin, vmax = np.percentile(im[circ_mask], [1, 99])
        axes[0, i].imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, i].set_title(r["method"], fontsize=9)
        axes[0, i].axis("off")

        res = r["residual"]
        vlim = np.percentile(np.abs(res[circ_mask]), 99)
        axes[1, i].imshow(res, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        axes[1, i].set_title(f"score={r['dust_score']:.4f} spots={r['n_spots']}", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle(f"Dust QC comparison — Z={worst_z}", fontsize=13)
    plt.tight_layout()
    out = Path("/hpc/mydata/taylla.theodoro/repo/biahub_dev/scripts/dust_qc_benchmark.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
