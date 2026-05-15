
#%%
"""
Z-stabilization evaluation script.

- Sign convention test: skimage vs biahub PCC × ANTs
- Chained drift synthetic toy test
- Realistic toy test: t=0 real phase + simulated drift, PCC with 3 inputs
- PCC on real data: pairwise-cumulative vs anchor-t0 × 3 inputs
- Saves one zarr per (strategy × input) combination → 6 zarrs total
- focus_from_transverse_band before and after, ZT kymographs
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ants
from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from waveorder.focus import focus_from_transverse_band
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation as pcc_skimage
from scipy.ndimage import shift as ndimage_shift

from biahub.core.transform import Transform

try:
    from biahub.registration.phase_cross_correlation import phase_cross_correlation as pcc_biahub
except ModuleNotFoundError:
    pcc_biahub = None  # only used in old notebook cells

# ── Old interactive notebook cells below ──
# Wrapped in a guard so they only run when executed cell-by-cell in an IDE,
# not when running the script from the command line.
if __name__ != "__main__":
    # ─────────────────────────────────────────────
    # CONFIG
    # ─────────────────────────────────────────────
    NA_DET     = 1.35
    LAMBDA_ILL = 0.500
    N_FRAMES   = 20
    
    dataset = "2024_12_03_A549_LAMP1_ZIKV_DENV"
    FOV     = "B/2/000001"
    root_path = Path(f"/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}/")
    phase_zarr_path = Path(
        f"{root_path}/1-preprocess/label-free/0-reconstruct/{dataset}.zarr/{FOV}"
    )
    vs_zarr_path = Path(
        f"/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}"
        f"/1-preprocess/label-free/1-virtual-stain/{dataset}.zarr/{FOV}"
    )
    phase_channel_name = "Phase3D"
    vs_channel_name    = "nuclei_prediction"
    
    output_path = Path(
        f"/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}/debug_stab/pcc"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    #%%
    # Options: "pairwise", "anchor", or "both"
    STRATEGY_MODE = "pairwise"
    STRATEGIES = ["pairwise", "anchor"] if STRATEGY_MODE == "both" else [STRATEGY_MODE]
    INPUT_NAMES = ["masked_phase"] #mask, phase
    CMAPS       = {"phase": "gray", "mask": "Reds", "masked_phase": "gray"}
    SIGN = {"skimage": +1, "biahub": -1}#%%
    
    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────
    def section(title):
        print(f"\n{'='*60}\n{title}\n{'='*60}")
    
    
    def ants_apply_shift(vol_zyx: np.ndarray, shift_zyx: np.ndarray, interpolation: str = "linear") -> np.ndarray:
        """
        Apply a 3D translation using ANTs via biahub Transform.
        ANTs pull-back: output(x) = input(x + t) → negate to get desired content shift.
        shift_zyx : desired content displacement (dz, dy, dx) in pixels.
        interpolation : "linear" for images, "genericlabel" for masks/labels.
        """
        M = np.eye(4)
        M[0, 3] = -shift_zyx[0]
        M[1, 3] = -shift_zyx[1]
        M[2, 3] = -shift_zyx[2]
        transform = Transform(matrix=M)
        ants_img  = ants.from_numpy(vol_zyx.astype(np.float32))
        return transform.to_ants().apply_to_image(ants_img, interpolation=interpolation).numpy()
    
    
    def run_pcc_biahub(ref, mov):
        """Returns sign-corrected shift: positive = content moved toward higher indices."""
        raw = np.array(pcc_biahub(ref, mov, normalization=None, verbose=False), dtype=float)
        return SIGN["biahub"] * raw
    
    
    
    #%%
    # ─────────────────────────────────────────────
    # LOAD REAL DATA
    # ─────────────────────────────────────────────
    section("LOADING DATA")
    
    with open_ome_zarr(phase_zarr_path) as ds:
        ch_idx                   = ds.channel_names.index(phase_channel_name)
        _, _, _, _, pixel_size   = ds.scale
        phase_tzyx               = np.asarray(ds.data[:N_FRAMES, ch_idx])
        ds_scale                 = ds.scale
        _, _, Z, Y, X = ds.data.shape
        print(f"  Phase : {phase_tzyx.shape}  pixel_size={pixel_size:.4f} µm")
    
    with open_ome_zarr(vs_zarr_path) as ds:
        ch_idx  = ds.channel_names.index(vs_channel_name)
        vs_tzyx = np.asarray(ds.data[:N_FRAMES, ch_idx])
        print(f"  VS    : {vs_tzyx.shape}")
    
    vs_mask = np.zeros_like(vs_tzyx, dtype=bool)
    for t in range(N_FRAMES):
        vs_mask[t] = vs_tzyx[t] > threshold_otsu(vs_tzyx[t])
        print(f"  t={t:02d}  mask coverage: {vs_mask[t].mean()*100:.1f}%")
    
    
    #%%
    
    pcc_shifts = [np.array([0.0, 0.0, 0.0])]
    cumulative_shift = np.array([0.0, 0.0, 0.0])
    for t in range(1, N_FRAMES):
        pcc_shift = run_pcc_biahub(phase_tzyx[t-1]*vs_mask[t-1], phase_tzyx[t]*vs_mask[t])
        cumulative_shift += np.array(pcc_shift)
        pcc_shifts.append(cumulative_shift.copy())
        print(f"  t={t:02d}  pcc_shift={pcc_shifts[t]}")
    
    corrected_phase = np.empty_like(phase_tzyx)
    
    output_zarr_path = output_path / f"{dataset}_corrected_phase_1.zarr"
    output_metadata = {
        "shape": (N_FRAMES, 1, Z, Y, X), # (T, C, Z, Y, X)
        "chunks": None,
        "scale": ds_scale,
        "channel_names": [phase_channel_name],
        "dtype": np.float32,
    }
    
    create_empty_plate(
            store_path=output_zarr_path,
            position_keys=[("B", "2", "000001")],
            **output_metadata,
    )
    with open_ome_zarr(output_zarr_path / FOV, mode="r+") as out_ds:
        out_ds[0][0, 0] = phase_tzyx[0].astype(np.float32)
    corrected_phase[0] = phase_tzyx[0]
    for t in range(1,N_FRAMES):
        print(f"  t={t:02d} Applying shifts to phase: pcc_shift={pcc_shifts[t]}")
        #apply the shifts to the phas
    
        corrected_phase[t] = ants_apply_shift(phase_tzyx[t], pcc_shifts[t])
    
        with open_ome_zarr(output_zarr_path / FOV, mode="r+") as out_ds:
            out_ds[0][t, 0] = corrected_phase[t].astype(np.float32)
        print(f"  t={t:02d} Saved corrected phase: {corrected_phase[t].shape}")
    
    #%%
    corrected_mask = np.empty_like(vs_mask)  # bool
    output_path.mkdir(parents=True, exist_ok=True)
    output_zarr_path = output_path / f"{dataset}_corrected_mask_1.zarr"
    output_metadata = {
        "shape": (N_FRAMES, 1, Z, Y, X),
        "chunks": None,
        "scale": ds_scale,
        "channel_names": [vs_channel_name],
        "dtype": bool,
    }
    
    create_empty_plate(
            store_path=output_zarr_path,
            position_keys=[("B", "2", "000001")],
            **output_metadata,
    )
    corrected_mask[0] = vs_mask[0]
    with open_ome_zarr(output_zarr_path / FOV, mode="r+") as out_ds:
        out_ds[0][0, 0] = corrected_mask[0]
    for t in range(1, N_FRAMES):
        print(f"  t={t:02d} Applying shifts to mask: pcc_shift={pcc_shifts[t]}")
        shifted = ants_apply_shift(vs_mask[t].astype(np.float32), pcc_shifts[t])
        corrected_mask[t] = shifted > 0.5
    
        with open_ome_zarr(output_zarr_path / FOV, mode="r+") as out_ds:
            out_ds[0][t, 0] = corrected_mask[t]
        print(f"  t={t:02d} Saved corrected mask: {corrected_mask[t].shape}")
    
    
    
    
    
    #%%
    # --- Iteration 2: PCC on corrected_phase (no mask), apply to corrected_phase ---
    pcc_shifts_2 = [np.array([0.0, 0.0, 0.0])]
    cumulative_shift_2 = np.array([0.0, 0.0, 0.0])
    for t in range(1, N_FRAMES):
        pcc_shift = run_pcc_biahub(corrected_phase[t-1]*corrected_mask[t-1], corrected_phase[t]*corrected_mask[t])
        cumulative_shift_2 += np.array(pcc_shift)
        pcc_shifts_2.append(cumulative_shift_2.copy())
        print(f"  t={t:02d} Applying shifts to corrected phase: pcc_shift_2={pcc_shifts_2[t]}")
    
    #%%
    corrected_phase_2 = np.empty_like(phase_tzyx)
    output_zarr_path_2 = output_path / f"{dataset}_corrected_phase_2_masked.zarr"
    output_metadata_2 = {
        "shape": (N_FRAMES, 1, Z, Y, X),
        "chunks": None,
        "scale": ds_scale,
        "channel_names": [phase_channel_name],
        "dtype": np.float32,
    }
    
    create_empty_plate(
            store_path=output_zarr_path_2,
            position_keys=[("B", "2", "000001")],
            **output_metadata_2,
    )
    corrected_phase_2[0] = corrected_phase[0]
    with open_ome_zarr(output_zarr_path_2 / FOV, mode="r+") as out_ds:
        out_ds[0][0, 0] = corrected_phase_2[0].astype(np.float32)
    for t in range(1, N_FRAMES):
        corrected_phase_2[t] = ants_apply_shift(corrected_phase[t], pcc_shifts_2[t])
        with open_ome_zarr(output_zarr_path_2 / FOV, mode="r+") as out_ds:
            out_ds[0][t, 0] = corrected_phase_2[t].astype(np.float32)
        print(f"  t={t:02d} Saved corrected phase_2: {corrected_phase_2[t].shape}")
    
    #%%
    
    corrected_mask_2 = np.empty_like(vs_mask)  # bool
    output_path.mkdir(parents=True, exist_ok=True)
    output_zarr_path = output_path / f"{dataset}_corrected_mask_2.zarr"
    output_metadata = {
        "shape": (N_FRAMES, 1, Z, Y, X),
        "chunks": None,
        "scale": ds_scale,
        "channel_names": [vs_channel_name],
        "dtype": bool,
    }
    
    create_empty_plate(
            store_path=output_zarr_path,
            position_keys=[("B", "2", "000001")],
            **output_metadata,
    )
    corrected_mask_2[0] = corrected_mask[0]
    with open_ome_zarr(output_zarr_path / FOV, mode="r+") as out_ds:
        out_ds[0][0, 0] = corrected_mask_2[0]
    for t in range(1, N_FRAMES):
        print(f"  t={t:02d} Applying shifts to corrected mask: pcc_shift={pcc_shifts_2[t]}")
        shifted = ants_apply_shift(corrected_mask[t].astype(np.float32), pcc_shifts_2[t])
        corrected_mask_2[t] = shifted > 0.5
    
        with open_ome_zarr(output_zarr_path / FOV, mode="r+") as out_ds:
            out_ds[0][t, 0] = corrected_mask_2[t]
        print(f"  t={t:02d} Saved corrected mask_2: {corrected_mask_2[t].shape}")
    
    
    
    #%%
    
    plt.imshow(corrected_phase_2[1,Z//2], cmap="gray")
    plt.colorbar()
    plt.show()
    plt.imshow(corrected_mask_2[1,Z//2], cmap="gray")
    plt.colorbar()
    plt.show()
    
    plt.imshow(corrected_phase_2[1,Z//2]*corrected_mask_2[1,Z//2], cmap="gray")
    plt.colorbar()
    plt.show()
    #%%
    pcc_shifts_3 = [np.array([0.0, 0.0, 0.0])]
    cumulative_shift_3 = np.array([0.0, 0.0, 0.0])
    for t in range(1, N_FRAMES):
        pcc_shift = run_pcc_biahub(corrected_phase_2[t-1]*corrected_mask_2[t-1], corrected_phase_2[t]*corrected_mask_2[t])
        cumulative_shift_3 += np.array(pcc_shift)
        pcc_shifts_3.append(cumulative_shift_3.copy())
        print(f"  t={t:02d} Applying shifts to corrected phase_2: pcc_shift_3={pcc_shifts_3[t]}")
    
    #%%
    corrected_phase_3 = np.empty_like(phase_tzyx)
    output_zarr_path_3 = output_path / f"{dataset}_corrected_phase_masked_3.zarr"
    output_metadata_3 = {
        "shape": (N_FRAMES, 1, Z, Y, X),
        "chunks": None,
        "scale": ds_scale,
        "channel_names": [phase_channel_name],
        "dtype": np.float32,
    }
    
    create_empty_plate(
            store_path=output_zarr_path_3,
            position_keys=[("B", "2", "000001")],
            **output_metadata_3,
    )
    corrected_phase_3[0] = corrected_phase_2[0]
    with open_ome_zarr(output_zarr_path_3 / FOV, mode="r+") as out_ds:
        out_ds[0][0, 0] = corrected_phase_3[0].astype(np.float32)
    for t in range(1, N_FRAMES):
        corrected_phase_3[t] = ants_apply_shift(corrected_phase_2[t], pcc_shifts_3[t])
        with open_ome_zarr(output_zarr_path_3 / FOV, mode="r+") as out_ds:
            out_ds[0][t, 0] = corrected_phase_3[t].astype(np.float32)
        print(f"  t={t:02d} Saved corrected phase_3: {corrected_phase_3[t].shape}")
    
    #%%
    
    #%%
    # ─────────────────────────────────────────────
    # SHIFT COMPARISON: iteration 1 vs iteration 2
    # ─────────────────────────────────────────────
    section("SHIFT COMPARISON")
    
    dz_1 = np.array([s[0] for s in pcc_shifts])
    dz_2 = np.array([s[0] for s in pcc_shifts_2])
    dz_3 = np.array([s[0] for s in pcc_shifts_3])
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(-dz_1, marker="o", ms=5, lw=1.4, color="steelblue", label="iter 1 (masked_phase)")
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("drift (px)"); axes[0].set_title("Iteration 1 — PCC on masked phase")
    axes[0].legend()
    
    axes[1].plot(-dz_2, marker="s", ms=5, lw=1.4, color="tomato", label="iter 2 (corrected_phase)")
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].set_ylabel("drift (px)"); axes[1].set_title("Iteration 2 — PCC on corrected phase")
    axes[1].set_xlabel("Frame t"); axes[1].legend()
    
    axes[2].plot(-dz_3, marker="^", ms=5, lw=1.4, color="green", label="iter 3 (corrected_phase_2)")
    axes[2].axhline(0, color="gray", ls="--", lw=0.8)
    axes[2].set_ylabel("drift (px)"); axes[2].set_title("Iteration 3 — PCC on corrected phase_2")
    axes[2].set_xlabel("Frame t"); axes[2].legend()
    
    plt.suptitle("Estimated Z drift — cascade PCC", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path / "cascade_shifts.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print(f"  Iter 1 max |dz|: {np.abs(dz_1).max():.2f} px")
    print(f"  Iter 2 max |dz|: {np.abs(dz_2).max():.2f} px  (residual after iter 1)")
    print(f"  Iter 3 max |dz|: {np.abs(dz_3).max():.2f} px  (residual after iter 2)")
    #%%
    # ─────────────────────────────────────────────
    # FOCUS: original vs corrected_1 vs corrected_2
    # ─────────────────────────────────────────────
    section("FOCUS — original vs corrected_1 vs corrected_2 vs corrected_3")
    
    focus_original = np.array([
        focus_from_transverse_band(phase_tzyx[t], NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size)
        for t in range(N_FRAMES)
    ])
    focus_corrected_1 = np.array([
        focus_from_transverse_band(corrected_phase[t], NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size)
        for t in range(N_FRAMES)
    ])
    focus_corrected_2 = np.array([
        focus_from_transverse_band(corrected_phase_2[t], NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size)
        for t in range(N_FRAMES)
    ])
    focus_corrected_3 = np.array([
        focus_from_transverse_band(corrected_phase_3[t], NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size)
        for t in range(N_FRAMES)
    ])
    
    print(f"  {'stage':<20}  {'focus_std':>10}  {'vs_original':>12}")
    for label, fvals in [("original", focus_original), ("corrected_1", focus_corrected_1),
                          ("corrected_2", focus_corrected_2), ("corrected_3", focus_corrected_3)]:
        std_val = fvals.std()
        if label == "original":
            impr = "-"
        else:
            s0 = focus_original.std()
            impr = f"{(s0 - std_val) / s0 * 100:.1f}%" if s0 > 0 else "nan"
        print(f"  {label:<20}  {std_val:>10.3f}  {impr:>12}")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(focus_original, marker="o", ms=4, lw=1.5, color="black", label="original")
    ax.plot(focus_corrected_1, marker="s", ms=4, lw=1.5, ls="--", color="steelblue", label="corrected_1")
    ax.plot(focus_corrected_2, marker="^", ms=4, lw=1.5, ls="--", color="tomato", label="corrected_2")
    ax.plot(focus_corrected_3, marker="D", ms=4, lw=1.5, ls="--", color="green", label="corrected_3")
    ax.set_xlabel("Frame t"); ax.set_ylabel("Focus slice")
    ax.set_title("Focus over time — cascade PCC"); ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / "cascade_focus_over_time.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    #%%
    # ─────────────────────────────────────────────
    # ZT KYMOGRAPHS: original vs corrected_1 vs corrected_2 vs corrected_3
    # ─────────────────────────────────────────────
    section("ZT KYMOGRAPHS — cascade PCC")
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax, (label, data, fvals) in zip(axes, [
        ("original", phase_tzyx, focus_original),
        ("corrected_1", corrected_phase, focus_corrected_1),
        ("corrected_2", corrected_phase_2, focus_corrected_2),
        ("corrected_3", corrected_phase_3, focus_corrected_3),
    ]):
        kym = data.mean(axis=(-2, -1)).T  # (Z, T)
        im = ax.imshow(kym, aspect="auto", origin="lower", cmap="gray")
        ax.plot(fvals, color="yellow", lw=1.5, label="focus")
        ax.set_xlabel("Frame (t)"); ax.set_ylabel("Z slice")
        ax.set_title(label); ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax)
    plt.suptitle("ZT kymographs — cascade PCC", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path / "cascade_kymographs.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    #%%
    # ─────────────────────────────────────────────
    # MID-Z SLICES: original vs corrected at selected timepoints
    # Shows the actual image content before/after each correction
    # ─────────────────────────────────────────────
    section("MID-Z SLICES — original vs corrected_1 vs corrected_2 vs corrected_3")
    
    z_mid = Z // 2
    t_samples = [0, N_FRAMES // 4, N_FRAMES // 2, 3 * N_FRAMES // 4, N_FRAMES - 1]
    all_data = [
        ("original", phase_tzyx),
        ("corrected_1", corrected_phase),
        ("corrected_2", corrected_phase_2),
        ("corrected_3", corrected_phase_3),
    ]
    
    fig, axes = plt.subplots(len(all_data), len(t_samples), figsize=(4 * len(t_samples), 4 * len(all_data)))
    for row, (label, data) in enumerate(all_data):
        for col, t in enumerate(t_samples):
            im = axes[row, col].imshow(data[t, z_mid], cmap="gray")
            axes[row, col].set_title(f"{label}  t={t}", fontsize=9)
            axes[row, col].axis("off")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    plt.suptitle(f"Mid-Z slice (z={z_mid}) — cascade PCC", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path / "cascade_midz_slices.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    #%%
    # ─────────────────────────────────────────────
    # DIFFERENCE MAPS: (corrected_t - original_t) at mid-Z
    # Highlights where the shift moved content
    # ─────────────────────────────────────────────
    section("DIFFERENCE MAPS — corrected vs original at mid-Z")
    
    fig, axes = plt.subplots(3, len(t_samples), figsize=(4 * len(t_samples), 12))
    for row, (label, data) in enumerate([
        ("corrected_1 - original", corrected_phase - phase_tzyx),
        ("corrected_2 - corrected_1", corrected_phase_2 - corrected_phase),
        ("corrected_3 - corrected_2", corrected_phase_3 - corrected_phase_2),
    ]):
        for col, t in enumerate(t_samples):
            diff = data[t, z_mid]
            vabs = np.abs(diff).max() or 1.0
            im = axes[row, col].imshow(diff, cmap="seismic", vmin=-vabs, vmax=vabs)
            axes[row, col].set_title(f"{label}  t={t}", fontsize=8)
            axes[row, col].axis("off")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    plt.suptitle(f"Difference maps at mid-Z (z={z_mid})", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path / "cascade_diff_maps.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    #%%
    # ─────────────────────────────────────────────
    # CUMULATIVE DRIFT: all 3 iterations overlaid + total accumulated
    # ─────────────────────────────────────────────
    section("CUMULATIVE DRIFT — all iterations overlaid")
    
    dz_total = dz_1 + dz_2 + dz_3  # total cumulative correction across all iterations
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(-dz_1, marker="o", ms=4, lw=1.4, color="steelblue", label="iter 1 (masked_phase)")
    ax.plot(-dz_2, marker="s", ms=4, lw=1.4, color="tomato", label="iter 2 residual")
    ax.plot(-dz_3, marker="^", ms=4, lw=1.4, color="green", label="iter 3 residual")
    ax.plot(-dz_total, marker="D", ms=5, lw=2.0, color="black", ls="-", label="total correction")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Frame t"); ax.set_ylabel("Drift (px)")
    ax.set_title("Cumulative drift — per iteration and total")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / "cascade_cumulative_drift.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print(f"  Total correction range: {(-dz_total).min():.2f} to {(-dz_total).max():.2f} px")
    print(f"  Iter 1 accounts for {np.abs(dz_1).sum() / np.abs(dz_total).sum() * 100:.1f}% of total correction")
    print(f"  Iter 2 accounts for {np.abs(dz_2).sum() / np.abs(dz_total).sum() * 100:.1f}% of total correction")
    print(f"  Iter 3 accounts for {np.abs(dz_3).sum() / np.abs(dz_total).sum() * 100:.1f}% of total correction")
    
    #%%
    # ─────────────────────────────────────────────
    # INCREMENTAL (per-frame) SHIFTS: should shrink with each iteration
    # ─────────────────────────────────────────────
    section("INCREMENTAL SHIFTS — per-frame deltas")
    
    inc_1 = np.diff(dz_1)
    inc_2 = np.diff(dz_2)
    inc_3 = np.diff(dz_3)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    frames = np.arange(1, N_FRAMES)
    ax.bar(frames - 0.25, np.abs(inc_1), width=0.25, color="steelblue", alpha=0.8, label="iter 1")
    ax.bar(frames,        np.abs(inc_2), width=0.25, color="tomato",    alpha=0.8, label="iter 2")
    ax.bar(frames + 0.25, np.abs(inc_3), width=0.25, color="green",     alpha=0.8, label="iter 3")
    ax.set_xlabel("Frame t"); ax.set_ylabel("|incremental dz| (px)")
    ax.set_title("Per-frame incremental shift magnitude — should decrease with iterations")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / "cascade_incremental_shifts.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print(f"  Mean |inc shift|:  iter1={np.abs(inc_1).mean():.3f}  iter2={np.abs(inc_2).mean():.3f}  iter3={np.abs(inc_3).mean():.3f}")
    
    #%%
    # ─────────────────────────────────────────────
    # XZ CROSS-SECTION over time: shows Z-drift as lateral wobble
    # ─────────────────────────────────────────────
    section("XZ CROSS-SECTIONS — mid-Y")
    
    y_mid = Y // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (label, data) in zip(axes.flat, [
        ("original", phase_tzyx),
        ("corrected_1", corrected_phase),
        ("corrected_2", corrected_phase_2),
        ("corrected_3", corrected_phase_3),
    ]):
        # XZ slice at mid-Y, averaged over a few Y slices for noise reduction
        y_slice = slice(y_mid - 5, y_mid + 5)
        xz_t = data[:, :, y_slice, :].mean(axis=2)  # (T, Z, X)
        # pick t=0 and last frame side-by-side
        xz_montage = np.concatenate([xz_t[0], xz_t[-1]], axis=1)  # (Z, 2*X)
        im = ax.imshow(xz_montage, aspect="auto", origin="lower", cmap="gray")
        ax.axvline(xz_t.shape[2], color="yellow", lw=1, ls="--")
        ax.set_title(f"{label}  |  t=0 (left) vs t={N_FRAMES-1} (right)")
        ax.set_xlabel("X"); ax.set_ylabel("Z")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"XZ cross-section at mid-Y (y={y_mid}) — t=0 vs t={N_FRAMES-1}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path / "cascade_xz_cross_sections.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    #%%
    # ─────────────────────────────────────────────
    # MEAN INTENSITY PER Z-SLICE over time (Z profile stability)
    # A well-stabilized volume should have consistent Z profiles
    # ─────────────────────────────────────────────
    section("Z-PROFILE STABILITY — mean intensity per Z slice")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (label, data) in zip(axes.flat, [
        ("original", phase_tzyx),
        ("corrected_1", corrected_phase),
        ("corrected_2", corrected_phase_2),
        ("corrected_3", corrected_phase_3),
    ]):
        # z-profile: mean over (Y, X) for each (T, Z)
        z_profiles = data.mean(axis=(-2, -1))  # (T, Z)
        for t in range(N_FRAMES):
            alpha = 0.3 + 0.7 * (t / (N_FRAMES - 1))
            ax.plot(z_profiles[t], color="steelblue", alpha=alpha, lw=0.8)
        ax.plot(z_profiles[0], color="black", lw=1.5, label="t=0")
        ax.plot(z_profiles[-1], color="red", lw=1.5, label=f"t={N_FRAMES-1}")
        ax.set_xlabel("Z slice"); ax.set_ylabel("Mean intensity")
        ax.set_title(label); ax.legend(fontsize=8)
    plt.suptitle("Z-profile per timepoint — should overlap after stabilization", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path / "cascade_z_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    #%%
    # ─────────────────────────────────────────────
    # TEMPORAL STD per voxel at mid-Z: measures flicker/residual drift
    # ─────────────────────────────────────────────
    section("TEMPORAL STD — voxel-wise at mid-Z")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    std_vals = []
    for ax, (label, data) in zip(axes, [
        ("original", phase_tzyx),
        ("corrected_1", corrected_phase),
        ("corrected_2", corrected_phase_2),
        ("corrected_3", corrected_phase_3),
    ]):
        temporal_std = data[:, z_mid].std(axis=0)  # std over T at mid-Z → (Y, X)
        std_vals.append(temporal_std.mean())
        im = ax.imshow(temporal_std, cmap="inferno")
        ax.set_title(f"{label}\nmean std={temporal_std.mean():.4f}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"Temporal std per pixel at mid-Z (z={z_mid}) — lower = more stable", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path / "cascade_temporal_std.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print(f"  Mean temporal std:  original={std_vals[0]:.4f}  c1={std_vals[1]:.4f}  c2={std_vals[2]:.4f}  c3={std_vals[3]:.4f}")
    
    #%%
    # ─────────────────────────────────────────────
    # BUILD INPUT ARRAYS FOR REAL DATA
    # ─────────────────────────────────────────────
    inputs = {
        "masked_phase":        phase_tzyx,
    }
    
    #%%
    # ─────────────────────────────────────────────
    # PCC — BOTH STRATEGIES × ALL THREE INPUTS
    # ─────────────────────────────────────────────
    section("PCC dz — pairwise")
    
    dz_results = {}   # (strategy, input_name) → correction array (N_FRAMES,)
    
    for inp_name, arr in inputs.items():
        # pairwise cumulative
        vals = [0.0]; dz_c = 0.0
        for t in range(1, N_FRAMES):
            dz_c += run_pcc_biahub(arr[t-1], arr[t])[0]
            vals.append(dz_c)
        dz_results[("pairwise", inp_name)] = np.array(vals)
    
       
    
        print(f"\n  [{inp_name}]")
        print(f"    pairwise  (correction): {np.round(dz_results[('pairwise', inp_name)], 2)}")
    #%%
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, inp_name in zip(axes, INPUT_NAMES):
        ax.plot(-dz_results[("pairwise", inp_name)], color="steelblue",
                marker="o", ms=5, lw=1.4, label="pairwise-cumulative")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_ylabel("drift (px)"); ax.set_title(f"Input: {inp_name}"); ax.legend()
    axes[-1].set_xlabel("Frame t")
    plt.suptitle("Estimated Z drift (−correction)  ×  strategy  ×  input", fontsize=13)
    plt.tight_layout(); plt.show()
    
    #%%
    
    
    # ─────────────────────────────────────────────
    # FOCUS — ORIGINAL × 3 INPUTS
    # ─────────────────────────────────────────────
    section("FOCUS — original data  ×  3 inputs")
    
    focus_original = {}
    for inp_name, arr in inputs.items():
        zf_list = []
        for t in range(N_FRAMES):
            zf = focus_from_transverse_band(
                arr[t], NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size
            )
            zf_list.append(zf)
            print(f"  [{inp_name}] t={t:02d}  z_focus={zf}")
        focus_original[inp_name] = np.array(zf_list)
    
    fig, ax = plt.subplots(figsize=(9, 4))
    for inp_name in INPUT_NAMES:
        ax.plot(focus_original[inp_name], marker="o", ms=4, label=inp_name)
    ax.set_xlabel("Frame t"); ax.set_ylabel("Focus slice")
    ax.set_title("Focus over time — original"); ax.legend()
    plt.tight_layout(); plt.show()
    
    
    # ─────────────────────────────────────────────
    # ZT KYMOGRAPHS — ORIGINAL × 3 INPUTS
    # ─────────────────────────────────────────────
    section("ZT KYMOGRAPHS — original")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (inp_name, arr) in zip(axes, inputs.items()):
        kym = arr.mean(axis=(-2, -1)).T
        ax.imshow(kym, aspect="auto", origin="lower", cmap=CMAPS[inp_name])
        ax.plot(focus_original[inp_name], color="yellow", lw=1.5, label="focus")
        ax.set_xlabel("Frame (t)"); ax.set_ylabel("Z slice")
        ax.set_title(f"ZT kymograph — {inp_name}"); ax.legend()
    plt.suptitle("ZT kymographs — original", fontsize=13)
    plt.tight_layout(); plt.show()
    
    
    #%%
    # ─────────────────────────────────────────────
    # APPLY STABILIZATION + WRITE ZARRS
    # One zarr per (strategy × input) = 6 zarrs total.
    # Zarr name encodes the combination so nothing is overwritten.
    # ─────────────────────────────────────────────
    section("APPLY STABILIZATION  ×  strategy  ×  input  →  zarr per combo")
    
    output_metadata_base = {
        "chunks":        None,
        "scale":         ds_scale,
        "channel_names": [phase_channel_name],
        "dtype":         np.float32,
    }
    
    focus_stabilized = {}   # (strategy, input_name) → focus array
    
    for strategy in STRATEGIES:
        for inp_name in INPUT_NAMES:
            run_tag          = f"strat-{strategy}_inp-{inp_name}"
            output_zarr_path = output_dirpath / f"{dataset}_{run_tag}.zarr"
            print(f"\n  [{run_tag}]  →  {output_zarr_path}")
    
            create_empty_plate(
                store_path=output_zarr_path,
                position_keys=[("B", "2", "000001")],
                shape=(N_FRAMES, 1, Z_full, Y_full, X_full),
                **output_metadata_base,
            )
    
            dz_chosen = dz_results[(strategy, inp_name)]
            zf_list   = []
    
            with open_ome_zarr(output_zarr_path / FOV, mode="r+") as out_ds:
                for t in range(N_FRAMES):
                    stab_vol = ants_apply_shift(phase_tzyx[t], np.array([dz_chosen[t], 0.0, 0.0]))
                    out_ds[0][t, 0] = stab_vol.astype(np.float32)
    
                    # focus on the fly — no need to keep stab_vol in memory
                    if inp_name == "phase":
                        focus_inp = stab_vol
                    elif inp_name == "mask":
                        focus_inp = vs_mask[t].astype(np.float32)
                    else:
                        focus_inp = stab_vol * vs_mask[t]
    
                    zf = focus_from_transverse_band(
                        focus_inp, NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size
                    )
                    zf_list.append(zf)
                    print(f"    t={t:02d}  dz={dz_chosen[t]:+.2f} px  z_focus={zf:.1f}  written")
    
            focus_stabilized[(strategy, inp_name)] = np.array(zf_list)
            print(f"    focus array: {np.round(focus_stabilized[(strategy, inp_name)], 1)}")
    
    #%%
    # ─────────────────────────────────────────────
    # FOCUS SUMMARY — before vs after
    # ─────────────────────────────────────────────
    section("FOCUS SUMMARY — before vs after  ×  strategy  ×  input")
    
    print(f"\n  {'combo':<30}  {'std_before':>12}  {'std_after':>10}  {'improvement':>12}")
    for strategy in STRATEGIES:
        for inp_name in INPUT_NAMES:
            s_before = focus_original[inp_name].std()
            s_after  = focus_stabilized[(strategy, inp_name)].std()
            impr     = (s_before - s_after) / s_before * 100 if s_before > 0 else float("nan")
            tag      = f"strat-{strategy} inp-{inp_name}"
            print(f"  {tag:<30}  {s_before:>12.3f}  {s_after:>10.3f}  {impr:>11.1f}%")
    
    # One figure per input: both strategies before/after
    for inp_name in INPUT_NAMES:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(focus_original[inp_name], marker="o", lw=1.5, color="black", label="before")
        for c, strategy in zip(["steelblue", "tomato"], STRATEGIES):
            ax.plot(focus_stabilized[(strategy, inp_name)],
                    marker="s", lw=1.5, ls="--", color=c, label=f"after ({strategy})")
        ax.set_xlabel("Frame t"); ax.set_ylabel("Focus slice")
        ax.set_title(f"Focus before vs after — input: {inp_name}")
        ax.legend(); plt.tight_layout(); plt.show()
    
    #%%
    # ─────────────────────────────────────────────
    # ZT KYMOGRAPHS — BEFORE vs AFTER (phase input, both strategies)
    # ─────────────────────────────────────────────
    section("ZT KYMOGRAPHS — before vs after  ×  strategy (phase input)")
    
    n_cols = 1 + len(STRATEGIES)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    
    # Before
    kym = phase_tzyx.mean(axis=(-2, -1)).T
    im  = axes[0].imshow(kym, aspect="auto", origin="lower", cmap="gray")
    axes[0].plot(focus_original["phase"], color="yellow", lw=1.5, label="focus")
    axes[0].set_xlabel("Frame (t)"); axes[0].set_ylabel("Z slice")
    axes[0].set_title("Before"); axes[0].legend(loc="upper right")
    plt.colorbar(im, ax=axes[0])
    
    # After — read stabilized phase back from zarr
    for ax, strategy in zip(axes[1:], STRATEGIES):
        run_tag          = f"strat-{strategy}_inp-phase"
        output_zarr_path = output_dirpath / f"{dataset}_{run_tag}.zarr"
        with open_ome_zarr(output_zarr_path / FOV, mode="r") as ds:
            stab_tzyx = np.asarray(ds[0][:N_FRAMES, 0])
        kym = stab_tzyx.mean(axis=(-2, -1)).T
        im  = ax.imshow(kym, aspect="auto", origin="lower", cmap="gray")
        ax.plot(focus_stabilized[(strategy, "phase")], color="yellow", lw=1.5, label="focus")
        ax.set_xlabel("Frame (t)"); ax.set_ylabel("Z slice")
        ax.set_title(f"After ({strategy})"); ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax)
    
    plt.suptitle("ZT kymograph — phase channel", fontsize=13)
    plt.tight_layout(); plt.show()   
    #%%
    
    
    # %%
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    from iohub import open_ome_zarr
    from waveorder.focus import focus_from_transverse_band
    from skimage.registration import phase_cross_correlation as pcc
    from biahub.core.transform import Transform 
    NA_DET = 1.35
    LAMBDA_ILL = 0.500
    
    #%%
    dataset = "2024_12_03_A549_LAMP1_ZIKV_DENV"
    FOV = "B/4/000000"      
    phase_zarr_path = Path(f"/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}/1-preprocess/label-free/0-reconstruct/{dataset}.zarr/{FOV}")
    vs_zarr_path = Path(f"/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}/1-preprocess/label-free/1-virtual-stain/{dataset}.zarr/{FOV}")
    
    phase_channel_name = "Phase3D"
    vs_channel_name = "nuclei_prediction"  # or "membrane"
    
    
    with open_ome_zarr(phase_zarr_path) as ds:
        phase_channel_index = ds.channel_names.index(phase_channel_name)
        _, _, _, _, pixel_size = ds.scale
        phase_zyx = np.asarray(ds.data[:20, 0])  # (Z, Y, X)
        print(f"Phase shape: {phase_zyx.shape}, pixel_size: {pixel_size:.4f}")
        T, Z, Y, X = phase_zyx.shape
    
    with open_ome_zarr(vs_zarr_path) as ds:
        vs_channel_index = ds.channel_names.index(vs_channel_name)
        vs_zyx = np.asarray(ds.data[:20, 0])  # (Z, Y, X)
        print(f"VS shape: {vs_zyx.shape}")
    
    
    #%%%
    #otsu thresholding
    from skimage.filters import threshold_otsu
    
    from skimage.registration import phase_cross_correlation as pcc
    from scipy.ndimage import shift as shift_ndimage
    shifts = []
    vs_mask = np.zeros_like(vs_zyx)
    
    for t in range(20):
        vs_mask[t] = vs_zyx[t] > threshold_otsu(vs_zyx[t])
    #%%
    # compute shifts between consecutive frames
    for t in range(1, 20):
        s, _, _ = pcc(phase_zyx[t-1]*vs_mask[t-1], phase_zyx[t]*vs_mask[t])
        print(f"Shift: {s}") # s is (dy, dx) for 2D
        shifts.append(s)
    
    #%%
    z_focus_original = []
    for t in range(20):
        z_focus = focus_from_transverse_band(phase_zyx[t]*vs_mask[t], NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size)
        print(f"t{t}, Z focus: {z_focus}")
        z_focus_original.append(z_focus)
    
    #%%
    # apply the shifts (note: ndimage.shift shifts the image; to align t onto t-1 you usually shift by -s)
    shift_cum = np.zeros(3, dtype=float)  # (dz, dy, dx)
    z_focus_shifts = []
    shifts_cum = []
    transforms_cum = [Transform(matrix=np.eye(4))]
    for t in range(1, 20):
        s = np.asarray(shifts[t - 1], dtype=float)   # pairwise shift (dz, dy, dx) for t vs t-1
        shift_cum += s                               # cumulative shift from t to 0 (in PCC sign convention)
        shifts_cum.append(shift_cum)
    
        dz, dy, dx = shift_cum
    
        transform = np.eye(4)
        transform[2, 3] = dx
        transform[1, 3] = dy
        transform[0, 3] = dz
        transform = Transform(matrix=transform)
        transforms_cum.append(transform)
    #%%
    from iohub.ngff.utils import create_empty_plate
    output_dirpath = Path(f"/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}/debug_stab")
    output_zarr_path = output_dirpath/f"{dataset}_vs.zarr"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    T,C,Z,Y,X = ds.data.shape
    
    output_metadata = {
        "shape": (20, 1, Z, Y, X),
        "chunks": None,
        "scale": ds.scale,
        "channel_names": ["Phase3D"],
        "dtype": np.float32,
    }
    create_empty_plate(
        store_path=output_zarr_path,
        position_keys=[("C", "4", "000000")],
        **output_metadata,
    )
    #%%
    
    transforms_cum
    #%%
    import ants
    for t in range(20):
        transform = transforms_cum[t]
        phase_zyx_ants = ants.from_numpy(phase_zyx[t])
        phase_zyx_shifted = transform.to_ants().apply_to_image(phase_zyx_ants).numpy()
        plt.imshow(phase_zyx_shifted[phase_zyx_shifted.shape[0] // 2], cmap="gray")
        plt.colorbar()
        plt.show()
    
        with open_ome_zarr(output_zarr_path /f"{FOV}" , mode="r+") as output_dataset:
            output_dataset[0][t, 0] = np.asarray(phase_zyx_shifted, dtype=np.float32)
    
    # %%  --- CONFIG ---
    phase_zarr_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}/1-preprocess/label-free/0-reconstruct/{dataset}.zarr/B/1/000000")
    vs_zarr_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}/1-preprocess/label-free/1-virtual-stain/{dataset}.zarr/B/1/000000")
    
    phase_channel_name = "Phase3D"
    vs_channel_name = "nuclei_prediction"  # or "membrane"
    
    t = 1  # timepoint to debug
    
    n_blocks = (3, 3)        # (n_y, n_x) grid
    vs_signal_threshold = 0.0  # skip blocks where mean(max-proj VS) <= this
    threshold_FWHM = 1.0       # passed to focus_from_transverse_band
    min_valid_patches = 3      # minimum patches needed for a reliable estimate
    
    NA_DET = 1.35
    LAMBDA_ILL = 0.500
    
    
    # %% --- LOAD DATA ---
    with open_ome_zarr(phase_zarr_path) as ds:
        phase_channel_index = ds.channel_names.index(phase_channel_name)
        _, _, _, _, pixel_size = ds.scale
        phase_zyx = np.asarray(ds.data[(t, phase_channel_index)])  # (Z, Y, X)
        print(f"Phase shape: {phase_zyx.shape}, pixel_size: {pixel_size:.4f}")
    
    with open_ome_zarr(vs_zarr_path) as ds:
        vs_channel_index = ds.channel_names.index(vs_channel_name)
        vs_zyx = np.asarray(ds.data[(t, vs_channel_index)])  # (Z, Y, X)
        print(f"VS shape: {vs_zyx.shape}")
    #%%
    ## sum phase to see if there are data
    phase_sum = phase_zyx.sum(axis=0)
    print(f"Phase sum: {phase_sum.shape}")
    plt.imshow(phase_zyx[phase_zyx.shape[0] // 2], cmap="gray")
    plt.colorbar()
    plt.show()
    
    
    vs_sum = vs_zyx.sum(axis=0)
    print(f"VS sum: {vs_sum.shape}")
    plt.imshow(vs_zyx[vs_zyx.shape[0] // 2], cmap="magma")
    plt.colorbar()
    plt.show()
    
    
    
    # %% --- BUILD VS PRESENCE MAP ---
    # Max-project VS along Z to get a 2D map of "where are the cells"
    vs_yx = vs_zyx.max(axis=0)  # (Y, X)
    
    Z, Y, X = phase_zyx.shape
    ny, nx = n_blocks
    bh = Y // ny
    bw = X // nx
    
    print(f"FOV: ({Y}, {X})  |  block size: ({bh}, {bw})")
    
    # %% --- PATCH SELECTION AND FOCUS ESTIMATION ---
    focus_indices = []
    block_info = []  # for visualization
    
    for i in range(ny):
        for j in range(nx):
            y0, y1 = i * bh, (i + 1) * bh
            x0, x1 = j * bw, (j + 1) * bw
    
            vs_block_mean = vs_yx[y0:y1, x0:x1].mean()
            if vs_block_mean <= vs_signal_threshold:
                print(f"Block ({i},{j}): SKIPPED  (VS mean = {vs_block_mean:.3f})")
                block_info.append((i, j, y0, y1, x0, x1, vs_block_mean, None))
                continue
    
            phase_block = phase_zyx[:, y0:y1, x0:x1]
            z_idx = focus_from_transverse_band(
                phase_block,
                NA_det=NA_DET,
                lambda_ill=LAMBDA_ILL,
                pixel_size=pixel_size,
            )
            focus_indices.append(z_idx)
            block_info.append((i, j, y0, y1, x0, x1, vs_block_mean, z_idx))
            print(f"Block ({i},{j}): VS mean = {vs_block_mean:.3f}  |  focus = {z_idx}")
    
    # %% --- AGGREGATE ---
    valid = [z for z in focus_indices if z is not None]
    if len(valid) >= min_valid_patches:
        final_focus = int(np.median(valid))
        print(f"\nFinal focus (median of {len(valid)} patches): {final_focus}")
    else:
        final_focus = None
        print(f"\nNot enough valid patches ({len(valid)} < {min_valid_patches}), focus = None")
    
    # %% --- VISUALIZE ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # VS max-projection with block grid
    axes[0].imshow(vs_yx, cmap="magma")
    axes[0].set_title(f"VS max-proj ({vs_channel_name})")
    for i in range(1, ny):
        axes[0].axhline(i * bh, color="white", lw=0.8, alpha=0.7)
    for j in range(1, nx):
        axes[0].axvline(j * bw, color="white", lw=0.8, alpha=0.7)
    
    # Phase at final focus slice
    if final_focus is not None:
        axes[1].imshow(phase_zyx[final_focus], cmap="gray")
        axes[1].set_title(f"Phase at z={final_focus} (estimated focus)")
    else:
        axes[1].imshow(phase_zyx[phase_zyx.shape[0] // 2], cmap="gray")
        axes[1].set_title("Phase at z=center (focus failed)")
    
    # Focus index per block
    focus_map = np.full((ny, nx), np.nan)
    for i, j, y0, y1, x0, x1, vs_mean, z in block_info:
        if z is not None:
            focus_map[i, j] = z
    im = axes[2].imshow(focus_map, cmap="viridis", vmin=0, vmax=Z)
    axes[2].set_title("Focus index per block")
    for i in range(ny):
        for j in range(nx):
            val = focus_map[i, j]
            txt = f"{val:.0f}" if not np.isnan(val) else "N/A"
            axes[2].text(j, i, txt, ha="center", va="center", color="white", fontsize=9)
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("debug_vs_guided_focus.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved debug_vs_guided_focus.png")
    
    
    #%%
    
    # %%
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    from iohub import open_ome_zarr
    from waveorder.focus import focus_from_transverse_band
    
    # %%  --- CONFIG ---
    
    n_blocks = (3, 3)          # (n_y, n_x) grid
    vs_signal_threshold = 0.0  # skip blocks where mean(max-proj VS) <= this
    threshold_FWHM = 1.0       # passed to focus_from_transverse_band
    min_valid_patches = 3      # minimum patches needed for a reliable estimate
    mask_radius = 0.9         # circular crop fraction (set to None to skip comparison)
    
    NA_DET = 1.35
    LAMBDA_ILL = 0.500
    
    
    # %% --- SETUP ---
    Z, Y, X = phase_zyx.shape
    vs_yx = vs_zyx.max(axis=0)  # max-project VS along Z
    
    ny, nx = n_blocks
    bh = Y // ny
    bw = X // nx
    print(f"FOV: ({Y}, {X})  |  block size: ({bh}, {bw})")
    
    # Build circular mask
    center = (Y // 2, X // 2)
    if mask_radius is not None:
        radius = int(mask_radius * min(center))
        y_grid, x_grid = np.ogrid[:Y, :X]
        circular_mask = (x_grid - center[1]) ** 2 + (y_grid - center[0]) ** 2 <= radius ** 2
        print(f"Circular mask: center={center}, radius={radius}px ({mask_radius:.0%} of half-dim)")
    else:
        circular_mask = np.ones((Y, X), dtype=bool)
        radius = None
    #%%
    z_idx = focus_from_transverse_band(
                phase_zyx,
                NA_det=NA_DET,
                lambda_ill=LAMBDA_ILL,
                pixel_size=pixel_size,
                threshold_FWHM=threshold_FWHM,
            )
    print(f"Focus index: {z_idx}")
    # %% --- CORE FUNCTION ---
    def estimate_focus_from_patches(use_circular_mask: bool) -> dict:
        """Run patch-based focus estimation, optionally excluding border blocks."""
        mask = circular_mask if use_circular_mask else np.ones((Y, X), dtype=bool)
    
        focus_indices = []
        block_info = []
    
        for i in range(ny):
            for j in range(nx):
                y0, y1 = i * bh, (i + 1) * bh
                x0, x1 = j * bw, (j + 1) * bw
    
                # Skip blocks mostly outside the circular mask
                if mask[y0:y1, x0:x1].mean() < 0.5:
                    block_info.append(dict(i=i, j=j, z=None, reason="border"))
                    continue
    
                # Skip blocks with low VS signal
                vs_block_mean = vs_yx[y0:y1, x0:x1].mean()
                if vs_block_mean <= vs_signal_threshold:
                    block_info.append(dict(i=i, j=j, z=None, reason="no_signal"))
                    continue
    
                phase_block = phase_zyx[:, y0:y1, x0:x1]
                z_idx = focus_from_transverse_band(
                    phase_block,
                    NA_det=NA_DET,
                    lambda_ill=LAMBDA_ILL,
                    pixel_size=pixel_size,
                    threshold_FWHM=threshold_FWHM,
                )
                block_info.append(dict(i=i, j=j, z=z_idx, reason="valid" if z_idx is not None else "flat"))
                if z_idx is not None:
                    focus_indices.append(z_idx)
    
        valid = [z for z in focus_indices if z is not None]
        final_focus = int(np.median(valid)) if len(valid) >= min_valid_patches else None
    
        return dict(final_focus=final_focus, n_valid=len(valid), block_info=block_info)
    
    
    # %% --- RUN BOTH ---
    result_no_mask = estimate_focus_from_patches(use_circular_mask=False)
    result_with_mask = estimate_focus_from_patches(use_circular_mask=True)
    
    print(f"\nWithout circular mask:  focus={result_no_mask['final_focus']}  ({result_no_mask['n_valid']} valid patches)")
    print(f"With    circular mask:  focus={result_with_mask['final_focus']}  ({result_with_mask['n_valid']} valid patches)")
    
    
    # %% --- VISUALIZE COMPARISON ---
    def build_focus_map(block_info):
        focus_map = np.full((ny, nx), np.nan)
        for b in block_info:
            if b["z"] is not None:
                focus_map[b["i"], b["j"]] = b["z"]
        return focus_map
    
    
    def draw_block_labels(ax, block_info):
        colors = {"valid": "white", "no_signal": "yellow", "border": "red", "flat": "orange"}
        for b in block_info:
            if b["z"] is not None:
                ax.text(b["j"], b["i"], str(b["z"]), ha="center", va="center",
                        color=colors["valid"], fontsize=9, fontweight="bold")
            else:
                ax.text(b["j"], b["i"], b["reason"], ha="center", va="center",
                        color=colors.get(b["reason"], "white"), fontsize=7)
    
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("VS-guided focus — with vs. without circular mask", fontsize=13)
    
    for row, (result, label) in enumerate([
        (result_no_mask,   "No circular mask"),
        (result_with_mask, f"Circular mask  r={mask_radius}"),
    ]):
        focus = result["final_focus"]
        block_info = result["block_info"]
        focus_map = build_focus_map(block_info)
    
        # Col 0: VS max-proj + grid
        axes[row, 0].imshow(vs_yx, cmap="magma")
        if row == 1 and radius is not None:
            circle = plt.Circle((center[1], center[0]), radius,
                                 color="cyan", fill=False, lw=1.5, linestyle="--")
            axes[row, 0].add_patch(circle)
        for k in range(1, ny):
            axes[row, 0].axhline(k * bh, color="white", lw=0.8, alpha=0.6)
        for k in range(1, nx):
            axes[row, 0].axvline(k * bw, color="white", lw=0.8, alpha=0.6)
        axes[row, 0].set_title(f"[{label}]  VS max-proj")
    
        # Col 1: phase at estimated focus
        z_show = focus if focus is not None else Z // 2
        title_suffix = f"z={focus}" if focus is not None else "z=center (failed)"
        axes[row, 1].imshow(phase_zyx[z_show], cmap="gray")
        axes[row, 1].set_title(f"[{label}]  Phase at {title_suffix}")
    
        # Col 2: per-block focus heatmap
        im = axes[row, 2].imshow(focus_map, cmap="viridis", vmin=0, vmax=Z)
        draw_block_labels(axes[row, 2], block_info)
        plt.colorbar(im, ax=axes[row, 2])
        axes[row, 2].set_title(f"[{label}]  Focus map  (n_valid={result['n_valid']})")
    
    plt.tight_layout()
    plt.savefig("debug_vs_guided_focus.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved debug_vs_guided_focus.png")
    
    # %%
    
    #%%
    
    
    # ─────────────────────────────────────────────
    # SIGN CONVENTION TEST
    # ─────────────────────────────────────────────
    section("SIGN CONVENTION TEST — skimage vs biahub PCC  ×  ANTs")
    
    rng       = np.random.default_rng(42)
    toy       = rng.standard_normal((64, 128, 128)).astype(np.float32)
    gt_shift  = np.array([5.0, 0.0, 0.0])
    toy_moved = ndimage_shift(toy, gt_shift)
    
    ski_raw      = np.array(pcc_skimage(toy, toy_moved)[0], dtype=float)
    toy_corr_ski = ants_apply_shift(toy_moved, +ski_raw)
    residual_ski = np.array(pcc_skimage(toy, toy_corr_ski)[0])
    
    bh_raw      = np.array(pcc_biahub(toy, toy_moved, normalization=None, verbose=False), dtype=float)
    toy_corr_bh = ants_apply_shift(toy_moved, -bh_raw)
    residual_bh = np.array(pcc_skimage(toy, toy_corr_bh)[0])
    
    print(f"  GT shift            : {gt_shift}")
    print(f"  skimage raw         : {ski_raw}  → correction +ski_raw  residual={residual_ski.round(2)}")
    print(f"  biahub  raw         : {bh_raw}   → correction -bh_raw   residual={residual_bh.round(2)}")
    SIGN = {"skimage": +1, "biahub": -1}
    
    print(f"\n  Sign summary: skimage={SIGN['skimage']:+d}, biahub={SIGN['biahub']:+d}")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    z_mid = toy.shape[0] // 2
    for row, (label, corr) in enumerate([("skimage", toy_corr_ski), ("biahub", toy_corr_bh)]):
        vmin, vmax = toy[z_mid].min(), toy[z_mid].max()
        axes[row,0].imshow(toy[z_mid],       cmap="gray", vmin=vmin, vmax=vmax); axes[row,0].set_title("Reference")
        axes[row,1].imshow(toy_moved[z_mid], cmap="gray", vmin=vmin, vmax=vmax); axes[row,1].set_title("Moved +5px Z")
        axes[row,2].imshow(corr[z_mid],      cmap="gray", vmin=vmin, vmax=vmax); axes[row,2].set_title(f"Corrected ({label})")
        diff = toy[z_mid] - corr[z_mid]; vd = np.abs(diff).max()
        axes[row,3].imshow(diff, cmap="seismic", vmin=-vd, vmax=vd); axes[row,3].set_title("Ref − Corrected")
        axes[row,0].set_ylabel(label, fontsize=12)
    for ax in axes.flat: ax.axis("off")
    plt.suptitle("Sign convention test", fontsize=12); plt.tight_layout(); plt.show()
    
    
    # ─────────────────────────────────────────────
    # CHAINED DRIFT SYNTHETIC TOY TEST
    # ─────────────────────────────────────────────
    section("CHAINED DRIFT SYNTHETIC TOY TEST — pairwise cumulative  ×  ANTs")
    
    gt_incremental = [0, +3, +2, -1, +4]
    gt_cumulative  = np.cumsum(gt_incremental)
    n_toy_frames   = len(gt_incremental)
    toy_frames     = [ndimage_shift(toy, [dz, 0, 0]) for dz in gt_cumulative]
    
    print(f"  GT incremental : {gt_incremental}")
    print(f"  GT cumulative  : {gt_cumulative.tolist()}\n")
    
    dz_cum_estimated = [0.0]; dz_cum = 0.0
    for t in range(1, n_toy_frames):
        dz_inc = run_pcc_biahub(toy_frames[t-1], toy_frames[t])[0]
        dz_cum += dz_inc
        dz_cum_estimated.append(dz_cum)
        print(f"  t={t}  dz_inc={dz_inc:+.1f}  dz_cum={dz_cum:+.1f}  gt_cum={gt_cumulative[t]:+d}")
    
    dz_cum_estimated   = np.array(dz_cum_estimated)
    dz_drift_estimated = -dz_cum_estimated
    
    print(f"\n  {'t':>3}  {'gt_drift':>10}  {'est_drift':>10}  {'error':>7}  {'residual_dz':>12}")
    corrected_frames = []
    for t in range(n_toy_frames):
        corr = ants_apply_shift(toy_frames[t], np.array([dz_cum_estimated[t], 0.0, 0.0]))
        corrected_frames.append(corr)
        residual = np.array(pcc_skimage(toy_frames[0], corr)[0])
        ok  = "✓" if abs(residual[0]) <= 1.5 else "✗"
        err = dz_drift_estimated[t] - gt_cumulative[t]
        print(f"  {t:>3}  {gt_cumulative[t]:>+10.1f}  {dz_drift_estimated[t]:>+10.1f}"
              f"  {err:>+7.1f}  {residual[0]:>+12.2f}  {ok}")
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(gt_cumulative,      marker="o", lw=1.5, label="GT drift")
    ax.plot(dz_drift_estimated, marker="s", lw=1.5, ls="--", label="Estimated drift")
    ax.set_xlabel("Frame t"); ax.set_ylabel("dz (px)")
    ax.set_title("Chained drift: GT vs estimated"); ax.legend(); plt.tight_layout(); plt.show()
    
    #%%
    # ─────────────────────────────────────────────
    # CASCADE PCC: run PCC on masked_phase, apply to phase,
    # then re-run PCC on corrected masked_phase, apply again.
    # ─────────────────────────────────────────────
    section("CASCADE PCC PIPELINE")
    
    N_CASCADE = 2  # number of PCC iterations
    position_keys = [tuple(FOV.split("/"))]
    
    cascade_phase = {0: phase_tzyx}  # iteration 0 = original
    cascade_dz = {}
    cascade_focus = {}
    
    # Focus on original masked phase
    cascade_focus[0] = np.array([
        focus_from_transverse_band(
            phase_tzyx[t] * vs_mask[t], NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size
        ) for t in range(N_FRAMES)
    ])
    print(f"  Original focus std: {cascade_focus[0].std():.3f}")
    
    for iteration in range(1, N_CASCADE + 1):
        section(f"CASCADE ITERATION {iteration}")
    
        current_phase = cascade_phase[iteration - 1]
        if iteration == 1:
            pcc_input = current_phase * vs_mask
        else:
            pcc_input = current_phase
    
        # 1. PCC (pairwise cumulative)
        print(f"  Running PCC on {'masked_phase' if iteration == 1 else 'phase_corrected'}...")
        dz_cum = [0.0]; cumulative = 0.0
        for t in range(1, N_FRAMES):
            raw = np.array(pcc_biahub(pcc_input[t-1], pcc_input[t], normalization=None, verbose=False), dtype=float)
            cumulative += SIGN["biahub"] * raw[0]
            dz_cum.append(cumulative)
        dz_corrections = np.array(dz_cum)
        cascade_dz[iteration] = dz_corrections
        print(f"  dz corrections: {np.round(dz_corrections, 2)}")
    
        # 2. Apply shifts to phase
        print(f"  Applying shifts to phase...")
        corrected_phase = np.empty_like(current_phase)
        for t in range(N_FRAMES):
            corrected_phase[t] = ants_apply_shift(current_phase[t], np.array([dz_corrections[t], 0.0, 0.0]))
            print(f"    t={t:02d}  dz={dz_corrections[t]:+.2f} px")
        cascade_phase[iteration] = corrected_phase
    
        # 3. Save corrected phase to zarr
        suffix = "phase_corrected" if iteration == 1 else f"phase_corrected_{iteration}"
        zarr_path = output_dirpath / f"{dataset}_{suffix}.zarr"
        create_empty_plate(
            store_path=zarr_path,
            position_keys=position_keys,
            shape=(N_FRAMES, 1, Z_full, Y_full, X_full),
            **output_metadata_base,
        )
        with open_ome_zarr(zarr_path / FOV, mode="r+") as out_ds:
            for t in range(N_FRAMES):
                out_ds[0][t, 0] = corrected_phase[t].astype(np.float32)
        print(f"  Saved: {zarr_path}")
    
        # 4. Focus on corrected phase (masked for iter 1, unmasked for iter 2+)
        if iteration == 1:
            focus_input = corrected_phase * vs_mask
        else:
            focus_input = corrected_phase
        cascade_focus[iteration] = np.array([
            focus_from_transverse_band(
                focus_input[t], NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size
            ) for t in range(N_FRAMES)
        ])
        print(f"  Focus std after iteration {iteration}: {cascade_focus[iteration].std():.3f}")
    
    #%%
    # ─────────────────────────────────────────────
    # CASCADE SUMMARY
    # ─────────────────────────────────────────────
    section("CASCADE FOCUS SUMMARY")
    
    cascade_labels = {0: "original"}
    for i in range(1, N_CASCADE + 1):
        cascade_labels[i] = f"corrected_{i}"
    
    print(f"  {'stage':<20}  {'focus_std':>10}  {'vs_original':>12}")
    for i in range(N_CASCADE + 1):
        std_val = cascade_focus[i].std()
        if i == 0:
            impr = "-"
        else:
            s0 = cascade_focus[0].std()
            impr = f"{(s0 - std_val) / s0 * 100:.1f}%" if s0 > 0 else "nan"
        print(f"  {cascade_labels[i]:<20}  {std_val:>10.3f}  {impr:>12}")
    
    #%%
    # --- Focus over time: original vs corrected_1 vs corrected_2 ---
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_cascade = ["black", "steelblue", "tomato", "green"]
    for i in range(N_CASCADE + 1):
        ls = "-" if i == 0 else "--"
        ax.plot(cascade_focus[i], marker="o", ms=4, lw=1.5, ls=ls,
                color=colors_cascade[i % len(colors_cascade)], label=cascade_labels[i])
    ax.set_xlabel("Frame t"); ax.set_ylabel("Focus slice")
    ax.set_title("Focus over time — cascade PCC"); ax.legend()
    plt.tight_layout()
    plt.savefig(output_dirpath / "cascade_focus_over_time.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    #%%
    # --- Drift corrections per iteration ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(1, N_CASCADE + 1):
        ax.plot(-cascade_dz[i], marker="o", ms=4, lw=1.4, label=f"iteration {i}")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Frame t"); ax.set_ylabel("Estimated drift (px)")
    ax.set_title("Estimated Z drift per cascade iteration"); ax.legend()
    plt.tight_layout()
    plt.savefig(output_dirpath / "cascade_drift_per_iteration.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    #%%
    # --- ZT kymographs: original / corrected_1 / corrected_2 ---
    n_panels = N_CASCADE + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        kym = cascade_phase[i].mean(axis=(-2, -1)).T
        im = ax.imshow(kym, aspect="auto", origin="lower", cmap="gray")
        ax.plot(cascade_focus[i], color="yellow", lw=1.5, label="focus")
        ax.set_xlabel("Frame (t)"); ax.set_ylabel("Z slice")
        ax.set_title(cascade_labels[i]); ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax)
    plt.suptitle("ZT kymographs — cascade PCC", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dirpath / "cascade_kymographs.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print("\nCascade PCC done. Outputs saved to:", output_dirpath)


#%%
# =============================================================
# PLATE-LEVEL LF STABILIZATION — cumulative PCC with submitit
# =============================================================
# Stabilizes the LF phase for all FOVs using cumulative PCC
# (t-1 as reference, masked by virtual stain).
# Each FOV runs as an independent slurm job.
# Output: a stabilized LF zarr that can be passed to dynacell.

import submitit
from glob import glob
from skimage.filters import threshold_otsu


def stabilize_fov(
    phase_fov_path: Path,
    vs_fov_path: Path,
    output_zarr_path: Path,
    fov: str,
    phase_channel: str = "Phase3D",
    ls_fov_path: Path | None = None,
    ls_channel: str = "raw GFP EX488 EM525-45",
) -> dict:
    """Stabilize one FOV using cumulative PCC with t-1 as reference.

    Computes pairwise PCC on masked phase (virtual stain mask) to estimate
    Z-drift, accumulates shifts, then applies them frame-by-frame with ANTs.
    Processes frame-by-frame to keep memory low (~4GB peak).

    If ls_fov_path is provided, also applies PCC shifts to LS and computes
    mid-slice 2D Laplacian variance per timepoint for blur detection.

    Returns dict with cumulative shifts, focus values, and Laplacian QC.
    """
    import ants
    from iohub import open_ome_zarr
    from skimage.registration import phase_cross_correlation as pcc_skimage
    from skimage.filters import threshold_otsu
    from waveorder.focus import focus_from_transverse_band
    from biahub.core.transform import Transform

    NA_DET = 1.35
    LAMBDA_ILL = 0.500

    def _ants_apply_shift(vol_zyx, shift_zyx):
        M = np.eye(4)
        M[0, 3] = -shift_zyx[0]
        M[1, 3] = -shift_zyx[1]
        M[2, 3] = -shift_zyx[2]
        transform = Transform(matrix=M)
        ants_img = ants.from_numpy(vol_zyx.astype(np.float32))
        return transform.to_ants().apply_to_image(ants_img, interpolation="linear").numpy()

    print(f"\n=== Stabilizing FOV {fov} ===")

    with open_ome_zarr(phase_fov_path) as phase_ds, \
         open_ome_zarr(vs_fov_path) as vs_ds:

        arr = phase_ds.data.dask_array()
        T, C, Z, Y, X = arr.shape
        pixel_size = phase_ds.scale[-1]
        ch_phase = phase_ds.channel_names.index(phase_channel)
        print(f"  Shape: {arr.shape}, pixel_size={pixel_size}")

        # --- Pass 1: estimate cumulative shifts ---
        pcc_shifts = [np.array([0.0, 0.0, 0.0])]
        cumulative_shift = np.array([0.0, 0.0, 0.0])

        prev_phase = np.asarray(arr[0, ch_phase])
        prev_vs = np.asarray(vs_ds.data[0, 0])  # nuclei
        prev_mask = prev_vs > threshold_otsu(prev_vs)

        for t in range(1, T):
            curr_phase = np.asarray(arr[t, ch_phase])
            curr_vs = np.asarray(vs_ds.data[t, 0])
            curr_mask = curr_vs > threshold_otsu(curr_vs)

            shift, _, _ = pcc_skimage(
                prev_phase * prev_mask,
                curr_phase * curr_mask,
            )
            cumulative_shift = cumulative_shift + np.array(shift)
            pcc_shifts.append(cumulative_shift.copy())
            print(f"  t={t:02d}  shift={shift}  cumulative={cumulative_shift}")

            prev_phase = curr_phase
            prev_mask = curr_mask

    # --- Pass 2: apply shifts and write ---
    with open_ome_zarr(phase_fov_path) as phase_ds, \
         open_ome_zarr(output_zarr_path / fov, mode="r+") as out_ds:

        arr = phase_ds.data.dask_array()
        T = arr.shape[0]
        ch_phase = phase_ds.channel_names.index(phase_channel)
        pixel_size = phase_ds.scale[-1]

        focus_before = []
        focus_after = []

        for t in range(T):
            phase_t = np.asarray(arr[t, ch_phase])

            # Focus before
            zf_before = focus_from_transverse_band(
                phase_t, NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size,
            )
            focus_before.append(float(zf_before))

            if t == 0:
                corrected = phase_t
            else:
                corrected = _ants_apply_shift(phase_t, pcc_shifts[t])

            # Focus after
            zf_after = focus_from_transverse_band(
                corrected, NA_det=NA_DET, lambda_ill=LAMBDA_ILL, pixel_size=pixel_size,
            )
            focus_after.append(float(zf_after))

            out_ds[0][t, 0] = corrected.astype(np.float32)
            print(f"  t={t:02d}  focus: {zf_before:.1f} -> {zf_after:.1f}")

    dz = np.array([s[0] for s in pcc_shifts])
    print(f"  Total Z drift range: {dz.min():.2f} to {dz.max():.2f} px")
    print(f"  Focus std: before={np.std(focus_before):.2f}, after={np.std(focus_after):.2f}")

    # --- Pass 3: apply PCC shifts to LS mid-slice, compute Laplacian ---
    laplacian_vars = []
    if ls_fov_path is not None:
        from scipy.ndimage import laplace
        from iohub import open_ome_zarr

        print(f"\n  --- LS Laplacian QC (mid-slice) ---")
        with open_ome_zarr(ls_fov_path) as ls_ds:
            ls_arr = ls_ds.data.dask_array()
            T_ls = ls_arr.shape[0]
            ch_ls = ls_ds.channel_names.index(ls_channel)
            Z_ls = ls_arr.shape[2]
            z_mid = Z_ls // 2
            print(f"  LS shape: {ls_arr.shape}, z_mid={z_mid}")

            for t in range(T_ls):
                # Adjust mid-slice index by cumulative Z shift
                z_shift = int(round(pcc_shifts[t][0])) if t < len(pcc_shifts) else 0
                z_adj = max(0, min(Z_ls - 1, z_mid - z_shift))

                slice_2d = np.asarray(ls_arr[t, ch_ls, z_adj]).astype(np.float32)
                lap = laplace(slice_2d)
                lap_var = float(np.var(lap))
                laplacian_vars.append(lap_var)
                print(f"  t={t:02d}  z_adj={z_adj}  Laplacian var={lap_var:.2f}")

        # Outlier detection: median / MAD
        arr_lap = np.array(laplacian_vars)
        med = np.median(arr_lap)
        mad = np.median(np.abs(arr_lap - med))
        if mad > 0:
            z_scores = (arr_lap - med) / (mad * 1.4826)
            outliers = np.where(z_scores < -2.0)[0]
        else:
            z_scores = np.zeros_like(arr_lap)
            outliers = np.array([], dtype=int)
        print(f"  Laplacian median={med:.2f}, MAD={mad:.2f}")
        print(f"  Outliers (z<-2): {outliers.tolist()}")

    return {
        "fov": fov,
        "pcc_shifts": pcc_shifts,
        "focus_before": focus_before,
        "focus_after": focus_after,
        "laplacian_vars": laplacian_vars,
    }


def run_stabilization(
    root_path: Path,
    dataset: str,
    local: bool = False,
    exclude_fovs: list[str] | None = None,
):
    """Stabilize all FOVs in parallel using submitit.

    Creates a stabilized LF zarr plate at:
        {root_path}/{dataset}/debug_stab/{dataset}_stabilized.zarr
    """
    phase_zarr = root_path / dataset / "1-preprocess" / "label-free" / "0-reconstruct" / f"{dataset}.zarr"
    vs_zarr = root_path / dataset / "1-preprocess" / "label-free" / "1-virtual-stain" / f"{dataset}.zarr"
    ls_zarr = root_path / dataset / "1-preprocess" / "light-sheet" / "raw" / "1-register" / f"{dataset}.zarr"
    output_zarr = root_path / dataset / "debug_stab" / f"{dataset}_stabilized.zarr"
    output_dir = root_path / dataset / "debug_stab"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover FOVs
    position_dirpaths = sorted([Path(p) for p in glob(str(phase_zarr / "*" / "*" / "*"))])
    position_keys = [p.parts[-3:] for p in position_dirpaths]
    fovs = ["/".join(k) for k in position_keys]
    print(f"Found {len(fovs)} FOVs")

    if exclude_fovs:
        exclude_set = set(exclude_fovs)
        fovs = [f for f in fovs if f not in exclude_set]
        position_keys = [tuple(f.split("/")) for f in fovs]
        print(f"After exclusions: {len(fovs)} FOVs")

    # Read metadata from first FOV
    first_fov = fovs[0]
    with open_ome_zarr(phase_zarr / first_fov) as ds:
        shape = ds.data.shape
        scale = list(ds.scale)
        channel_names = list(ds.channel_names)
        dtype = np.float32

    print(f"Shape per FOV: {shape}")
    print(f"Output zarr: {output_zarr}")

    # Create output plate
    create_empty_plate(
        store_path=output_zarr,
        position_keys=[tuple(f.split("/")) for f in fovs],
        shape=shape,
        chunks=(1, 1, shape[2], shape[3], shape[4]),
        scale=scale,
        channel_names=channel_names,
        dtype=dtype,
    )
    print(f"Created output plate at {output_zarr}")

    # Submit jobs
    slurm_out = output_dir / "slurm_output"
    slurm_out.mkdir(parents=True, exist_ok=True)

    cluster = "local" if local else "slurm"
    executor = submitit.AutoExecutor(folder=slurm_out, cluster=cluster)
    executor.update_parameters(
        slurm_job_name="lf_stabilize",
        slurm_mem_per_cpu="8G",
        slurm_cpus_per_task=4,
        slurm_array_parallelism=100,
        slurm_time=120,
        slurm_partition="gpu",
    )

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for fov in fovs:
            job = executor.submit(
                stabilize_fov,
                phase_fov_path=phase_zarr / fov,
                vs_fov_path=vs_zarr / fov,
                output_zarr_path=output_zarr,
                fov=fov,
                ls_fov_path=ls_zarr / fov,
            )
            jobs.append(job)

    print(f"Submitted {len(jobs)} stabilization jobs")
    job_ids = [job.job_id for job in jobs]
    log_path = slurm_out / "job_ids.log"
    with log_path.open("w") as f:
        f.write("\n".join(job_ids))
    print(f"Job IDs: {log_path}")

    # Wait for all jobs
    print("Waiting for jobs to complete...")
    results = [job.result() for job in jobs]

    # Summary plot: focus std before/after per FOV
    fig, ax = plt.subplots(figsize=(12, 5))
    fov_names = [r["fov"] for r in results]
    std_before = [np.std(r["focus_before"]) for r in results]
    std_after = [np.std(r["focus_after"]) for r in results]
    x = np.arange(len(fov_names))
    ax.bar(x - 0.15, std_before, 0.3, label="before", color="tab:red", alpha=0.7)
    ax.bar(x + 0.15, std_after, 0.3, label="after", color="tab:blue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("/", "_") for f in fov_names], rotation=90, fontsize=6)
    ax.set_ylabel("Focus std (slices)")
    ax.set_title("Z-focus stability: before vs after PCC stabilization")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stabilization_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot to {output_dir / 'stabilization_summary.png'}")

    # Laplacian QC plot per FOV
    for r in results:
        if r["laplacian_vars"]:
            lap = np.array(r["laplacian_vars"])
            med = np.median(lap)
            mad = np.median(np.abs(lap - med))
            z_scores = (lap - med) / (mad * 1.4826) if mad > 0 else np.zeros_like(lap)
            outliers = np.where(z_scores < -2.0)[0]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            t_axis = np.arange(len(lap))

            ax1.plot(t_axis, lap, "o-", ms=4)
            ax1.axhline(med, color="green", ls="--", label=f"median={med:.1f}")
            for idx in outliers:
                ax1.axvline(idx, color="red", alpha=0.5, lw=1.5)
            ax1.set_ylabel("Laplacian variance")
            ax1.set_title(f"LS mid-slice Laplacian QC — {r['fov']}")
            ax1.legend()

            ax2.plot(t_axis, z_scores, "o-", ms=4)
            ax2.axhline(-2.0, color="red", ls="--", label="threshold=-2")
            for idx in outliers:
                ax2.axvline(idx, color="red", alpha=0.5, lw=1.5)
            ax2.set_xlabel("Timepoint")
            ax2.set_ylabel("z-score (MAD)")
            ax2.legend()

            plt.tight_layout()
            fname = f"laplacian_qc_{r['fov'].replace('/', '_')}.png"
            plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved Laplacian QC plot: {fname}")
            if len(outliers) > 0:
                print(f"  Outlier timepoints: {outliers.tolist()}")

    print(f"\nStabilized zarr: {output_zarr}")
    return output_zarr


# --- Run stabilization ---
if __name__ == "__main__":
    stab_root = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/")
    stab_dataset = "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV"
    run_stabilization(
        root_path=stab_root,
        dataset=stab_dataset,
        local=False,
        exclude_fovs=[
            "A/1/000000", "A/1/000001", "A/1/001000", "A/1/002000", "A/1/002001",
            "A/2/000000", "A/2/000001", "A/2/001000", "A/2/001001", "A/2/002000", "A/2/002001",
            "A/3/000000", "A/3/000001", "A/3/001000", "A/3/001001",
            "B/1/000000", "B/1/000001", "B/1/001000", "B/1/001001", "B/1/002000", "B/1/002001",
            "B/2/000000", "B/2/000001", "B/2/001000", "B/2/001001", "B/2/002000", "B/2/002001",
            "C/1/000000", "C/1/000001", "C/1/001000", "C/1/001001", "C/1/002000", "C/1/002001",
            "C/2/000000", "C/2/000001", "C/2/001000", "C/2/001001", "C/2/002000", "C/2/002001",
        ],
    )