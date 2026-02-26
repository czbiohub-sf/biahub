# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from iohub import open_ome_zarr
from waveorder.focus import focus_from_transverse_band

# %%  --- CONFIG ---
phase_zarr_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_12_03_A549_LAMP1_ZIKV_DENV/1-preprocess/label-free/0-reconstruct/2024_12_03_A549_LAMP1_ZIKV_DENV.zarr/B/1/000000")
vs_zarr_path = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_12_03_A549_LAMP1_ZIKV_DENV/1-preprocess/label-free/1-virtual-stain/2024_12_03_A549_LAMP1_ZIKV_DENV.zarr/B/1/000000")

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
