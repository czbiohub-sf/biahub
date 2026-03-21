"""Stage 2: Cropping FOVs into a unified plate for dynacell preprocessing."""

import numpy as np
import pandas as pd
from contextlib import ExitStack
from pathlib import Path
from tqdm import tqdm
from iohub import open_ome_zarr


def crop_fov(
    input_zarr_paths: list[Path],
    output_zarr: Path,
    output_plots_dir: Path,
    fov: str,
    global_drop_csv: Path,
    z_final: int = 64,
    T_out: int | None = None,
    Y_out: int | None = None,
    X_out: int | None = None,
    drop_frames: bool = True,
):
    """Stage 2: Crop one FOV and write into the pre-created plate.

    Reads bbox, z_focus from stage 1 CSVs and keep_indices from the
    global drop list.
    Crops to uniform (T_out, C, z_final, Y_out, X_out) with padding.

    Parameters
    ----------
    input_zarr_paths : list of Path
        Paths to input zarr FOV positions (e.g. [lf_zarr/fov, ls_zarr/fov]).
    global_drop_csv : Path
        Path to drop_list_all_fovs.csv (fov, t, reason). Unique t values are dropped.
    drop_frames : bool
        If True (default), drop timepoints listed in global_drop_csv.
        If False, keep all timepoints (no T cropping).
    """
    # Read stage 1 metadata
    z_focus_df = pd.read_csv(output_plots_dir / "z_focus.csv")
    z_focus = z_focus_df["z_focus"].tolist()

    T_total = len(z_focus)
    if drop_frames:
        # Read global drop list -- unique timepoints across all FOVs
        drop_set = set()
        if global_drop_csv.exists() and global_drop_csv.stat().st_size > 0:
            drop_df = pd.read_csv(global_drop_csv)
            if len(drop_df) > 0:
                drop_set = set(drop_df["t"].unique().tolist())
        keep_indices = np.array(sorted(set(range(T_total)) - drop_set), dtype=int)
    else:
        keep_indices = np.arange(T_total)

    summary = pd.read_csv(output_plots_dir / "fov_summary.csv")
    bbox_str = summary["bbox"].iloc[0]
    bbox = [int(x.strip()) for x in bbox_str.strip("[]").split(",")]
    y_min, y_max, x_min, x_max = bbox

    Y_crop = y_max - y_min + 1
    X_crop = x_max - x_min + 1
    z_below = z_final // 3
    z_above = z_final - z_below - 1
    Z_out = z_final

    # Center-crop / center-pad offsets when per-FOV crop differs from plate dims
    y_src_off = max(0, (Y_crop - Y_out) // 2)
    x_src_off = max(0, (X_crop - X_out) // 2)
    y_dst_off = max(0, (Y_out - Y_crop) // 2)
    x_dst_off = max(0, (X_out - X_crop) // 2)
    y_size = min(Y_crop, Y_out)
    x_size = min(X_crop, X_out)

    # Limit to T_out timepoints (min across all FOVs after dropping)
    if T_out is not None and len(keep_indices) > T_out:
        keep_indices = keep_indices[:T_out]

    fov_str = fov

    print(f"\nCropping FOV {fov_str} to {output_zarr}")
    print(f"  FOV crop: ({len(keep_indices)}, ?, {Z_out}, {Y_crop}, {X_crop})")
    print(f"  Plate dims: T={T_out}, Y={Y_out}, X={X_out}")
    if Y_crop != Y_out or X_crop != X_out:
        print(f"  Center-crop src[{y_src_off}:{y_src_off+y_size}, {x_src_off}:{x_src_off+x_size}]"
              f" -> dst[{y_dst_off}:{y_dst_off+y_size}, {x_dst_off}:{x_dst_off+x_size}]")
    print(f"  z_final: {z_final} (1/3 below={z_below}, 2/3 above={z_above})")
    print(f"  Input zarrs: {len(input_zarr_paths)}")

    # Open all input zarrs
    with ExitStack() as stack:
        src_arrays = []
        for zarr_path in input_zarr_paths:
            ds = stack.enter_context(open_ome_zarr(zarr_path))
            src_arrays.append(ds.data.dask_array())

        pos_path = output_zarr / fov_str
        with open_ome_zarr(pos_path, mode="r+") as out_ds:
            out_img = out_ds["0"]

            for t_out, t_in in enumerate(tqdm(keep_indices, desc=f"Cropping {fov_str}")):
                z_center = int(z_focus[t_in])

                c_out = 0
                for src_arr in src_arrays:
                    Z_total = src_arr.shape[2]
                    z_start = max(0, z_center - z_below)
                    z_end = min(Z_total, z_center + z_above + 1)
                    pad_top = max(0, z_below - z_center)

                    for c in range(src_arr.shape[1]):
                        slc = np.asarray(
                            src_arr[t_in, c, z_start:z_end, y_min:y_max+1, x_min:x_max+1]
                        )
                        out_slice = np.zeros((Z_out, Y_out, X_out), dtype=np.float32)
                        z_actual = slc.shape[0]
                        out_slice[
                            pad_top:pad_top + z_actual,
                            y_dst_off:y_dst_off + y_size,
                            x_dst_off:x_dst_off + x_size,
                        ] = slc[:, y_src_off:y_src_off + y_size, x_src_off:x_src_off + x_size]
                        out_img[t_out, c_out] = out_slice
                        c_out += 1

    print(f"Saved cropped FOV {fov_str}")
