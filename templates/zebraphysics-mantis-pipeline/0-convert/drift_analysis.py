# %%
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_sorted_chunk_dirs(base_path: Path, keyword="chunked"):
    """Return chunked_* dirs sorted by their numeric suffix (…_chunked_1, …_chunked_2, …)."""

    def chunk_index(name: str):
        m = re.search(r"_chunked_(\d+)$", name)
        return int(m.group(1)) if m else float("inf")

    dirs = [
        d for d in os.listdir(base_path) if keyword in d and (base_path / d).is_dir()
    ]
    return sorted(dirs, key=chunk_index)


def _parse_fish_pos(filename: str):
    """
    Extract fish_id and position number from names like:
      autotracker_fov_1-Pos000.csv
    Returns tuple (fish_id, pos_number) or None if no match.
    """
    m = re.search(r"_fov_(\d+)-Pos(\d+)\.csv$", filename)
    if not m:
        return None
    return m.group(1), m.group(2)


def create_symlinks(source_path: Path, target_root: Path, dirnames):
    for dirname in dirnames:
        target = target_root / dirname
        if not target.exists():
            os.symlink(source_path / dirname / "logs", target, target_is_directory=True)


def load_and_concatenate_csvs(root: Path, dirnames):
    """
    Reads CSVs from each chunk's logs folder (already symlinked into root/<chunk>),
    builds a dict: positions[full_pos_id] -> list of DataFrames with globally offset TimepointID.
    full_pos_id format: "<fish_id>_<pos_number>", e.g., "1_000".
    """
    positions = {}
    timepoint_offsets = {}

    for dirname in dirnames:
        folder_path = root / dirname
        files_csv = sorted(
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith(".csv") and not f.startswith(".")
        )

        for f_csv in files_csv:
            parsed = _parse_fish_pos(f_csv)
            if not parsed:
                continue  # skip unrelated files
            fish_id, pos_number = parsed
            full_pos_id = f"{fish_id}_{pos_number}"

            if full_pos_id not in positions:
                positions[full_pos_id] = []
                timepoint_offsets[full_pos_id] = 0

            full_path = folder_path / f_csv
            df = pd.read_csv(full_path)

            # Coerce numerics (prevents string arithmetic surprises)
            for c in [
                "TimepointID",
                "StageX",
                "StageY",
                "StageZ",
                "ShiftX",
                "ShiftY",
                "ShiftZ",
            ]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # Make TimepointID globally unique per position using span (collision-proof)
            if "TimepointID" in df.columns:
                pre_min = df["TimepointID"].min()
                pre_max = df["TimepointID"].max()
                df["TimepointID"] += timepoint_offsets[full_pos_id]
                if pd.notna(pre_min) and pd.notna(pre_max):
                    span = int(pre_max - pre_min + 1)
                else:
                    span = int(df["TimepointID"].nunique())
                timepoint_offsets[full_pos_id] += span

            positions[full_pos_id].append(df)

    return positions


def save_concatenated_csvs(positions: dict, root: Path):
    """
    Writes concatenated CSVs per position and returns grouped_fish:
    grouped_fish[fish_id][pos_id] -> concatenated DataFrame
    """
    grouped_fish = {}
    for pos_id, dfs in positions.items():
        concat_df = pd.concat(dfs, ignore_index=True)
        csv_path = root / f"concatenated_position_{pos_id}.csv"
        concat_df.to_csv(csv_path, index=False)
        print(f"Saved concatenated CSV for position {pos_id} to {csv_path}")

        fish_id = pos_id.split("_", 1)[0]  # correct fish grouping
        grouped_fish.setdefault(fish_id, {})[pos_id] = concat_df
    return grouped_fish


def plot_fish_displacement(
    fish_id,
    fish_data,
    output_dir,
    dataset="",
    unit="µm",
    voxel_size=(0.174, 0.1494, 0.1494),
):
    if not fish_data:
        return

    n_positions = len(fish_data)
    fig_disp, axs_disp_xy = plt.subplots(n_positions, 1, figsize=(12, 4 * n_positions))
    fig_drift, axs_drift_xy = plt.subplots(
        n_positions, 1, figsize=(12, 4 * n_positions)
    )

    # Always work with lists
    if n_positions == 1:
        axs_disp_xy = [axs_disp_xy]
        axs_drift_xy = [axs_drift_xy]

    for ax_idx, (pos_id, df) in enumerate(fish_data.items()):
        df = df.copy()

        # Stage displacement relative to t=0
        df_stage = df.groupby("TimepointID", as_index=False)[
            ["StageX", "StageY", "StageZ"]
        ].mean()
        df_stage["DisplacementX"] = df_stage["StageX"] - df_stage["StageX"].iloc[0]
        df_stage["DisplacementY"] = df_stage["StageY"] - df_stage["StageY"].iloc[0]
        df_stage["DisplacementZ"] = df_stage["StageZ"] - df_stage["StageZ"].iloc[0]

        ax_xy = axs_disp_xy[ax_idx]
        ax_z = ax_xy.twinx()  # create twin y-axis for Z

        ax_xy.plot(
            df_stage["TimepointID"], df_stage["DisplacementX"], label=f"ΔX ({unit})"
        )
        ax_xy.plot(
            df_stage["TimepointID"], df_stage["DisplacementY"], label=f"ΔY ({unit})"
        )
        ax_z.plot(
            df_stage["TimepointID"],
            df_stage["DisplacementZ"],
            label=f"ΔZ ({unit})",
            color="tab:red",
        )

        ax_xy.set_ylabel(f"Stage ΔXY ({unit})")
        ax_z.set_ylabel(f"Stage ΔZ ({unit})", color="tab:red")
        ax_xy.set_title(f"Stage Displacement - Position 0_{pos_id}")
        ax_xy.grid(True)
        ax_xy.set_xlim(left=0)

        # Merge legends from both axes
        lines1, labels1 = ax_xy.get_legend_handles_labels()
        lines2, labels2 = ax_z.get_legend_handles_labels()
        ax_xy.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Cumulative drift from ShiftX/Y/Z
        if {"ShiftX", "ShiftY", "ShiftZ"}.issubset(df.columns):
            df_shift = df.groupby("TimepointID", as_index=False)[
                ["ShiftX", "ShiftY", "ShiftZ"]
            ].mean()
            df_shift["CumulativeShiftX"] = df_shift["ShiftX"].cumsum()
            df_shift["CumulativeShiftY"] = df_shift["ShiftY"].cumsum()
            df_shift["CumulativeShiftZ"] = df_shift["ShiftZ"].cumsum()
            df_shift["DriftMagnitudeXYZ"] = np.sqrt(
                df_shift["CumulativeShiftX"] ** 2
                + df_shift["CumulativeShiftY"] ** 2
                + df_shift["CumulativeShiftZ"] ** 2
            )

            ax_xy_drift = axs_drift_xy[ax_idx]
            ax_z_drift = ax_xy_drift.twinx()  # twin axis for Z drift

            ax_xy_drift.plot(
                df_shift["TimepointID"],
                df_shift["CumulativeShiftX"],
                label=f"ΔX ({unit})",
            )
            ax_xy_drift.plot(
                df_shift["TimepointID"],
                df_shift["CumulativeShiftY"],
                label=f"ΔY ({unit})",
            )
            ax_z_drift.plot(
                df_shift["TimepointID"],
                df_shift["CumulativeShiftZ"],
                label=f"ΔZ ({unit})",
                color="tab:red",
            )
            ax_xy_drift.plot(
                df_shift["TimepointID"],
                df_shift["DriftMagnitudeXYZ"],
                label="Total Displacement (XYZ)",
                linestyle="--",
                color="black",
            )

            ax_xy_drift.set_ylabel(f"Shift XY ({unit})")
            ax_z_drift.set_ylabel(f"Shift Z ({unit})", color="tab:red")
            ax_xy_drift.set_title(f"Total Displacement - Position 0_{pos_id}")
            ax_xy_drift.grid(True)
            ax_xy_drift.set_xlim(left=0)

            lines1, labels1 = ax_xy_drift.get_legend_handles_labels()
            lines2, labels2 = ax_z_drift.get_legend_handles_labels()
            ax_xy_drift.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            fig_overlay, axs_overlay = plt.subplots(3, 1, figsize=(12, 4))
            axs_overlay[0].plot(
                df_stage["TimepointID"],
                df_stage["DisplacementX"],
                label=f"Stage ΔX ({unit})",
            )
            axs_overlay[0].plot(
                df_shift["TimepointID"],
                df_shift["CumulativeShiftX"],
                label=f"Total Displacement ΔX ({unit})",
                linestyle="--",
                color="black",
            )
            axs_overlay[0].set_ylabel(f"Stage Δ ({unit})")
            axs_overlay[0].legend(loc="upper right")
            axs_overlay[0].grid(True)
            axs_overlay[0].set_xlim(left=0)
            axs_overlay[1].plot(
                df_stage["TimepointID"],
                df_stage["DisplacementY"],
                label=f"Stage ΔY ({unit})",
            )
            axs_overlay[1].plot(
                df_shift["TimepointID"],
                df_shift["CumulativeShiftY"],
                label=f"Total Displacement ΔY ({unit})",
                linestyle="--",
                color="black",
            )
            axs_overlay[1].set_ylabel(f"Stage Δ ({unit})")
            axs_overlay[1].legend(loc="upper right")
            axs_overlay[1].grid(True)
            axs_overlay[1].set_xlim(left=0)
            axs_overlay[2].plot(
                df_stage["TimepointID"],
                df_stage["DisplacementZ"],
                label=f"Stage ΔZ ({unit})",
            )
            axs_overlay[2].plot(
                df_shift["TimepointID"],
                df_shift["CumulativeShiftZ"],
                label=f"Total Displacement ΔZ ({unit})",
                linestyle="--",
                color="black",
            )
            axs_overlay[2].set_ylabel(f"Stage Δ ({unit})")
            axs_overlay[2].legend(loc="upper right")
            axs_overlay[2].grid(True)
            axs_overlay[2].set_xlim(left=0)
            fig_overlay.tight_layout()
            if fish_id == "8":
                fig_overlay.suptitle(
                    f"Overlay: Stage vs PCC Displacements- Beads", y=1.02, fontsize=16
                )
                fig_overlay.savefig(
                    output_dir / f"{dataset}_beads_stage_vs_drift.png",
                    bbox_inches="tight",
                )
            else:
                fig_overlay.suptitle(
                    f"Overlay: Stage vs PCC Displacements - Fish {fish_id} - Position 0_{pos_id}",
                    y=1.02,
                    fontsize=16,
                )
                fig_overlay.savefig(
                    output_dir / f"{dataset}_fish_{fish_id}_stage_vs_drift.png",
                    bbox_inches="tight",
                )
            plt.close(fig_overlay)

    axs_disp_xy[-1].set_xlabel("Time Point")
    axs_drift_xy[-1].set_xlabel("Time Point")

    fig_disp.tight_layout()
    fig_drift.tight_layout()
    if fish_id == "8":
        fig_disp.suptitle(f"Stage Displacement - Beads", y=1.02, fontsize=16)
        fig_disp.savefig(
            output_dir / f"{dataset}_beads_stage_displacement.png", bbox_inches="tight"
        )

        fig_drift.suptitle(f"PCC Displacements (XYZ) - Beads", y=1.02, fontsize=16)
        fig_drift.savefig(
            output_dir / f"{dataset}_beads_pcc_shift.png", bbox_inches="tight"
        )
    else:
        fig_disp.suptitle(f"Stage Displacement - Fish {fish_id}", y=1.02, fontsize=16)
        fig_disp.savefig(
            output_dir / f"{dataset}_fish_{fish_id}_stage_displacement.png",
            bbox_inches="tight",
        )

        fig_drift.suptitle(
            f"PCC Displacements (XYZ) - Fish {fish_id}", y=1.02, fontsize=16
        )
        fig_drift.savefig(
            output_dir / f"{dataset}_fish_{fish_id}_pcc_shift.png", bbox_inches="tight"
        )

    plt.close(fig_disp)
    plt.close(fig_drift)


def main():
    # get dataset from env
    # Set dataset name in current shell with `export DATASET='dataset_name`
    dataset = os.environ.get("DATASET")
    if dataset is None:
        print("$DATASET environmental variable is not set")
        exit(1)

    # dataset = "2025_08_01_zebrafish_golden_trio"
    path = Path(f"/hpc/instruments/cm.mantis/{dataset}/0-convert")
    root = Path(f"/hpc/projects/tlg2_mantis/{dataset}/0-convert/drift_analysis")
    root.mkdir(parents=True, exist_ok=True)

    dirnames = get_sorted_chunk_dirs(path)
    print("Symlinking:", dirnames)

    create_symlinks(path, root, dirnames)

    output_csv = root / "concatenated_positions"
    os.makedirs(output_csv, exist_ok=True)

    positions = load_and_concatenate_csvs(root, dirnames)
    grouped_fish = save_concatenated_csvs(positions, output_csv)

    # remove symlinks
    print("Removing symlinks in:", root)
    for dirname in dirnames:
        target = root / dirname
        if target.exists():
            target.unlink()

    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True)

    for fish_id, fish_data in grouped_fish.items():
        print(f"Processing fish {fish_id} positions")
        if not fish_data:
            print(f"No data for fish {fish_id}, skipping.")
            continue
        # Plot displacement and drift
        plot_fish_displacement(fish_id, fish_data, output_dir, dataset)


# %%
if __name__ == "__main__":
    main()
