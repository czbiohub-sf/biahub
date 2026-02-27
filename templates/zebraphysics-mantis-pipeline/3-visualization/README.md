# Visualization
You can run both visualization scripts in a conda environment with biahub: https://github.com/czbiohub-sf/biahub and with ultrack: https://github.com/royerlab/ultrack

**a. Vizualize Fluorecent and Phase Channels**
Running the command line
```bash
conda activate biahub
cd 3-visualizationpython visualization fluorecent.py dataset_dirpath fov
```

For example:

```bash
conda activate biahub
cd 3-visualization
python visualization fluorecent.py /hpc/projects/intracellular_dashboard/organelle_dynamics/2024_10_31_A549_SEC61_ZIKV_DENV B/1/000000
```

It will open a napari window with GFP, mCherry, phase . Here is a video of how the layers will appear, and after its open the user can change any config or order of layers. This script is for visualization purposes, it does not save any output. I will also use the napari pluging to generate videos for all datasets.



**b. Visualize Tracking in 2D with Phase and Virtual Stanining Channels**
Running the command line

```bash
conda activate biahub
cd 3-visualization
python visualization_tracks.py dataset_dirpath fov zslices
```

For example:

```bash
conda activate biahub
cd 3-visualization
python visualization_tracks.py /hpc/projects/intracellular_dashboard/organelle_dynamics/2024_10_31_A549_SEC61_ZIKV_DENV B/1/000000 28,38
```

It will open a napari window with phase in the z focal plane of t=0, nuc and mem virtual staning max projected from the z stack of the input slices (recommended to use the ones used for tracking) and the tracks as contour and the tracking lineage. Here is a video of how the layers will appear, and after its open the user can change any config or order of layers. This script is for visualization purposes, it does not save any output. I will also use the napari pluging to generate videos for all datasets.

**c. Generate Videos for FOVs and Wells**

This section documents three Python scripts that automate the creation of .mp4 videos using either 5D OME-Zarr datasets or existing .mp4 inputs. All scripts use YAML configuration files for flexible settings. 

- **FOV-Level Video Generation**

`napari_zarr_to_mp4.py`  script generates .mp4 videos from individual OME-Zarr FOVs . It supports:

- Multiple imaging channels
- Channel-specific contrast limits and colormaps
- Optional Z-slice or maximum projection
- Adjustable frame rate (fps)

*Z-Stack Handling*

- If z_index is specified, the selected slice is used for every frame.
- If z_index is omitted, the script performs a maximum intensity projection over Z per timepoint.


*Config file example:* `3-visualization/napari_zarr_to_mp4.yaml`

```yaml
input_path: "/path/to/zarrs/*/*/*"
output_path: "/path/to/output"
fps: 3
z_index: 66

channels:
  - name: "Phase3D"
    contrast_limits: [100, 500]
    colormap: "gray"
  - name: "GFP EX488 EM525-45"
    contrast_limits: [100, 800]
    colormap: "green"
  - name: "mCherry EX561 EM600-37"
    contrast_limits: [150, 900]
    colormap: "magenta"

```

*Usage*
```bash
conda activate biahub
cd 3-visualization
python napari_zarr_to_mp4.py
```

This script will read the Zarr data and generate one video per FOV in the output directory specified.

- **Well-Level Video Grid Assembly**

 `grid_per_well.py`  script creates grid-layout videos by grouping .mp4 videos of FOVs that belong to the same well. It automatically parses well identifiers from filenames (e.g., B_1_000000.mp4) and creates one output video per well.

*Config file example:* `3-visualization/grid_per_well.yaml`

```yaml
mov_fov_path: /path/to/videos
output_path: /path/to/output
grid: [2, 2]
time_sampling_min: 5
rotation: 90  # good for deskewed ls
max_frames: null       # or 300
frame_step: 5          # skip every 4, keep every 5th
legend: null
legend_font_size: 20
timestamp_font_size: 24
add_legend: true


```

The script assumes that FOV filenames are formatted with well identifiers (e.g. `B_1_000000.mp4`) and automatically groups by well.

All videos must already exist in the mov_fov_path.
*Usage*
```bash
conda activate biahub
cd 3-visualization
python grid_per_well.py
```

It will create grid videos like `B1.mp4`, `C2.mp4`, etc. inside the output_path.


- **Flexible Grid Assembly**

`grid_mov.py` script combines an arbitrary list of .mp4 videos into a single grid video. It’s ideal for combining different channels, views, or processing outputs of a single FOV. You can also batch-generate videos for a list of FOVs by templating paths and legends.

*Key Features*

- Per-video settings: crop, rotation, legend, timestamp, etc.
- Batch mode using `fov_list` for multi-FOV processing
- Ensures output filenames are unique to avoid overwrites

*Config example:* 3-visualization/grid_mov.yaml
```yaml
fov_list:
  - 0/1/000000
  - 0/1/001000
  - 0/1/002000
  - 0/2/000000
  - 0/2/001000
  - 0/2/002000
  - 0/2/003000
  - 0/8/000000

output_dir: /path/to/output/
output_filename: BF_nuc_mem_0_2_002000.mp4
grid: [1, 3]

movs:
  - mov_path: BF/fov/0_2_002000.mp4
    crop_box: [176, 1072, 970, 2166]
    legend: "BF - 0/2/002000"
    legend_font_size: 20
    timestamp_font_size: 20
    rotation: 0
    frame_step: 12
    frame_time_sampling_min: 5

  - mov_path: nuclei_projection/fov/0_2_002000.mp4
    crop_box: [0, 784, 240, 881]
    legend: "Nuclei max projection - 0/2/002000"
    legend_font_size: 20
    timestamp_font_size: 20
    rotation: -90
    frame_step: 12
    frame_time_sampling_min: 5

  - mov_path: membrane_projection/fov/0_2_002000.mp4
    crop_box: [0, 1248, 976, 2168]
    legend: "Membrane max projection - 0/2/002000"
    legend_font_size: 20
    timestamp_font_size: 20
    rotation: -90
    frame_step: 12
    frame_time_sampling_min: 5
```
*Usage*

```bash
conda activate biahub
cd 3-visualization
python grid_mov.py
```

If `fov_list` is present, it will generate one video per FOV by replacing the placeholders in output_filename and mov_path.

- **Summary Table**

| Script Name              | Purpose                                                    | Input            | Output Type            |
|--------------------------|-------------------------------------------------------------|------------------|-------------------------|
| `napari_zarr_to_mp4.py`  | Generate videos from OME-Zarr FOVs                         | OME-Zarr         | `.mp4` per FOV          |
| `grid_per_well.py`       | Combine FOV `.mp4`s into well-level grids                  | FOV `.mp4` files | `.mp4` per well         |
| `grid_mov.py`            | Grid arbitrary video lists (e.g., all channels of one FOV) | `.mp4` list      | `.mp4` per FOV (batched) |
