# %%
import os
import time
from pathlib import Path
import napari
import numpy as np
from biahub.visualize.animation_utils import (
    ElementPosition,
    add_scale_bar,
    add_text_overlay,
    simple_recording,
)
from iohub import open_ome_zarr
from skimage.morphology import disk, erosion
from tqdm import tqdm
from waveorder.focus import focus_from_transverse_band
import random
import dask.array as da
import imageio.v2 as imageio

import os
NA_DET = 1.35
LAMBDA_ILL = 0.500

from ultrack.reader.napari_reader import read_csv


def clean_viewer(viewer, text_overlay=None, text_overlay_event=None):
    """Properly cleans up a Napari viewer instance to avoid layer conflicts."""
    if viewer is not None:
        print(f"Cleaning up viewer with {len(viewer.layers)} layers...")

        # Disconnect the text overlay event before removing it
        if text_overlay is not None and text_overlay in viewer.layers:
            try:
                if text_overlay_event:
                    viewer.dims.events.current_step.disconnect(text_overlay_event)
            except Exception as e:
                print(f"Warning: Failed to disconnect text overlay event. Error: {e}")

            viewer.layers.remove(text_overlay)
            time.sleep(1)  # Allow Napari to process updates

        # Remove all remaining layers safely
        while len(viewer.layers) > 0:
            layer = viewer.layers[-1]
            viewer.layers.remove(layer)

        # Close the viewer properly
        viewer.close()


def get_segments_bourder(segments, T, erosion_radius=4):
    # Create an empty array for contours (same shape as labels)
    border_labels = np.zeros_like(segments, dtype=int)
    erosion_kernel = disk(erosion_radius)
    # Erode and subtract to find the borders
    for t in range(T):
        for label in np.unique(segments[t]):
            if label == 0:
                continue  # Skip background

            # Create a binary mask for this label
            binary_mask = segments[t] == label

            # Erode the binary mask
            eroded_mask = erosion(binary_mask, erosion_kernel)

            # Subtract: Original - Eroded = Borders
            border_mask = binary_mask ^ eroded_mask  # XOR to get the difference

            # Store only the border pixels with the original label value
            border_labels[t][border_mask] = label
    return border_labels


def get_tracking_phase_viewer(
    im_arr, z_focal_plane, yx_scale, track_labels_borders, track_df, track_graph):

    #im_arr_focus = im_arr[:, z_focal_plane, :, :]



    print("Calculating contrast limits...")
    vmin, vmax = np.percentile(im_arr, (2, 98))

    viewer = napari.Viewer()
    kwargs = dict(scale=yx_scale, blending="additive")
    viewer.add_image(im_arr, name="Phase", contrast_limits=(vmin, vmax), **kwargs)
    viewer.add_labels(track_labels_borders, name="Track Labels", **kwargs)
    viewer.add_tracks(
        track_df[["track_id", "t", "y", "x"]],
        graph=track_graph,
        name="Tracks",
        scale=yx_scale,
        colormap="hsv",
        blending="opaque",
    )
    viewer.dims.set_point(0, 0)

    return viewer


def get_tracking_vs_viewer(
    nuc_arr, mem_arr, yx_scale, track_labels_borders, track_df, track_graph
):

    viewer = napari.Viewer()
    kwargs = dict(scale=yx_scale, blending="additive")
    viewer.add_image(nuc_arr, name="VS nuclei", colormap="green", **kwargs)
    viewer.add_image(mem_arr, name="VS membrane", colormap="magenta", **kwargs)
    viewer.add_labels(track_labels_borders, name="Track Labels", **kwargs)
    viewer.add_tracks(
        track_df[["track_id", "t", "y", "x"]],
        graph=track_graph,
        name="Tracks",
        scale=yx_scale,
        colormap="hsv",
        blending="opaque",
    )
    viewer.dims.set_point(0, 0)

    return viewer


def get_tracking_vs_viewer_4D(nuc_arr, mem_arr, track_labels_borders, zyx_scale):

    z_shape = nuc_arr.shape[1]

    # Convert (T, Y, X) → (T, Z, Y, X) using np.tile()
    border_labels_4D = np.tile(
        track_labels_borders[:, np.newaxis, :, :], (1, z_shape, 1, 1)
    )

    viewer = napari.Viewer()

    kwargs = dict(scale=zyx_scale, blending="additive")
    viewer.add_image(nuc_arr, name="VS nuclei", colormap="green", **kwargs)
    viewer.add_image(mem_arr, name="VS membrane", colormap="magenta", **kwargs)
    viewer.add_labels(border_labels_4D, name="Track Labels", **kwargs)
    viewer.dims.set_point(0, 0)


    return viewer


def get_virtual_staining_viewer(nuc_arr, mem_arr, zyx_scale):

    # Get 3D slices for the visualization (T, Y, X)
    viewer = napari.Viewer()

    kwargs = dict(scale=zyx_scale, blending="additive")
    viewer.add_image(nuc_arr, name="VS nuclei", colormap="green", **kwargs)
    viewer.add_image(mem_arr, name="VS membrane", colormap="magenta", **kwargs)

    return viewer


def get_phase_viewer(im_arr, zyx_scale):

    print("Calculating contrast limits...") 
    viewer = napari.Viewer()

    kwargs = dict(scale=zyx_scale, blending="additive")
    vmin, vmax = np.percentile(im_arr, (2, 98))
    viewer.add_image(im_arr, name="Phase", **kwargs, contrast_limits=(vmin, vmax))

    return viewer


def get_fluorescent_viewer(gfp_arr, mCherry_arr, zyx_scale):

    print("Calculating contrast limits...")
    gfp_vmin = max(0, np.percentile(gfp_arr, 20))
    gfp_vmax = np.percentile(gfp_arr, 99.8)
    print("GFP:", gfp_vmin, gfp_vmax)

    mCherry_vmin = max(0, np.percentile(mCherry_arr, 90))
    mCherry_vmax = np.percentile(mCherry_arr, 99.8)
    print("mCherry:", mCherry_vmin, mCherry_vmax)

    viewer = napari.Viewer()
    kwargs = dict(scale=zyx_scale, blending="additive")

    viewer.add_image(
        gfp_arr,
        name="GFP EX488 EM525-45",
        **kwargs,
        colormap="green",
        contrast_limits=(gfp_vmin, gfp_vmax),
    )
    viewer.add_image(
        mCherry_arr,
        name="mCherry EX561 EM600-37",
        **kwargs,
        colormap="magenta",
        contrast_limits=(mCherry_vmin, mCherry_vmax),
    )

    return viewer


def get_videos_per_fov(im_path, track_path, fov, z_slice, output_path):

    im_ds = open_ome_zarr(im_path / fov)
    print("Dataset shape:", im_ds.data.shape)

    channel_names = im_ds.channel_names
    print("Channels:", channel_names)
    


    print("Reading Phase3D.....")
    im_arr_z = im_ds.data[-1, channel_names.index("Phase3D"), :, :, :]
    print("Reading nuclei_prediction...")
    nuc_arr_z = im_ds.data[-1, channel_names.index("nuclei_prediction"), :, :, :]
    print("Reading membrane_prediction...")
    mem_arr_z = im_ds.data[-1, channel_names.index("membrane_prediction"), :, :, :]
    

    gfp_arr_z = im_ds.data[-1, channel_names.index("GFP EX488 EM525-45"), :, :, :]
    print("Reading mCherry EX561 EM600-37...")
    mcherry_arr_z = im_ds.data[-1, channel_names.index("mCherry EX561 EM600-37"), :, :, :]

    # T, _, _, _ = im_arr.shape

    _, _, Z_scale, Y_scale, X_scale = im_ds.scale
    zyx_scale = [Z_scale, Y_scale, X_scale]
    yx_scale = [Y_scale, X_scale]

    # tracking viewer
    print("Reading tracking...")
    track_seg = open_ome_zarr(track_path / fov)
    tracks_labels = track_seg.data[:, 0, 0, :, :]

    # Create an empty array for contours (same shape as labels)
    track_labels_borders = get_segments_bourder(
        tracks_labels, tracks_labels.shape[0], erosion_radius=4
    )

    w1, w2, f = fov.split("/")
    df_path = track_path / f"{w1}" / f"{w2}" / f"{f}" / f"tracks_{w1}_{w2}_{f}.csv"
    track_df, kwargs, *_ = read_csv(df_path)
    track_graph = kwargs["graph"]

    print("Calculating focal plane...")
    z_focal_plane =  focus_from_transverse_band(
        gfp_arr_z,
        NA_det=NA_DET,
        lambda_ill=LAMBDA_ILL,
        pixel_size=X_scale,
    )
    gfp_arr_t = im_ds.data[:, channel_names.index("GFP EX488 EM525-45"), z_focal_plane, :, :]
    mcherry_arr_t = im_ds.data[:, channel_names.index("mCherry EX561 EM600-37"), z_focal_plane, :, :]
    im_arr_t = im_ds.data[:, channel_names.index("Phase3D"), z_focal_plane, :, :]
    nuc_arr_t = im_ds.data[:, channel_names.index("nuclei_prediction"), z_focal_plane, :, :]
    mem_arr_t = im_ds.data[:, channel_names.index("membrane_prediction"), z_focal_plane, :, :]

    print("Z focal plane for GFP:", z_focal_plane)

    # ----------------- Phase Movie -----------------
    # Generate Phase Movie Over Time (T)
    print("Getting phase mov in T...")
    viewer = get_phase_viewer(im_arr_t, zyx_scale)
    text_overlay, text_overlay_event = add_text_overlay(viewer, time_axis=0, position=ElementPosition.BOTTOM_LEFT, text_size=30, color="red")

    phase_mov_path_t = output_path / "phase_T.mp4"
    if phase_mov_path_t.exists():
        os.remove(phase_mov_path_t)

    simple_recording(viewer, output_path=phase_mov_path_t, loop_axes=[(0, (None, None), 5)], fps=30)
    clean_viewer(viewer, text_overlay, text_overlay_event)

    # Generate Phase Movie Over Depth (Z)
    print("Getting phase mov in Z...")
    viewer = get_phase_viewer(im_arr_z, zyx_scale)
    text_overlay, text_overlay_event = add_text_overlay(viewer, z_axis=0, position=ElementPosition.BOTTOM_LEFT, text_size=30, color="red")

    phase_mov_path_z = output_path / "phase_Z.mp4"
    if phase_mov_path_z.exists():
        os.remove(phase_mov_path_z)

    simple_recording(viewer, output_path=phase_mov_path_z, loop_axes=[(0, (None, None), 5)], fps=30)
    clean_viewer(viewer, text_overlay, text_overlay_event)

    # Merge Phase Movies (T + Z)
    reader1 = imageio.get_reader(phase_mov_path_t)
    reader2 = imageio.get_reader(phase_mov_path_z)
    writer = imageio.get_writer(output_path / "phase_T_Z.mp4", fps=30)

    for frame in reader1:
        writer.append_data(frame)
    for frame in reader2:
        writer.append_data(frame)

    writer.close()


    # ----------------- Fluorescent Movie -----------------
    print("Getting fluorescent mov in T...")
    viewer = get_fluorescent_viewer(gfp_arr_t, mcherry_arr_t, zyx_scale)

    text_overlay, text_overlay_event = add_text_overlay(
        viewer,
        time_axis=0,
        position=ElementPosition.BOTTOM_LEFT,
        text_size=30,
        color="white",
    )

    fluorescent_mov_path = output_path / "fluorescent_T.mp4"
    if fluorescent_mov_path.exists():
        os.remove(fluorescent_mov_path)

    simple_recording(
        viewer,
        output_path=fluorescent_mov_path,
        loop_axes=[(0, (None, None), 5)],
        fps=30,
    )
    

    clean_viewer(viewer, text_overlay, text_overlay_event)

    print("Getting fluorescent mov in Z...")
    viewer = get_fluorescent_viewer(gfp_arr_z, mcherry_arr_z, zyx_scale)

    text_overlay, text_overlay_event = add_text_overlay(
        viewer,
        z_axis=0,
        position=ElementPosition.BOTTOM_LEFT,
        text_size=30,
        color="white",
    )

    fluorescent_mov_path = output_path / "fluorescent_Z.mp4"
    if fluorescent_mov_path.exists():
        os.remove(fluorescent_mov_path)

    simple_recording(
        viewer,
        output_path=fluorescent_mov_path,
        loop_axes=[(0, (None, None), 5)],
        fps=30,
    )
    
    clean_viewer(viewer, text_overlay, text_overlay_event)

    # Read videos
    reader1 = imageio.get_reader(output_path / "fluorescent_T.mp4")
    reader2 = imageio.get_reader(output_path / "fluorescent_Z.mp4")

    # Get metadata from the first video
    fps = reader1.get_meta_data()['fps']
    size = reader1.get_meta_data()['size']  # (width, height)

    # Create output video writer
    writer = imageio.get_writer(output_path / "fluorescent_T_Z.mp4", fps=fps)

    # Write frames from first video
    for frame in reader1:
        writer.append_data(frame)

    # Write frames from second video
    for frame in reader2:
        writer.append_data(frame)

    # Close the writer
    writer.close()


   # Generate Tracking with Virtual Staining Movie Over Time (T)
    # print("Getting tracking with virtual staining mov in T...")
    # viewer = get_tracking_vs_viewer_4D(nuc_arr_t, mem_arr_t, track_labels_borders, zyx_scale)
    # text_overlay, text_overlay_event = add_text_overlay(viewer, time_axis=0, position=ElementPosition.BOTTOM_LEFT, text_size=30, color="white")

    # tracking_vs_mov_path_t = output_path / "tracking_vs_T.mp4"
    # if tracking_vs_mov_path_t.exists():
    #     os.remove(tracking_vs_mov_path_t)

    # simple_recording(viewer, output_path=tracking_vs_mov_path_t, loop_axes=[(0, (None, None), 5)], fps=30)
    # clean_viewer(viewer, text_overlay, text_overlay_event)

    # print("Getting tracking with virtual staining mov in Z...")
    # viewer = get_tracking_vs_viewer_4D(nuc_arr_z, mem_arr_z, track_labels_borders, zyx_scale)
    # text_overlay, text_overlay_event = add_text_overlay(viewer, z_axis=0, position=ElementPosition.BOTTOM_LEFT, text_size=30, color="white")

    # tracking_vs_mov_path_z = output_path / "tracking_vs_Z.mp4"
    # if tracking_vs_mov_path_z.exists():
    #     os.remove(tracking_vs_mov_path_z)

    # simple_recording(viewer, output_path=tracking_vs_mov_path_z, loop_axes=[(0, (None, None), 5)], fps=30)
    # clean_viewer(viewer, text_overlay, text_overlay_event)

    # # Merge Tracking with Virtual Staining Movies (T + Z)
    # reader1 = imageio.get_reader(tracking_vs_mov_path_t)
    # reader2 = imageio.get_reader(tracking_vs_mov_path_z)
    # writer = imageio.get_writer(output_path / "tracking_vs_T_Z.mp4", fps=30)

    # for frame in reader1:
    #     writer.append_data(frame)
    # for frame in reader2:
    #     writer.append_data(frame)

    # writer.close()


    # ----------------- Virtual Staining Movie -----------------
   # Generate Virtual Staining Movie Over Time (T)
    print("Getting virtual staining mov in T...")
    viewer = get_virtual_staining_viewer(nuc_arr_t, mem_arr_t, zyx_scale)
    text_overlay, text_overlay_event = add_text_overlay(viewer, time_axis=0, position=ElementPosition.BOTTOM_LEFT, text_size=30, color="white")

    virtual_staining_mov_path_t = output_path / "virtual_staining_T.mp4"
    if virtual_staining_mov_path_t.exists():
        os.remove(virtual_staining_mov_path_t)

    simple_recording(viewer, output_path=virtual_staining_mov_path_t, loop_axes=[(0, (None, None), 5)], fps=30)
    clean_viewer(viewer, text_overlay, text_overlay_event)

    # Generate Virtual Staining Movie Over Depth (Z)
    print("Getting virtual staining mov in Z...")
    viewer = get_virtual_staining_viewer(nuc_arr_z, mem_arr_z, zyx_scale)
    text_overlay, text_overlay_event = add_text_overlay(viewer, z_axis=0, position=ElementPosition.BOTTOM_LEFT, text_size=30, color="white")

    virtual_staining_mov_path_z = output_path / "virtual_staining_Z.mp4"
    if virtual_staining_mov_path_z.exists():
        os.remove(virtual_staining_mov_path_z)

    simple_recording(viewer, output_path=virtual_staining_mov_path_z, loop_axes=[(0, (None, None), 5)], fps=30)
    clean_viewer(viewer, text_overlay, text_overlay_event)

    # Merge Virtual Staining Movies (T + Z)
    reader1 = imageio.get_reader(virtual_staining_mov_path_t)
    reader2 = imageio.get_reader(virtual_staining_mov_path_z)
    writer = imageio.get_writer(output_path / "virtual_staining_T_Z.mp4", fps=30)

    for frame in reader1:
        writer.append_data(frame)
    for frame in reader2:
        writer.append_data(frame)

    writer.close()


    # ----------------- Tracking Phase Movie -----------------

    print("Getting tracking with phase mov in T...")
    viewer = get_tracking_phase_viewer(
        im_arr_t, z_focal_plane, yx_scale, track_labels_borders, track_df, track_graph
    )
    text_overlay, text_overlay_event = add_text_overlay(
        viewer,
        time_axis=0,  # First dimension is time
        position=ElementPosition.BOTTOM_LEFT,
        text_size=30,
        color="black",
    )
    track_phase_output = output_path / "tracking_phase_T.mp4"
    if track_phase_output.exists():
        os.remove(track_phase_output)
    simple_recording(
        viewer,
        output_path=track_phase_output,
        loop_axes=[
            (0, (None, None), 5),  # Time axis
        ],
        fps=30,
    )

    clean_viewer(viewer, text_overlay, text_overlay_event)

    # ----------------- Tracking Virtual Staining Movie -----------------
    print("Getting tracking with virtual staining mov in T...")

    nuc_arr_focus =  im_ds.data[:, channel_names.index("nuclei_prediction"), z_slice,:, :].max(axis=1)
    mem_arr_focus = im_ds.data[:, channel_names.index("membrane_prediction"), z_slice,:, :].max(axis=1)
   
    viewer = get_tracking_vs_viewer(
        nuc_arr_focus,
        mem_arr_focus,
        yx_scale,
        track_labels_borders,
        track_df,
        track_graph,
    )
    track_vs_output = output_path / "tracking_vs_T.mp4"
    text_overlay, text_overlay_event = add_text_overlay(
        viewer,
        time_axis=0,
        position=ElementPosition.BOTTOM_LEFT,
        text_size=30,
        color="white",
    )
    if track_vs_output.exists():
        os.remove(track_vs_output)
    simple_recording(
        viewer, output_path=track_vs_output, loop_axes=[(0, (None, None), 5)], fps=30
    )
    clean_viewer(viewer, text_overlay, text_overlay_event)

    print("done!")

def get_videos_beads_fov(im_path, fov, output_path):
    im_ds = open_ome_zarr(im_path / fov)
    print("Dataset shape:", im_ds.data.shape)

    channel_names = im_ds.channel_names
    print("Channels:", channel_names)

    print("Reading Phase3D.....")
    im_arr = im_ds.data[:, channel_names.index("Phase3D"), :, :, :]
    print("Reading nuclei_prediction...")
    nuc_arr = im_ds.data[:, channel_names.index("nuclei_prediction"), :, :, :]
    print("Reading membrane_prediction...")
    mem_arr = im_ds.data[:, channel_names.index("membrane_prediction"), :, :, :]
    print("Reading GFP EX488 EM525-45...")
    gfp_arr = im_ds.data[:, channel_names.index("GFP EX488 EM525-45"), :, :, :]
    print("Reading mCherry EX561 EM600-37...")
    mCherry_arr = im_ds.data[:, channel_names.index("mCherry EX561 EM600-37"), :, :, :]

    T, _, _, _ = im_arr.shape

    _, _, Z_scale, Y_scale, X_scale = im_ds.scale
    zyx_scale = [Z_scale, Y_scale, X_scale]
    yx_scale = [Y_scale, X_scale]


def main() -> None:
    # NOTE modify these paths to point to the correct zarr files

    datasets =[ "2024_11_21_A549_TOMM20_DENV",
               "2024_12_11_A549_LAMP1_DENV",
                ]
    z_slices = [slice(30, 55), slice(33, 43), slice(28, 50)]

    for dataset, z_slice in zip(datasets, z_slices):
        print("Dataset:", dataset)
        root = Path(f"/hpc/projects/intracellular_dashboard/organelle_dynamics/{dataset}")
        im_path = root / f"2-assemble/{dataset}.zarr"
        track_path = root / f"1-preprocess/label-free/3-track/{dataset}_cropped.zarr"

        positions =[]
        wells_collums = os.listdir(im_path)
        # keep only directories
        wells_collums = [well for well in wells_collums if os.path.isdir(im_path / well)]
        

        for well in wells_collums:
            wells_rows = os.listdir(im_path / well)
            wells_rows = [row for row in wells_rows if os.path.isdir(im_path / well / row)]
            for row in wells_rows:
                random_fov = []
                fovs = os.listdir(im_path / well / row)
                fovs = [fov for fov in fovs if os.path.isdir(im_path / well / row / fov)]

                if len(fovs) > 2:
                    random_fov = random.sample(fovs, 2)
                else:
                    random_fov = fovs
                for fov in random_fov:
                    positions.append(f"{well}/{row}/{fov}")
        
        #save positions in a txt file create and save
        os.makedirs(root / f"3-visualization/movs", exist_ok=True)

        with open(root / f"3-visualization/movs/positions.txt", "a") as f:
            for item in positions:
                f.write("%s\n" % item)
        
        if "C/1/000000" in positions: # beads well
            positions.remove("C/1/000000")

        if dataset == "2024_11_07_A549_SEC61_ZIKV_DENV":
           # remove position with "B/3/" or "B/2/"
            positions = [pos for pos in positions if "B/3/" not in pos and "B/2/" not in pos]
        print("Positions:", positions)
        for fov in positions:
            print("FOV:", fov)

            output_path = root / f"3-visualization/movs/{fov}"
            os.makedirs(output_path, exist_ok=True)

            get_videos_per_fov(im_path, track_path, fov, z_slice, output_path)
        #get_videos_beads_fov(im_path, "C/1/000000", output_path)


# %%
if __name__ == "__main__":
    main()
