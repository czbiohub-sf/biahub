import logging
import shutil

from pathlib import Path

import click
import numpy as np
import pandas as pd
import submitit
import zarr

from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_labels
from numpy.typing import ArrayLike

from biahub.cli.monitor import monitor_jobs
from biahub.cli.parsing import (
    cluster,
    config_filepath,
    init_only,
    input_position_dirpaths,
    monitor,
    output_dirpath,
    sbatch_filepath,
    sbatch_to_submitit,
)
from biahub.settings import (
    HoctConfig,
    ProcessingInputChannel,
    TrackastraConfig,
    TrackingSettings,
    TracksdataConfig,
)
from biahub.tracking.graph import (
    TRACK_COLUMNS as TRACK_COLUMNS,
)
from biahub.tracking.graph import (
    _empty_result as _empty_result,
)
from biahub.tracking.graph import (
    _isolated_tracks as _isolated_tracks,
)
from biahub.tracking.graph import (
    _select_every_detection as _select_every_detection,
)
from biahub.tracking.graph import (
    _solution_to_tracks as _solution_to_tracks,
)
from biahub.tracking.methods.hoct import (
    _import_hoct as _import_hoct,
)
from biahub.tracking.methods.hoct import (
    _run_hoct as _run_hoct,
)
from biahub.tracking.methods.hoct import (
    _select_sparse_links as _select_sparse_links,
)
from biahub.tracking.methods.hoct import (
    hoct_solver_config as hoct_solver_config,
)
from biahub.tracking.methods.hoct import (
    run_hoct as run_hoct,
)
from biahub.tracking.methods.hoct import (
    warm_hoct_weights as warm_hoct_weights,
)
from biahub.tracking.methods.trackastra import (
    _run_trackastra as _run_trackastra,
)
from biahub.tracking.methods.trackastra import (
    run_trackastra as run_trackastra,
)
from biahub.tracking.methods.trackastra import (
    warm_trackastra_weights as warm_trackastra_weights,
)
from biahub.tracking.methods.tracksdata import (
    _num_threads as _num_threads,
)
from biahub.tracking.methods.tracksdata import (
    _solve_tracksdata as _solve_tracksdata,
)
from biahub.tracking.methods.tracksdata import (
    run_tracksdata as run_tracksdata,
)
from biahub.tracking.methods.ultrack import (
    _database_dirpath as _database_dirpath,
)
from biahub.tracking.methods.ultrack import (
    _patch_ultrack_readonly_buffer as _patch_ultrack_readonly_buffer,
)
from biahub.tracking.methods.ultrack import (
    _resolved_ultrack_config as _resolved_ultrack_config,
)
from biahub.tracking.methods.ultrack import (
    _ultrack_patched as _ultrack_patched,
)
from biahub.tracking.methods.ultrack import (
    detect_foreground_segmentation as detect_foreground_segmentation,
)
from biahub.tracking.methods.ultrack import (
    run_ultrack as run_ultrack,
)
from biahub.tracking.postprocess import (
    _merge_single_children as _merge_single_children,
)
from biahub.tracking.postprocess import (
    _upsample_result as _upsample_result,
)

# Moved to biahub.tracking (#449); re-exported so `from biahub.track import ...` keeps working.
from biahub.tracking.postprocess import (
    filter_short_siblings as filter_short_siblings,
)
from biahub.tracking.preprocess import (
    CUSTOM_FUNCTIONS as CUSTOM_FUNCTIONS,
)
from biahub.tracking.preprocess import (
    _downsample as _downsample,
)
from biahub.tracking.preprocess import (
    fill_empty_frames as fill_empty_frames,
)
from biahub.tracking.preprocess import (
    fill_empty_frames_from_csv as fill_empty_frames_from_csv,
)
from biahub.tracking.preprocess import (
    get_empty_frames_idx_from_csv as get_empty_frames_idx_from_csv,
)
from biahub.tracking.preprocess import (
    mem_nuc_contour as mem_nuc_contour,
)
from biahub.tracking.preprocess import (
    run_preprocessing_pipeline as run_preprocessing_pipeline,
)
from biahub.utils.cluster import (
    apply_sbatch_overrides,
    echo_resources,
    estimate_resources,
    get_submitit_cluster,
    local_executor_parameters,
)
from biahub.utils.config import update_model, yaml_to_model
from biahub.utils.ngff import label_layout, labels_exist

logger = logging.getLogger(__name__)

# Provenance key on each label image: the track settings and the resolved Ultrack config.
PROVENANCE_KEY = "biahub-track"

# Lazy imports for ultrack - imported only when needed in specific functions


def check_input_dimensionality(output_mode: str, z_size: int) -> None:
    """Refuse an input whose Z does not match ``output_mode``.

    Track neither projects nor selects Z-planes (that is `biahub project`): 2D tracking
    needs a Z=1 plate and 3D tracking a stack.
    """
    if output_mode == "2D" and z_size > 1:
        raise ValueError(
            f"output_mode is 2D but the input is 3D (Z={z_size}). Project the plate with "
            "`biahub project` and track the projected (Z=1) plate, or set output_mode: 3D."
        )
    if output_mode == "3D" and z_size == 1:
        raise ValueError(
            "output_mode is 3D but the input has one Z-plane; set output_mode: 2D."
        )


def load_data(
    position_key: tuple[str, str, str],
    input_images: list[ProcessingInputChannel],
    visualize: bool = False,
) -> dict[str, ArrayLike]:
    """
    Load and extract specified channels from an OME-Zarr dataset for a given position.

    This function opens the OME-Zarr dataset corresponding to a specific position key,
    extracts the required channels (every Z-plane) as defined in the `input_images`
    configuration, and stores them in a dictionary for further processing.

    Parameters
    ----------
    position_key : tuple of str
        A tuple of three strings representing the hierarchical path to a specific position
        in the dataset (e.g., (plate, well, position)).
    input_images : list of ProcessingInputChannel
        List of input image channel configurations. Each item defines the path and channels to load.
    visualize : bool, optional
        If True, opens a Napari viewer and displays each loaded channel. Default is False.

    Returns
    -------
    dict of str to ArrayLike
        Dictionary mapping each channel name to its corresponding image data array.
        Each array is a Dask array of shape (T, Z, Y, X), extracted from the OME-Zarr
        store, or (T, Y, X) when the store has one Z-plane (a projected plate).

    Notes
    -----
    - Only ``input_images`` entries with a non-null ``path`` are loaded here; entries
      with ``path: null`` are virtual/computed channels produced later by
      ``run_preprocessing_pipeline`` (``track`` resolves the primary null path before
      submitting, see :func:`track`).
    - Channel names must match those present in the dataset.
    """
    import napari

    data_dict = {}
    for image in input_images:
        # load the data from the zarr path
        if image.path is not None:
            image_path = image.path / Path(*position_key)
            with open_ome_zarr(image_path) as dataset:
                image_channel_names = dataset.channel_names
                for channel_name, _ in image.channels.items():
                    click.echo(f"Loading data for channel {channel_name} from {image.path}")
                    tzyx = dataset.data.dask_array()[
                        :, image_channel_names.index(channel_name)
                    ]
                    # A projected (Z=1) plate is a 2D series: drop the singleton axis so
                    # the processing functions see (T, Y, X).
                    data_dict[channel_name] = tzyx[:, 0] if tzyx.shape[1] == 1 else tzyx
            if visualize:
                viewer = napari.Viewer()
                viewer.add_image(data_dict[channel_name], name=channel_name)
    return data_dict


def _load_and_preprocess(
    position_key: str,
    input_images: list[ProcessingInputChannel],
    blank_frames_path: Path | None,
    visualize: bool = False,
) -> dict[str, ArrayLike]:
    """Load, run the processing pipeline, and fill blank frames for one FOV.

    Used by the foreground+contour path (the labels path reads a label image).
    """
    fov = "/".join(position_key)
    data_dict = load_data(
        position_key=position_key, input_images=input_images, visualize=visualize
    )
    data_dict = run_preprocessing_pipeline(data_dict, input_images, visualize=visualize)
    return fill_empty_frames_from_csv(fov, data_dict, blank_frames_path)


def load_labels(
    position_dirpath: Path,
    label_name: str,
    output_mode: str,
    blank_frames_path: Path | None = None,
) -> np.ndarray:
    """Read the instance labels ``labels/<label_name>`` of one position.

    The label image is (T, 1, Z, Y, X), the image's axes with one channel. Returns
    (T, Y, X) labels for 2D tracking (the single plane of a projected plate) and
    (T, Z, Y, X) for 3D. Frames listed as blank for this FOV in ``blank_frames_path``
    are filled with the nearest non-blank frame's labels, as the images used to be
    before segmentation moved out of track, so a dropped timepoint does not break
    every track.
    """
    with open_ome_zarr(str(position_dirpath), mode="r") as position:
        if label_name not in position.label_names():
            raise ValueError(
                f"{position_dirpath} has no label image {label_name!r} (labels: "
                f"{position.label_names()}); segment the plate with `biahub segment` first."
            )
        labels = np.asarray(position.get_label(label_name)["0"][:, 0])  # (T, Z, Y, X)
    if output_mode == "2D":
        labels = labels[:, 0]
    if blank_frames_path:
        fov = "/".join(Path(position_dirpath).parts[-3:])
        empty_frames_idx = get_empty_frames_idx_from_csv(pd.read_csv(blank_frames_path), fov)
        if empty_frames_idx:
            labels = fill_empty_frames(labels, empty_frames_idx)
    n_cells = [len(np.unique(frame)) - 1 for frame in labels]
    click.echo(
        f"Labels {label_name!r}: cells per frame mean={np.mean(n_cells):.1f}, "
        f"min={np.min(n_cells)}, max={np.max(n_cells)}"
    )
    return labels


# The provenance key `biahub segment` records on its label images (biahub.segment).
SEGMENT_PROVENANCE_KEY = "biahub-segment"


def tracker_image_channel(
    position_dirpath: Path, label_name: str, image_channel: str | None = None
) -> str:
    """Return the image channel a label-linking tracker (hoct, trackastra) reads.

    ``image_channel`` when set; otherwise the first channel the model that wrote
    ``labels/<label_name>`` segmented, from that label image's ``biahub-segment``
    provenance, so the tracker sees what cellpose saw.
    """
    if image_channel is not None:
        return image_channel
    with open_ome_zarr(str(position_dirpath), mode="r") as position:
        attrs = dict(position.get_label(label_name).zattrs)
    model = attrs.get(SEGMENT_PROVENANCE_KEY, {}).get("models", {}).get(label_name) or {}
    channels = model.get("channels")
    if not channels:
        raise ValueError(
            f"labels/{label_name} records no `{SEGMENT_PROVENANCE_KEY}` channels (it was not "
            "written by `biahub segment`); set image_channel in the tracker's config."
        )
    if len(channels) > 1:
        click.echo(
            f"labels/{label_name} was segmented from {channels}; the tracker reads the "
            f"first, {channels[0]!r} (set image_channel to choose another)."
        )
    return channels[0]


def load_tracker_image(
    position_dirpath: Path, channel: str, blank_frames_path: Path | None = None
) -> np.ndarray:
    """Read the (T, Y, X) image of ``channel`` from a projected (Z=1) position.

    Frames listed as blank for this FOV are filled like the labels (:func:`load_labels`),
    so the image and the labels stay frame-aligned.
    """
    with open_ome_zarr(str(position_dirpath), mode="r") as position:
        if channel not in position.channel_names:
            raise ValueError(
                f"{position_dirpath} has no channel {channel!r} (channels: "
                f"{position.channel_names}); set image_channel to one of them."
            )
        image = np.asarray(position["0"][:, position.channel_names.index(channel), 0])
    if blank_frames_path:
        fov = "/".join(Path(position_dirpath).parts[-3:])
        empty_frames_idx = get_empty_frames_idx_from_csv(pd.read_csv(blank_frames_path), fov)
        if empty_frames_idx:
            image = fill_empty_frames(image, empty_frames_idx)
    return image


def link_fixed_labels(
    tracker: str,
    position_dirpath: Path,
    label_name: str,
    blank_frames_path: Path | None,
    pixel_size: float,
    tracksdata_config: TracksdataConfig,
    hoct_config: HoctConfig,
    trackastra_config: TrackastraConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Link the (T, Y, X) instance labels of ``labels/<label_name>`` with ``tracker``.

    tracksdata, hoct and trackastra link the labels as they are (2D only); hoct and
    trackastra also read the image the labels were segmented from (see
    :func:`tracker_image_channel`). Returns the labels painted with track ids and the
    tracks table (``track_id, t, y, x, parent_track_id``).
    """
    labels = load_labels(position_dirpath, label_name, "2D", blank_frames_path)
    if tracker == "tracksdata":
        click.echo(f"Tracking the labels of labels/{label_name} with tracksdata...")
        return run_tracksdata(labels, tracksdata_config, pixel_size)
    config = hoct_config if tracker == "hoct" else trackastra_config
    channel = tracker_image_channel(position_dirpath, label_name, config.image_channel)
    images = load_tracker_image(position_dirpath, channel, blank_frames_path)
    click.echo(
        f"Tracking the labels of labels/{label_name} with {tracker} (image {channel!r})..."
    )
    if tracker == "hoct":
        return run_hoct(labels, images, hoct_config, pixel_size)
    return run_trackastra(labels, images, trackastra_config)


def write_tracking_labels(label_array, labels: ArrayLike, output_mode: str) -> None:
    """Write tracked labels into the (T, 1, Z, Y, X) label array of the plate track read.

    The label image has the image's axes with one channel and the plate's Z: one
    plane for a projected plate, where 2D tracking writes its (T, Y, X) labels; the
    full stack for 3D tracking, whose (T, Z, Y, X) labels fill it. The labels are
    stored in the label image's dtype (the source label's, see ``label_layout``);
    track ids beyond it are refused rather than wrapped.
    """
    labels = np.asarray(labels)
    dtype = np.dtype(label_array.dtype)
    if labels.size and labels.max() > np.iinfo(dtype).max:
        raise ValueError(
            f"Track ids reach {int(labels.max())}, more than the label image's {dtype} holds "
            f"({np.iinfo(dtype).max}); segment with a wider label dtype."
        )
    labels = labels.astype(dtype, copy=False)
    if output_mode == "2D":
        # A single-plane series that kept its Z axis.
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 3:
            raise ValueError(
                f"output_mode='2D' expects (T, Y, X) labels but tracking produced "
                f"shape {labels.shape}."
            )
        label_array[:, 0, 0] = labels
        return
    if labels.ndim != 4 or labels.shape[1] != label_array.shape[2]:
        raise ValueError(
            f"output_mode='3D' expects (T, {label_array.shape[2]}, Y, X) labels but "
            f"tracking produced shape {labels.shape}."
        )
    label_array[:, 0] = labels


def track_one_position(
    position_key: tuple[str, str, str],
    position_dirpath: Path,
    input_images: list[ProcessingInputChannel],
    output_dirpath: Path | None,
    tracking_config,
    label_name: str,
    blank_frames_path: Path | None = None,
    scale: tuple[float, ...] = (1, 1, 1, 1, 1),
    input_label: str | None = None,
    labels_sigma: float = 5.0,
    output_mode: str = "2D",
    min_sibling_length: int | None = None,
    tracks_stem: str = "tracks",
    tracker: str = "ultrack",
    tracksdata_config: TracksdataConfig | None = None,
    hoct_config: HoctConfig | None = None,
    trackastra_config: TrackastraConfig | None = None,
) -> None:
    """
    Run tracking on a single field of view.

    Supports two segmentation inputs:
    - **labels** (``input_label`` set): the instance labels of ``labels/<input_label>``
      in the position (written by `biahub segment`) are handed to ultrack, which builds
      its own segmentation hypotheses from their contours (smoothed by ``labels_sigma``).
    - **foreground+contour** (legacy): the preprocessing pipeline must produce a binary
      foreground mask and a corresponding contour map, which are handed to ultrack.

    Parameters
    ----------
    position_key : tuple of str
        (row, column, fov) of the field of view.
    position_dirpath : Path
        The ``-i`` position the outputs are written into. ``track --init`` has already
        created its empty ``labels/<label_name>`` image.
    input_images : list of ProcessingInputChannel
        Configuration describing which channels to load and how to preprocess them
        (foreground+contour); may be empty with ``input_label``.
    output_dirpath : Path or None
        Where to keep the Ultrack database and config for this FOV, in
        ``{output_dirpath}/{row}_{col}_{fov}/``. If None they live in a scratch
        directory in this worker's ``$TMPDIR`` and are deleted after export (see
        :func:`_database_dirpath`).
    tracking_config : MainConfig
        Ultrack configuration containing segmentation, linking, and optimization settings.
    label_name : str
        Name of the label image under ``{position_dirpath}/labels/``.
    blank_frames_path : Path, optional
        Path to CSV file indicating empty frames for the current FOV. If None, blank frame
        filling is skipped.
    scale : tuple of float, optional
        Physical scale used for tracking, in (Z, Y, X) or (Y, X) order (already trimmed
        by the caller to match ``output_mode``). Default is (1, 1, 1, 1, 1).
    input_label : str, optional
        Name of the label image to track, ``labels/<input_label>`` in the position. If
        None, the foreground+contour path is used.
    labels_sigma : float, optional
        Smoothing of the labels' contours before ultrack re-segments them (ultrack's
        ``labels_to_contours`` sigma). Default is 5.0.
    output_mode : str, optional
        "2D" tracks a Z=1 input and writes (T, Y, X) labels into its one plane; "3D"
        tracks the stack and writes (T, Z, Y, X) labels. Track does not project (see
        `biahub project`). Default is "2D".
    min_sibling_length : int, optional
        If set, remove daughter tracks of this many frames or fewer before writing
        (see :func:`filter_short_siblings`); the labels, the GEFF and the CSV all
        come from the filtered tracks. Default None keeps every division.
    tracks_stem : str, optional
        Stem of the GEFF and CSV file names. Default ``"tracks"``.
    tracker : str, optional
        ``"ultrack"`` (default) or a tracker that links ``input_label``'s labels as they
        are: ``"tracksdata"``, ``"hoct"`` or ``"trackastra"`` (see
        :func:`link_fixed_labels`). Those write the labels and the CSV but no GEFF yet:
        ultrack's export reads its own database.
    tracksdata_config, hoct_config, trackastra_config : optional
        The selected tracker's settings (defaults when None).

    Returns
    -------
    None
        Outputs are saved into ``position_dirpath``:
        - ``labels/<label_name>``: the tracked labels, painted with ``track_id``
        - ``<tracks_stem>.geff``: the track graph, linked to the labels by ``seg_id``
        - ``<tracks_stem>_<row>_<col>_<fov>.csv``: the same tracks as a CSV, kept for now
        ``tracks_stem`` is ``tracks`` unless the config names the tracked label itself
        (``output_label``), when it is ``tracks_<output_label>``.

    Notes
    -----
    - In foreground+contour mode, missing "foreground"/"contour" channels raise a ValueError.
    """
    position_dirpath = Path(position_dirpath)
    fov = "_".join(position_key)
    click.echo(f"Processing FOV: {fov.replace('_', '/')}")

    if tracker != "ultrack":
        tracking_labels, tracks_df = link_fixed_labels(
            tracker,
            position_dirpath,
            input_label,
            blank_frames_path,
            pixel_size=scale[-1],
            tracksdata_config=tracksdata_config or TracksdataConfig(),
            hoct_config=hoct_config or HoctConfig(),
            trackastra_config=trackastra_config or TrackastraConfig(),
        )
        if min_sibling_length is not None and len(tracks_df):
            tracks_df, tracking_labels = filter_short_siblings(
                tracks_df, tracking_labels, min_sibling_length
            )
        # No GEFF for these trackers yet (ultrack's export reads its own database):
        # remove one an earlier ultrack run left, so it cannot disagree with the labels.
        stale_geff = position_dirpath / f"{tracks_stem}.geff"
        if stale_geff.exists():
            shutil.rmtree(stale_geff)
            click.echo(
                f"Removed {stale_geff.name}, left by an earlier run: {tracker} writes none."
            )
        tracks_df.to_csv(position_dirpath / f"{tracks_stem}_{fov}.csv", index=False)
        with open_ome_zarr(position_dirpath, mode="r+") as position:
            write_tracking_labels(position.get_label(label_name)["0"], tracking_labels, "2D")
        click.echo(f"Saved labels and tracks to: {position_dirpath}")
        return

    with open_ome_zarr(position_dirpath, mode="r") as position:
        time_scale = position.scale[0]
        units = {axis.name.lower(): getattr(axis, "unit", None) for axis in position.axes}

    with _database_dirpath(output_dirpath, fov) as database_path:
        geff_kwargs = {
            "zarr_format": 3,
            "include_masks": False,
            "include_overlaps": False,
            "segmentation_path": f"../labels/{label_name}",
            "scale": tuple(scale),
            "spatial_unit": units.get("x"),
            "time_scale": time_scale if units.get("t") else None,
            "time_unit": units.get("t"),
        }
        if input_label is not None:
            labels = load_labels(position_dirpath, input_label, output_mode, blank_frames_path)
            detection_kwargs = {"labels": labels, "sigma": labels_sigma}
            click.echo(f"Tracking the labels of labels/{input_label}...")
        else:
            # Load, run the processing pipeline, and fill blank frames.
            data_dict = _load_and_preprocess(position_key, input_images, blank_frames_path)
            foreground_mask, contour_gradient_map = detect_foreground_segmentation(data_dict)
            detection_kwargs = {"detection": foreground_mask, "edges": contour_gradient_map}
            click.echo("Tracking with foreground + contour...")

        tracking_labels, tracks_df, _ = run_ultrack(
            tracking_config=tracking_config,
            database_path=database_path,
            geff_path=position_dirpath / f"{tracks_stem}.geff",
            geff_kwargs=geff_kwargs,
            min_sibling_length=min_sibling_length,
            scale=scale,
            overwrite=True,
            **detection_kwargs,
        )

        # Kept alongside the GEFF for downstream readers that still expect a CSV.
        tracks_df.to_csv(position_dirpath / f"{tracks_stem}_{fov}.csv", index=False)

        with open_ome_zarr(position_dirpath, mode="r+") as position:
            label_image = position.get_label(label_name)
            write_tracking_labels(label_image["0"], tracking_labels, output_mode)

    # Record the full Ultrack config, so provenance does not depend on the (usually
    # deleted) database directory.
    label_group = zarr.open_group(position_dirpath / "labels" / label_name, mode="r+")
    provenance = dict(label_group.attrs.get(PROVENANCE_KEY, {}))
    provenance["ultrack_config"] = _resolved_ultrack_config(tracking_config)
    label_group.attrs[PROVENANCE_KEY] = provenance

    click.echo(f"Saved labels and tracks to: {position_dirpath}")


def _init_labels(
    input_position_dirpaths: list[Path],
    settings: TrackingSettings,
) -> tuple[int, int, int, int, int]:
    """Create the empty ``labels/<tracked_label>`` image in every input position.

    The input's dimensionality must match ``settings.output_mode`` (see
    :func:`check_input_dimensionality`), and with ``settings.labels`` every position
    must already have that label image (``biahub segment --init`` creates it, so a
    pipeline missing its segment step fails here); the tracked labels then take its
    dtype, chunks and shards. The tracked label image is (T, 1, Z, Y, X) with the
    plate's Z: one plane for a projected plate, the stack for 3D tracking.
    Positions that already have the output label are left untouched, and the track
    settings are recorded on each label image.

    Returns the input (T, C, Z, Y, X) shape used for resource estimation.
    """
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as dataset:
        Z = dataset.data.shape[2]
    check_input_dimensionality(settings.output_mode, Z)

    if settings.labels is not None:
        missing = [
            str(p) for p in input_position_dirpaths if not labels_exist([p], settings.labels)
        ]
        if missing:
            raise ValueError(
                f"labels: {settings.labels!r} names a label image that "
                f"{len(missing)} position(s) do not have (first: {missing[0]}); run "
                "`biahub segment` on the plate first (its model name is the label name)."
            )

    # The tracked labels take the chunks and shards of the label image they are
    # linked from (segment's), so the two read alike; without one, the defaults.
    shape, layout = label_layout(input_position_dirpaths[0], source_label=settings.labels)
    create_empty_labels(
        Path(input_position_dirpaths[0]).parents[2],
        [Path(p).parts[-3:] for p in input_position_dirpaths],
        [settings.tracked_label],
        # The settings as written; the worker adds the resolved Ultrack config under
        # the same key once it has tracked (track_one_position).
        label_metadata={PROVENANCE_KEY: settings.model_dump(mode="json")},
        **layout,
    )
    click.echo(
        f"Created labels/{settings.tracked_label} in "
        f"{Path(input_position_dirpaths[0]).parents[2]} "
        f"({len(input_position_dirpaths)} positions)"
    )
    return shape


def track(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path | None = None,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = True,
    init_only: bool = False,
    input_images_path: str | None = None,
):
    """Launch tracking jobs for multiple imaging positions.

    Parameters
    ----------
    input_position_dirpaths : list[Path]
        Paths to input position directories (used for plate structure and metadata).
    config_filepath : Path
        Path to the tracking configuration YAML.
    output_dirpath : Path, optional
        Where to keep the per-FOV Ultrack databases (``{output_dirpath}/{row}_{col}_{fov}/``)
        and the SLURM logs (``{output_dirpath}/slurm_output``). If None, each worker
        builds its database in its own ``$TMPDIR`` and deletes it after export, and the
        SLURM logs go to ``./slurm_output``. The labels and tracks themselves are
        always written into the ``-i`` positions.
    sbatch_filepath : str, optional
        Path to a SLURM batch file to override defaults.
    cluster : str, optional
        Execution cluster: 'slurm', 'local', or 'debug'.
    monitor : bool, optional
        If True, monitor submitted jobs.
    init_only : bool, optional
        Only create the empty label images in the ``-i`` positions and exit.
    input_images_path : str, optional
        Explicit pixel-data source filling the first null ``input_images`` path
        (Nextflow). If unset, that null path falls back to the ``-i`` plate (see Notes).

    Notes
    -----
    Outputs go into each ``-i`` position: ``labels/tracks_<labels>`` (OME-NGFF label
    image; ``labels/<target_channel>`` on the foreground+contour path),
    ``tracks.geff`` and ``tracks_<row>_<col>_<fov>.csv``. The ``-i`` plate is the plate
    track reads, so 2D tracking writes into the projected (Z=1) plate, next to the
    ``labels/<labels>`` image `biahub segment` wrote there.

    Two input sources: ``-i`` (``input_position_dirpaths``) gives the plate structure,
    shape/scale and the label image, while ``input_images[].path`` gives pixel data
    (foreground+contour). The first null ``path`` is the primary source
    (``--input-images-path`` if given, else the ``-i`` plate); later null entries are
    virtual/computed channels.
    """
    from ultrack import MainConfig

    if output_dirpath is not None:
        output_dirpath = Path(output_dirpath)
        slurm_out_path = output_dirpath / "slurm_output"
    else:
        # Not $TMPDIR: submitit's folder must be visible from the compute nodes.
        slurm_out_path = Path("slurm_output").absolute()
    slurm_out_path.mkdir(parents=True, exist_ok=True)

    settings = yaml_to_model(config_filepath, TrackingSettings)

    # Resolve the primary (first null) input_images path: an explicit
    # --input-images-path wins, else fall back to the -i input plate root.
    input_plate = Path(input_position_dirpaths[0]).parents[2]
    primary_path = Path(input_images_path) if input_images_path is not None else input_plate
    for image in settings.input_images:
        if image.path is None:
            image.path = primary_path
            break

    # Workers skip the scaffold --init already made: they all share the -i plate.
    if init_only or not labels_exist(input_position_dirpaths, settings.tracked_label):
        _init_labels(input_position_dirpaths, settings)
    with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as dataset:
        T, C, Z, Y, X = dataset.data.shape
        scale = dataset.scale
    check_input_dimensionality(settings.output_mode, Z)

    _, num_cpus, gb_ram_per_cpu = estimate_resources(
        shape=[T, C, Z, Y, X], ram_multiplier=16, max_num_cpus=16
    )
    time_minutes = 60
    mem_gb = num_cpus * gb_ram_per_cpu
    # Emit the JSON resources contract consumed by parse_resources in
    # nextflow/modules/common.nf (shared with every other step's --init).
    echo_resources(num_cpus, mem_gb, time_minutes)

    if init_only:
        if settings.tracker in ("hoct", "trackastra"):
            # Fail here, once, rather than in every worker: the image channel resolves
            # (segment provenance or image_channel) and exists in the plate.
            config = getattr(settings, f"{settings.tracker}_config")
            channel = tracker_image_channel(
                input_position_dirpaths[0], settings.labels, config.image_channel
            )
            with open_ome_zarr(str(input_position_dirpaths[0]), mode="r") as dataset:
                if channel not in dataset.channel_names:
                    raise ValueError(
                        f"{settings.tracker} would read channel {channel!r}, which the plate "
                        f"does not have ({dataset.channel_names}); set image_channel."
                    )
        # --init runs once, before the fan-out: the one place to download model weights
        # into the shared cache without every worker racing to do it.
        if settings.tracker == "hoct":
            hoct_solver_config(settings.hoct_config)  # check the solver overrides too
            warm_hoct_weights(settings.hoct_config.model)
        if (
            settings.tracker == "trackastra"
            and not Path(settings.trackastra_config.model).exists()
        ):
            warm_trackastra_weights(settings.trackastra_config.model)
        click.echo(f"Initialized tracking ({len(input_position_dirpaths)} positions)")
        return

    track_scale = scale[-2:] if settings.output_mode == "2D" else scale[-3:]

    tracking_cfg = None
    if settings.tracker == "ultrack":
        tracking_cfg = settings.tracking_config
        tracking_cfg["segmentation_config"]["n_workers"] = num_cpus
        tracking_cfg["linking_config"]["n_workers"] = num_cpus
        default_config = MainConfig()
        tracking_cfg = update_model(default_config, tracking_cfg)
    # HOCT and Trackastra run their models on a GPU when there is one.
    use_gpu = settings.tracker in ("hoct", "trackastra")

    position_keys = [Path(p).parts[-3:] for p in input_position_dirpaths]

    slurm_args = {
        "slurm_job_name": "track",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_cpus,
        "slurm_array_parallelism": 100,  # process up to 100 positions at a time
        "slurm_time": time_minutes,
        # Ultrack is CPU-bound (segmentation is `biahub segment`'s, on the GPU); it
        # cannot resume, but a position tracks in minutes, so a preemption costs little.
        # Override via --sbatch-filepath if a different partition is needed.
        "slurm_partition": "preempted",
        "slurm_use_srun": False,
    }
    if use_gpu:
        slurm_args.update({"slurm_partition": "gpu", "slurm_gpus_per_node": 1})

    if sbatch_filepath:
        slurm_args = apply_sbatch_overrides(slurm_args, sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    executor = submitit.AutoExecutor(folder=slurm_out_path, cluster=resolved_cluster)
    if resolved_cluster == "local":
        # Explicit --sbatch-filepath values (e.g. '#LOCAL --timeout-min') win over
        # the computed local executor parameters.
        slurm_args = {**local_executor_parameters(time_minutes, gpu=use_gpu), **slurm_args}
    executor.update_parameters(**slurm_args)

    click.echo("Submitting jobs...")
    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for position_key, position_dirpath in zip(
            position_keys, input_position_dirpaths, strict=True
        ):
            job = executor.submit(
                track_one_position,
                position_key=position_key,
                position_dirpath=Path(position_dirpath),
                output_dirpath=output_dirpath,
                tracking_config=tracking_cfg,
                label_name=settings.tracked_label,
                input_images=settings.input_images,
                blank_frames_path=settings.blank_frames_path,
                scale=track_scale,
                input_label=settings.labels,
                labels_sigma=settings.labels_sigma,
                output_mode=settings.output_mode,
                min_sibling_length=settings.min_sibling_length,
                tracks_stem=settings.tracks_stem,
                tracker=settings.tracker,
                tracksdata_config=settings.tracksdata_config,
                hoct_config=settings.hoct_config,
                trackastra_config=settings.trackastra_config,
            )
            jobs.append(job)

    job_ids = [job.job_id for job in jobs]
    log_path = Path(slurm_out_path / "submitit_jobs_ids.log")
    with log_path.open("w") as log_file:
        log_file.write("\n".join(job_ids))

    # submitit's DebugExecutor is lazy: .submit() wraps the callable in a
    # DebugJob but execution only happens when .wait()/.done()/.result() is
    # called.  On the Nextflow path (--cluster debug) we run in-process so
    # Nextflow handles the fan-out and resource scheduling.
    if resolved_cluster == "debug":
        for job, pk in zip(jobs, position_keys, strict=True):
            job.wait()
            click.echo(f"Tracking complete: {'/'.join(pk)}")
        return

    if monitor:
        monitor_jobs(jobs, input_position_dirpaths)


@click.command("track")
@input_position_dirpaths()
@config_filepath()
@output_dirpath(
    required=False,
    help="Directory to keep the per-FOV Ultrack databases and the SLURM logs in. "
    "If omitted, each worker builds its database in its own $TMPDIR and deletes it "
    "after export, and the SLURM logs go to ./slurm_output.",
)
@sbatch_filepath()
@cluster()
@monitor()
@init_only()
@click.option(
    "--input-images-path",
    default=None,
    type=click.Path(exists=True),
    help="Pixel-data source filling the first null input_images path (used by "
    "Nextflow). If omitted, that null path falls back to the -i input plate.",
)
def track_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path | None = None,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
    monitor: bool = False,
    init_only: bool = False,
    input_images_path: str | None = None,
):
    """Track the instance labels of a label image (`biahub segment`) with ultrack.

    Reads labels/<labels> from each -i position and writes into it: the tracked labels
    as an OME-NGFF label image (labels/tracks_<labels>), the tracks as tracks.geff,
    and a tracks CSV. 2D tracking reads (and writes into) the projected Z=1 plate
    `biahub project` makes; the Ultrack databases are temporary unless -o names a
    directory to keep them in. Track does not segment or project.

    \b
    Full SLURM fan-out:
    >>> biahub track -i ./assemble_proj.zarr/*/*/* -c config.yml

    \b
    Keep the Ultrack databases (e.g. to re-solve with other weights):
    >>> biahub track -i ./assemble_proj.zarr/*/*/* -o ./track -c config.yml

    \b
    Create the empty label images only (Nextflow init step):
    >>> biahub track --init -i ./assemble_proj.zarr/*/*/* -c config.yml

    \b
    In-process run of a single position (Nextflow per-position worker):
    >>> biahub track --cluster debug -i ./assemble_proj.zarr/B/3/000000 -c config.yml
    """  # noqa: D301
    track(
        input_position_dirpaths=input_position_dirpaths,
        output_dirpath=output_dirpath,
        config_filepath=config_filepath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
        monitor=monitor,
        init_only=init_only,
        input_images_path=input_images_path,
    )


if __name__ == "__main__":
    track_cli()
