"""Place Zarr chunk/shard files directly, for copies that do not change the data.

``biahub concatenate`` normally assembles its output by reading each input
volume, cropping it, and writing it back — a full decompress/recompress of every
byte. When the output geometry matches the input exactly and no crop is
requested, that round trip produces stored files that are byte-for-byte what the
input already holds, so the file can simply be placed at its new key instead.
For the mantis-v2 assemble step that is ~16 TiB of codec work replaced by a file
copy, or by nothing at all in ``link`` mode.

Two properties make the decision mechanical rather than a judgement call:

*The array metadata document is the contract.* A stored chunk or shard is
interpretable only through its array's metadata — data type, chunk grid, codecs,
fill value, key encoding. If two arrays agree on all of it a file is
interchangeable between them, and if they disagree anywhere it is not.
:func:`copy_incompatibilities` therefore compares the whole document instead of
a hand-picked list of fields, so a codec option introduced by a future Zarr
release disqualifies the copy by default rather than silently producing an
output that cannot be decoded.

*The file is the unit.* One file holds one chunk-grid cell — a shard when the
array is sharded, a chunk when it is not. Remapping ``(t, c)`` is then a rename
of that cell, but only if a cell holds exactly one channel and one timepoint,
which is what the pipeline's stores do (``shards_ratio`` defaults to a T ratio
of 1, and ``process_single_position`` already rejects sharding along C).
Anything coarser is rejected.

Placement is atomic: each file is written under a temporary name in its final
directory and then ``os.replace``d into place. An interrupted run therefore
leaves either the old file or the new one and never a torn one, which is the
failure mode ``--resume`` exists to recover from on the read-write path.

Two differences from the read-write path are deliberate and worth knowing:

- ``copy_n_paste`` replaces NaN with zero and ``process_single_position`` skips a
  ``(t, c)`` block that is entirely zero or NaN. A byte copy cannot inspect the
  data, so **NaNs are carried through unchanged** instead of becoming zeros. An
  input block that was never written has no file, and the destination is left
  with no file either, which reads back as the fill value exactly as before.
- Progress is recorded by the destination files themselves rather than by
  ``iohub``'s per-unit markers, so ``resume`` here means "the destination file is
  already the same size as its source". Because placement is atomic that is exact
  for a run interrupted on this path; a file torn by an interrupted *read-write*
  run is only caught if its size differs, which in practice it does.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
import shutil
import threading

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import click

from humanize import naturalsize
from iohub import open_ome_zarr

#: How a file is materialised at its destination. ``copy`` moves the bytes, so
#: the output store is independent of its inputs. ``link`` creates a hard link,
#: so both stores reference one copy of the data: instant and free of disk, but
#: writing into either store afterwards would corrupt the other.
PlacementMode = Literal["copy", "link"]

#: Metadata fields left out of the interchangeability comparison. ``shape``
#: differs by design — the output holds more channels and possibly fewer
#: timepoints — and is checked per axis instead. ``attributes`` and
#: ``dimension_names`` are descriptive: neither takes part in encoding a chunk,
#: so a difference there cannot make a stored file uninterpretable.
_UNCOMPARED_METADATA_KEYS = frozenset({"shape", "attributes", "dimension_names"})

#: zattrs keys owned by the OME-Zarr spec, which provenance must not overwrite.
#: Mirrors ``iohub.ngff.utils._OME_KEYS``.
_OME_KEYS = frozenset({"ome", "multiscales", "omero", "labels", "version"})

#: Longest value rendered inline when reporting a metadata difference. A codec
#: list runs to hundreds of characters and drowns the message it belongs to.
_MAX_INLINE_REPR = 60


class UnsupportedLayoutError(ValueError):
    """The array's on-disk layout is not one this module can address by key."""


@dataclass(frozen=True)
class ArrayLayout:
    """Everything about a Zarr array that decides whether its files can be reused.

    Attributes
    ----------
    path : Path
        Directory holding the array's metadata and chunk/shard files.
    zarr_format : int
        2 (OME-Zarr v0.4) or 3 (OME-Zarr v0.5).
    shape : tuple[int, ...]
        TCZYX array shape.
    file_shape : tuple[int, ...]
        Array extent covered by one file on disk: the shard shape when the array
        is sharded, the chunk shape when it is not. This is the outer chunk grid
        in both cases, which is exactly the granularity at which files exist.
    metadata : dict
        The parsed metadata document, compared field-by-field against the
        destination's by :func:`copy_incompatibilities`.
    key_prefix : str
        Leading path component of a chunk key (``"c"`` for Zarr v3's default
        encoding, empty for Zarr v2 and for v3's ``v2`` encoding).
    key_separator : str
        Separator between grid indices in a chunk key.
    """

    path: Path
    zarr_format: int
    shape: tuple[int, ...]
    file_shape: tuple[int, ...]
    metadata: dict[str, Any]
    key_prefix: str
    key_separator: str

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Number of files along each axis, rounding up at the array bound."""
        return tuple(
            math.ceil(extent / step)
            for extent, step in zip(self.shape, self.file_shape, strict=True)
        )

    def file_path(self, cell: Sequence[int]) -> Path:
        """Path of the file holding the chunk-grid cell at ``cell``."""
        key = self.key_separator.join(str(index) for index in cell)
        if self.key_prefix:
            key = self.key_separator.join((self.key_prefix, key)) if key else self.key_prefix
        return self.path / key


def read_array_layout(array_path: Path) -> ArrayLayout:
    """Read the on-disk layout of the Zarr array at ``array_path``.

    Parameters
    ----------
    array_path : Path
        Directory of the array itself, e.g. ``plate.zarr/A/1/0/0``.

    Returns
    -------
    ArrayLayout

    Raises
    ------
    FileNotFoundError
        If neither ``zarr.json`` nor ``.zarray`` is present.
    UnsupportedLayoutError
        If the metadata uses a chunk grid or key encoding this module cannot
        turn into a file path.
    """
    array_path = Path(array_path)

    v3_metadata_path = array_path / "zarr.json"
    if v3_metadata_path.exists():
        metadata = json.loads(v3_metadata_path.read_text())
        chunk_grid = metadata.get("chunk_grid", {})
        if chunk_grid.get("name") != "regular":
            raise UnsupportedLayoutError(f"unsupported chunk grid {chunk_grid.get('name')!r}")
        # For a sharded array this is the *shard* shape: the sharding codec's
        # own ``chunk_shape`` describes the chunks nested inside one file, and
        # is compared as part of ``codecs`` rather than used for addressing.
        file_shape = tuple(chunk_grid["configuration"]["chunk_shape"])
        prefix, separator = _v3_key_encoding(metadata)
        return ArrayLayout(
            path=array_path,
            zarr_format=3,
            shape=tuple(metadata["shape"]),
            file_shape=file_shape,
            metadata=metadata,
            key_prefix=prefix,
            key_separator=separator,
        )

    v2_metadata_path = array_path / ".zarray"
    if v2_metadata_path.exists():
        metadata = json.loads(v2_metadata_path.read_text())
        return ArrayLayout(
            path=array_path,
            zarr_format=2,
            shape=tuple(metadata["shape"]),
            file_shape=tuple(metadata["chunks"]),
            metadata=metadata,
            key_prefix="",
            key_separator=metadata.get("dimension_separator") or ".",
        )

    raise FileNotFoundError(f"No zarr.json or .zarray under {array_path}")


def _v3_key_encoding(metadata: dict[str, Any]) -> tuple[str, str]:
    """``(key prefix, separator)`` for a Zarr v3 array's chunk keys."""
    encoding = metadata.get("chunk_key_encoding") or {"name": "default"}
    configuration = encoding.get("configuration") or {}
    if encoding["name"] == "default":
        prefix, default_separator = "c", "/"
    elif encoding["name"] == "v2":
        prefix, default_separator = "", "."
    else:
        raise UnsupportedLayoutError(f"unsupported chunk key encoding {encoding['name']!r}")
    return prefix, configuration.get("separator", default_separator)


def copy_incompatibilities(
    source: ArrayLayout,
    destination: ArrayLayout,
    zyx_slicing_params: Sequence[slice],
    input_time_indices: Sequence[int],
    output_time_indices: Sequence[int],
) -> list[str]:
    """Reasons ``source``'s files cannot be reused verbatim in ``destination``.

    An empty list means a file-level copy reproduces exactly what reading,
    cropping and rewriting would have produced (NaN handling aside, see the
    module docstring).

    Parameters
    ----------
    source, destination : ArrayLayout
        Layouts of the input and output arrays.
    zyx_slicing_params : Sequence[slice]
        The Z, Y and X slices concatenate would apply. Anything other than the
        full extent of the input is a crop, which no rename can express.
    input_time_indices, output_time_indices : Sequence[int]
        The time remapping, paired element-wise.

    Returns
    -------
    list[str]
        Human-readable reasons, empty if the copy is valid.
    """
    if source.zarr_format != destination.zarr_format:
        # Nothing below compares meaningfully across formats, so stop here.
        return [
            f"Zarr format differs (input v{source.zarr_format}, "
            f"output v{destination.zarr_format})"
        ]

    reasons = []

    differences = [
        _describe_difference(key, source.metadata.get(key), destination.metadata.get(key))
        for key in sorted(set(source.metadata) | set(destination.metadata))
        if key not in _UNCOMPARED_METADATA_KEYS
        and source.metadata.get(key) != destination.metadata.get(key)
    ]
    if differences:
        reasons.append("array metadata differs: " + ", ".join(differences))

    if source.shape[2:] != destination.shape[2:]:
        reasons.append(
            f"ZYX shape differs (input {source.shape[2:]}, output {destination.shape[2:]})"
        )

    cropped_axes = _cropped_axes(zyx_slicing_params, source.shape[2:])
    if cropped_axes:
        reasons.append("ROI crops " + ", ".join(cropped_axes))

    if source.file_shape[1] != 1 or destination.file_shape[1] != 1:
        reasons.append(
            "a stored file spans several channels "
            f"(input {source.file_shape[1]}, output {destination.file_shape[1]}), "
            "so channels cannot be remapped by key"
        )

    time_extent = max(source.file_shape[0], destination.file_shape[0])
    if time_extent != 1 and not _time_mapping_is_identity(
        source, destination, input_time_indices, output_time_indices
    ):
        # A file holding several timepoints carries all of them with it, so it is
        # only reusable when the copy leaves the time axis alone and the trailing
        # file clips at the same index on both sides.
        reasons.append(
            f"a stored file spans {time_extent} timepoints and the copy is not "
            "the identity on the time axis"
        )

    return reasons


def _describe_difference(key: str, source_value: Any, destination_value: Any) -> str:
    """Name a differing metadata field, with values when they are short enough."""
    source_repr, destination_repr = repr(source_value), repr(destination_value)
    if max(len(source_repr), len(destination_repr)) <= _MAX_INLINE_REPR:
        return f"{key} (input {source_repr}, output {destination_repr})"
    return key


def _cropped_axes(zyx_slicing_params: Sequence[slice], zyx_shape: Sequence[int]) -> list[str]:
    """Axes whose requested slice is not the full extent of the input."""
    cropped = []
    for axis, requested, extent in zip("ZYX", zyx_slicing_params, zyx_shape, strict=True):
        start, stop, step = requested.indices(extent)
        if (start, stop, step) != (0, extent, 1):
            cropped.append(f"{axis}[{start}:{stop}]")
    return cropped


def _time_mapping_is_identity(
    source: ArrayLayout,
    destination: ArrayLayout,
    input_time_indices: Sequence[int],
    output_time_indices: Sequence[int],
) -> bool:
    """Whether every timepoint keeps its index and the whole axis is copied."""
    every_timepoint = list(range(source.shape[0]))
    return (
        source.shape[0] == destination.shape[0]
        and list(input_time_indices) == every_timepoint
        and list(output_time_indices) == every_timepoint
    )


@dataclass(frozen=True)
class DirectCopyPlan:
    """A validated file-level copy of one input position into one output position.

    Holds only the two layouts and the index remapping; the file pairs are
    enumerated on demand by :meth:`file_pairs` so that a driver planning
    hundreds of positions does not carry tens of thousands of paths.
    """

    source: ArrayLayout
    destination: ArrayLayout
    input_time_indices: tuple[int, ...]
    output_time_indices: tuple[int, ...]
    input_channel_indices: tuple[int, ...]
    output_channel_indices: tuple[int, ...]

    @property
    def _time_cells(self) -> list[tuple[int, int]]:
        """``(input cell, output cell)`` pairs along the time axis, deduplicated."""
        return list(
            dict.fromkeys(
                (
                    input_index // self.source.file_shape[0],
                    output_index // self.destination.file_shape[0],
                )
                for input_index, output_index in zip(
                    self.input_time_indices, self.output_time_indices, strict=True
                )
            )
        )

    @property
    def _channel_cells(self) -> list[tuple[int, int]]:
        """``(input cell, output cell)`` pairs along the channel axis.

        A file spans exactly one channel here — :func:`copy_incompatibilities`
        rejects anything else — so a channel index *is* its cell index.
        """
        return list(
            dict.fromkeys(
                zip(self.input_channel_indices, self.output_channel_indices, strict=True)
            )
        )

    @property
    def num_files(self) -> int:
        """How many files the copy addresses, present on disk or not."""
        spatial = math.prod(self.source.grid_shape[2:])
        return len(self._time_cells) * len(self._channel_cells) * spatial

    def file_pairs(self) -> list[tuple[Path, Path]]:
        """``(source, destination)`` paths for every file the copy addresses.

        The ZYX grid is shared: ``copy_incompatibilities`` has already required
        the two arrays to agree on both ZYX shape and file shape, so a spatial
        cell index means the same thing on each side.
        """
        spatial_cells = tuple(
            itertools.product(*(range(count) for count in self.source.grid_shape[2:]))
        )
        return [
            (
                self.source.file_path((input_time, input_channel, *cell)),
                self.destination.file_path((output_time, output_channel, *cell)),
            )
            for (input_time, output_time), (
                input_channel,
                output_channel,
            ) in itertools.product(self._time_cells, self._channel_cells)
            for cell in spatial_cells
        ]


def plan_position_copy(
    input_position_path: Path,
    output_position_path: Path,
    input_channel_indices: Sequence[int],
    output_channel_indices: Sequence[int],
    input_time_indices: Sequence[int],
    output_time_indices: Sequence[int],
    zyx_slicing_params: Sequence[slice],
    input_array_name: str = "0",
    output_array_name: str = "0",
) -> tuple[DirectCopyPlan | None, list[str]]:
    """Plan a file-level copy between two positions, or explain why there is none.

    Reads only the two arrays' metadata, so this is cheap enough to call for
    every position from the submitting process.

    Returns
    -------
    tuple[DirectCopyPlan | None, list[str]]
        The plan and an empty reason list, or None and the reasons the copy is
        not valid. A metadata document that cannot be read or understood is
        reported as a reason rather than raised, so one odd store degrades to the
        read-write path instead of aborting the run.
    """
    try:
        source = read_array_layout(Path(input_position_path) / input_array_name)
        destination = read_array_layout(Path(output_position_path) / output_array_name)
    except (OSError, KeyError, ValueError) as error:
        return None, [f"could not read array metadata: {error}"]

    reasons = copy_incompatibilities(
        source,
        destination,
        zyx_slicing_params,
        input_time_indices,
        output_time_indices,
    )
    if reasons:
        return None, reasons

    plan = DirectCopyPlan(
        source=source,
        destination=destination,
        input_time_indices=tuple(input_time_indices),
        output_time_indices=tuple(output_time_indices),
        input_channel_indices=tuple(input_channel_indices),
        output_channel_indices=tuple(output_channel_indices),
    )
    return plan, []


@dataclass
class PlacementReport:
    """Outcome counts for a batch of file placements."""

    placed: int = 0
    reused: int = 0
    cleared: int = 0
    absent: int = 0
    bytes_placed: int = 0
    #: Set when ``link`` was requested but the filesystems made it impossible.
    mode: PlacementMode = "copy"

    def record(self, outcome: str, size: int) -> None:
        setattr(self, outcome, getattr(self, outcome) + 1)
        self.bytes_placed += size

    def __str__(self) -> str:
        verb = "linked" if self.mode == "link" else "copied"
        parts = [f"{self.placed} {verb} ({naturalsize(self.bytes_placed, binary=True)})"]
        if self.reused:
            parts.append(f"{self.reused} already present")
        if self.cleared:
            parts.append(f"{self.cleared} removed (no source file)")
        if self.absent:
            parts.append(f"{self.absent} absent on both sides")
        return ", ".join(parts)


def resolve_placement_mode(
    source: Path, destination: Path, mode: PlacementMode
) -> PlacementMode:
    """Downgrade ``link`` to ``copy`` when the two paths are on different devices.

    A hard link cannot cross a filesystem boundary. Checking once here turns
    what would be an ``EXDEV`` failure on the first file into a warning and a
    byte copy.
    """
    if mode != "link":
        return mode
    try:
        same_device = os.stat(source).st_dev == os.stat(destination).st_dev
    except OSError:
        same_device = False
    if same_device:
        return "link"
    click.echo(
        f"Warning: {source} and {destination} are not on the same filesystem, "
        "so hard links are not possible. Copying bytes instead."
    )
    return "copy"


def place_files(
    pairs: Sequence[tuple[Path, Path]],
    mode: PlacementMode = "copy",
    num_workers: int = 1,
    resume: bool = False,
) -> PlacementReport:
    """Materialise every ``(source, destination)`` pair, in parallel.

    Parameters
    ----------
    pairs : Sequence[tuple[Path, Path]]
        Files to place. A pair whose source does not exist means the input never
        wrote that block, so any destination file is *removed* rather than left
        behind — matching what the read-write path does when it skips an
        all-zero or all-NaN block, and making a re-run over a stale output store
        converge on the input's contents.
    mode : PlacementMode, optional
        ``copy`` (default) or ``link``; see :data:`PlacementMode`. Pass a value
        already through :func:`resolve_placement_mode`.
    num_workers : int, optional
        Threads to place files with. The work is filesystem calls with the GIL
        released, so threads are the right tool and a network filesystem
        benefits from many of them. Defaults to 1.
    resume : bool, optional
        Skip a destination that already exists with the same size as its source.
        Exact for a run interrupted on this path, because placement is atomic.
        Defaults to False.

    Returns
    -------
    PlacementReport
    """
    report = PlacementReport(mode=mode)
    if not pairs:
        return report

    num_workers = max(1, min(num_workers, len(pairs)))
    if num_workers == 1:
        for source, destination in pairs:
            report.record(*_place_one(source, destination, mode=mode, resume=resume))
        return report

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(_place_one, source, destination, mode=mode, resume=resume)
            for source, destination in pairs
        ]
        for future in as_completed(futures):
            report.record(*future.result())
    return report


def _place_one(
    source: Path,
    destination: Path,
    mode: PlacementMode,
    resume: bool,
) -> tuple[str, int]:
    """Place one file atomically. Returns ``(outcome, bytes placed)``.

    The file is created under a temporary name in the destination's own
    directory and then renamed over the target, which is atomic on a POSIX
    filesystem. A kill at any point therefore leaves the destination either
    untouched or complete, never truncated.
    """
    try:
        source_size = os.stat(source).st_size
    except FileNotFoundError:
        # The input never wrote this block. Remove a stale destination so the
        # output reads back as the fill value, as it would after a skipped write.
        try:
            destination.unlink()
        except FileNotFoundError:
            return "absent", 0
        return "cleared", 0

    if resume:
        with contextlib.suppress(OSError):
            if os.stat(destination).st_size == source_size:
                return "reused", 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Unique per thread and process so concurrent placements never share a
    # scratch name, and dot-prefixed so a leftover cannot be mistaken for a
    # chunk key by a reader globbing the directory.
    scratch = (
        destination.parent / f".{destination.name}.{os.getpid()}-{threading.get_ident()}.tmp"
    )
    try:
        if mode == "link":
            os.link(source, scratch)
        else:
            shutil.copyfile(source, scratch)
        os.replace(scratch, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            scratch.unlink()
        raise
    return "placed", source_size


def copy_position_files(
    input_position_path: Path,
    output_position_path: Path,
    input_channel_indices: Sequence[int],
    output_channel_indices: Sequence[int],
    input_time_indices: Sequence[int],
    output_time_indices: Sequence[int],
    zyx_slicing_params: Sequence[slice],
    input_array_name: str = "0",
    output_array_name: str = "0",
    mode: PlacementMode = "copy",
    num_workers: int = 1,
    resume: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> PlacementReport:
    """Copy one input position into one output position at the file level.

    The read-write counterpart of this function is
    ``iohub.ngff.utils.process_single_position``, and the arguments mirror it so
    a caller can choose between them per position.

    Re-validates the copy rather than trusting the caller's plan, and raises if
    it no longer holds: the caller decided from the same metadata, so a mismatch
    means the stores changed underneath and silently writing something else
    would be worse than stopping.

    Raises
    ------
    RuntimeError
        If the two arrays are not file-level compatible.
    """
    click.echo(f"Direct chunk copy ({mode}) from:\t{input_position_path}")
    click.echo(f"Output data path:\t{output_position_path}")

    plan, reasons = plan_position_copy(
        input_position_path=input_position_path,
        output_position_path=output_position_path,
        input_channel_indices=input_channel_indices,
        output_channel_indices=output_channel_indices,
        input_time_indices=input_time_indices,
        output_time_indices=output_time_indices,
        zyx_slicing_params=zyx_slicing_params,
        input_array_name=input_array_name,
        output_array_name=output_array_name,
    )
    if plan is None:
        raise RuntimeError(
            f"{input_position_path} cannot be copied into {output_position_path} at "
            f"the file level: {'; '.join(reasons)}"
        )

    _write_extra_metadata(output_position_path, extra_metadata)

    resolved_mode = resolve_placement_mode(plan.source.path, plan.destination.path, mode)
    report = place_files(
        plan.file_pairs(),
        mode=resolved_mode,
        num_workers=num_workers,
        resume=resume,
    )
    click.echo(f"Finished {output_position_path}: {report}")
    return report


def _write_extra_metadata(
    output_position_path: Path, extra_metadata: dict[str, Any] | None
) -> None:
    """Record per-step provenance on the output position.

    Each entry becomes its own top-level zattrs key so successive steps
    accumulate sibling provenance rather than overwriting one shared key, which
    is what ``process_single_position``'s ``extra_metadata`` does.
    """
    if not extra_metadata:
        return
    reserved = _OME_KEYS.intersection(extra_metadata)
    if reserved:
        raise ValueError(
            f"extra_metadata keys {sorted(reserved)} are reserved OME-Zarr keys. "
            "Use a namespaced key (e.g. '<package>-<step>') instead."
        )
    with open_ome_zarr(output_position_path, layout="fov", mode="r+") as position:
        for key, value in extra_metadata.items():
            position.zattrs[key] = value


@dataclass
class DirectCopySummary:
    """How much of a multi-position copy can bypass the read-write path."""

    total: int = 0
    direct: int = 0
    files: int = 0
    #: Reason string -> number of position sources it disqualified.
    fallback_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def all_direct(self) -> bool:
        return self.total > 0 and self.direct == self.total


def summarize_direct_copy(
    plans: Sequence[DirectCopyPlan | None], reasons: Sequence[list[str]]
) -> DirectCopySummary:
    """Aggregate per-position planning results for reporting."""
    summary = DirectCopySummary(total=len(plans))
    for plan, plan_reasons in zip(plans, reasons, strict=True):
        if plan is not None:
            summary.direct += 1
            summary.files += plan.num_files
            continue
        for reason in plan_reasons:
            summary.fallback_reasons[reason] = summary.fallback_reasons.get(reason, 0) + 1
    return summary


def echo_direct_copy_summary(summary: DirectCopySummary) -> None:
    """Report the copy plan, and why anything fell back, on stdout."""
    click.echo(
        f"Direct chunk copy: {summary.direct}/{summary.total} position sources "
        f"({summary.files} files)"
    )
    for reason, count in sorted(
        summary.fallback_reasons.items(), key=lambda item: (-item[1], item[0])
    ):
        click.echo(f"  read-write fallback ({count}): {reason}")
