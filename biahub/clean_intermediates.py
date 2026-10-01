"""Delete a finished mantis-v2 run's intermediate step stores once the assembled plate holds their pixels.

Deletes ONLY the four step zarrs of a ``nextflow/mantis-v2.nf`` project::

    <OUTPUT>/<N>-flatfield/<DATASET>.zarr
    <OUTPUT>/<N>-deskew/<DATASET>.zarr
    <OUTPUT>/<N>-reconstruct/<DATASET>.zarr
    <OUTPUT>/<N>-virtual-stain/<DATASET>.zarr

Everything else stays: each step's ``slurm_output/``, ``.iohub-progress``,
``2-reconstruct/transfer_function.zarr``, ``nextflow/`` (trace, reports,
provenance), the assembled and tracking stores, QC.

WHY PIXELS AND NOT SHAPES. ``concatenate --init`` scaffolds the assembled plate
at full shape before any data is copied, and an unwritten or torn shard reads
back as the fill value. So a plate whose copy failed has exactly the right
shape, dtype and channel count. The only proof the data arrived is to read it
back and compare it with the source.

WHAT VERIFICATION COMPARES. For every position, timepoint and channel, the
source volume (deskew, reconstruct, virtual-stain, in ``-i`` order) cast to the
assembled dtype must equal the assembled volume exactly, NaN equal to NaN.
``concatenate`` copies pixels unchanged (it casts to float32 only when source
dtypes differ, which is lossless for uint16), so any difference is a failed
transfer, and so is a volume that cannot be read. Source volumes that are
entirely zero or NaN are reported but do not block: blank wells and dropped
frames are empty in the acquisition, and zarr skips writing all-zero shards, so
their gaps in the assembled plate are correct.

``timepoints`` compares N evenly spaced timepoints per position (always the
first and last) instead of all of them: about T/N times faster, since the check
is bound by filesystem reads. A volume that transferred wrong at an unchecked
timepoint is not seen. The default is all timepoints, and the status, the
delete dry run and the marker file all say which was used.

Deletion also waits for the whole Nextflow run to finish, not just assemble:
tracking reads the assembled plate and QC writes tables into it.

Flat-field is not in the assembled plate. It is deleted on the strength of
deskew, which is computed from it, being verified complete.

Channels are matched by POSITION, not name: ``rename_channels.py`` renames the
assembled channels after the run, so names no longer agree with the sources.
"""

import datetime as dt
import getpass
import json
import re
import shutil
import socket

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
import submitit

from iohub.ngff import open_ome_zarr

from biahub.cli.parsing import cluster, sbatch_filepath, sbatch_to_submitit
from biahub.utils.cluster import get_submitit_cluster

STEP_RE = re.compile(r"^(\d+)-(flatfield|deskew|reconstruct|virtual-stain|assemble|track)$")
# Deleted, in pipeline order. The last three are the assembled plate's sources,
# in the order `concatenate` receives them as `-i` groups.
DELETABLE = ["flatfield", "deskew", "reconstruct", "virtual-stain"]
SOURCES = ["deskew", "reconstruct", "virtual-stain"]
# Beside nextflow/provenance.txt, at a fixed path: run_mantis_v2.sh looks for it,
# so it cannot live under the (overridable) verify directory.
MARKER = Path("nextflow") / "intermediates_cleaned.txt"
DONE_STATUSES = {"COMPLETED", "CACHED"}


# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------


def _step_stores(output_dirpath: Path) -> dict[str, Path]:
    """Map step name -> its <DATASET>.zarr (transfer_function.zarr excluded)."""
    stores = {}
    for d in sorted(output_dirpath.iterdir()):
        m = STEP_RE.match(d.name)
        if not m or not d.is_dir():
            continue
        zarrs = [z for z in d.glob("*.zarr") if z.name != "transfer_function.zarr"]
        if len(zarrs) > 1:
            raise click.ClickException(
                f"{d}: expected one step store, found {[z.name for z in zarrs]}"
            )
        if not zarrs:
            continue
        # Step numbers depend on which steps a run performed, so a project can
        # hold a stale 4-reconstruct beside a current 3-reconstruct. Refuse
        # rather than pick one: verifying or deleting the wrong set is silent.
        if m.group(2) in stores:
            raise click.ClickException(
                f"two stores for step '{m.group(2)}': {stores[m.group(2)]} and {zarrs[0]}; "
                "move the stale one out of the project first"
            )
        stores[m.group(2)] = zarrs[0]
    return stores


def _verify_dirpath(output_dirpath: Path, verify_dirpath: Path | None) -> Path:
    return verify_dirpath or output_dirpath / "nextflow" / "clean_intermediates"


def _log_dirpath(output_dirpath: Path, verify_dirpath: Path | None) -> Path:
    """SLURM logs, beside every other step's: nextflow/slurm_output/<step>/."""
    if verify_dirpath:
        return verify_dirpath / "slurm_output"
    return output_dirpath / "nextflow" / "slurm_output" / "clean_intermediates"


def _result_path(verify_dirpath: Path, position: str) -> Path:
    return verify_dirpath / "verify" / (position.replace("/", "_") + ".json")


def _array_meta(position_dirpath: Path) -> dict:
    """shape/dtype of a position's full-resolution array, from metadata only."""
    v3 = position_dirpath / "0" / "zarr.json"
    if v3.exists():
        m = json.loads(v3.read_text())
        return {"shape": m["shape"], "dtype": m["data_type"]}
    m = json.loads((position_dirpath / "0" / ".zarray").read_text())
    return {"shape": m["shape"], "dtype": m["dtype"]}


def _positions(store: Path) -> list[str]:
    found = set(store.glob("*/*/*/0/zarr.json")) | set(store.glob("*/*/*/0/.zarray"))
    return sorted(str(p.parent.parent.relative_to(store)) for p in found)


def _uncompressed_bytes(store: Path) -> int:
    total = 0
    for pos in _positions(store):
        meta = _array_meta(store / pos)
        total += int(np.prod(meta["shape"])) * np.dtype(meta["dtype"]).itemsize
    return total


def _human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} EB"


def _sample_timepoints(n_t: int, n: int) -> list[int]:
    """N evenly spaced timepoints, always the first and last; all when n <= 0 or n >= n_t."""
    if n <= 0 or n >= n_t:
        return list(range(n_t))
    if n == 1:
        return [0]
    return sorted({round(i * (n_t - 1) / (n - 1)) for i in range(n)})


def _coverage(results: list[dict]) -> str:
    """'all timepoints' or how many were sampled, over a set of verify results."""

    def n_checked(r):
        # Results without the field predate sampling and compared every timepoint.
        return len(r.get("timepoints_checked", range(r["n_timepoints"])))

    if not results:
        return "-"
    sampled = [r for r in results if n_checked(r) < r["n_timepoints"]]
    if not sampled:
        return "all timepoints"
    counts = sorted({n_checked(r) for r in sampled})
    return (
        f"SAMPLED ({len(sampled)}/{len(results)} positions checked at "
        f"{'/'.join(map(str, counts))} timepoints, not all)"
    )


def _verify_summary(verify_dirpath: Path, positions: list[str]):
    passed, failed, missing = [], [], []
    for p in positions:
        rp = _result_path(verify_dirpath, p)
        if not rp.exists():
            missing.append(p)
            continue
        r = json.loads(rp.read_text())
        (passed if r["status"] == "pass" else failed).append(r)
    return passed, failed, missing


# ----------------------------------------------------------------------------
# check: metadata only
# ----------------------------------------------------------------------------


def already_cleaned(output_dirpath: Path) -> str | None:
    """Say why there is nothing to clean, or None if there is.

    Parameters
    ----------
    output_dirpath : Path
        The mantis-v2 project root.

    Returns
    -------
    str | None
        A one-line reason (cleaned by this command, or by hand with no step
        store left), or None when intermediates remain.
    """
    stores = _step_stores(output_dirpath)
    if any(s in stores for s in DELETABLE):
        return None
    marker = output_dirpath / MARKER
    if marker.exists():
        first = marker.read_text().splitlines()[0].split()
        when = first[1] if len(first) > 1 else "?"
        return f"already cleaned on {when}, see {marker}: nothing to do"
    return "no intermediate stores left: nothing to do"


def check_intermediates(output_dirpath: Path) -> tuple[list[str], list[str], list[str]]:
    """Check, from metadata alone, that a project's intermediates may be verified.

    Parameters
    ----------
    output_dirpath : Path
        The mantis-v2 project root (Nextflow ``--output``).

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``(errors, warnings, positions)``: blocking problems, informational
        notes, and the assembled plate's position keys. Reads no pixels.
    """
    output_dirpath = Path(output_dirpath).resolve()
    errors, warnings = [], []

    if (output_dirpath / MARKER).exists():
        errors.append(f"already cleaned: {output_dirpath / MARKER}")
    if list(output_dirpath.glob("*-*/*.zarr.deleting-*")):
        warnings.append(
            "an interrupted delete left *.deleting-* stores; `delete --yes` finishes them"
        )

    stores = _step_stores(output_dirpath)
    if "assemble" not in stores:
        errors.append("no <N>-assemble/<DATASET>.zarr, nothing to verify against")
        return errors, warnings, []
    missing = [s for s in SOURCES if s not in stores]
    if missing:
        errors.append(f"source stores missing, cannot verify: {missing}")
        return errors, warnings, []

    asm = stores["assemble"]
    asm_positions = _positions(asm)
    if not asm_positions:
        errors.append(f"{asm}: no positions")
        return errors, warnings, []

    # Every position in every source must be in the assembled plate, and vice versa.
    for s in SOURCES:
        src_positions = _positions(stores[s])
        if set(src_positions) != set(asm_positions):
            only_src = sorted(set(src_positions) - set(asm_positions))
            only_asm = sorted(set(asm_positions) - set(src_positions))
            errors.append(
                f"{s}: positions differ from assemble "
                f"(only in {s}: {only_src[:5]}, only in assemble: {only_asm[:5]})"
            )

    # Geometry: assembled C = sum of source Cs, identical T/Z/Y/X. A crop or a
    # time/channel subset in concatenate.yml means the intermediates hold data
    # the assembled plate does not, so refuse.
    for pos in asm_positions:
        a = _array_meta(asm / pos)["shape"]
        src_shapes = {s: _array_meta(stores[s] / pos)["shape"] for s in SOURCES}
        c_sum = sum(sh[1] for sh in src_shapes.values())
        if a[1] != c_sum:
            errors.append(
                f"{pos}: assembled has {a[1]} channels, sources sum to {c_sum} "
                "(channel subset in concatenate.yml?)"
            )
        for s, sh in src_shapes.items():
            if [sh[0], *sh[2:]] != [a[0], *a[2:]]:
                errors.append(
                    f"{pos}: {s} TZYX {[sh[0], *sh[2:]]} != assembled {[a[0], *a[2:]]} "
                    "(crop or time subset in concatenate.yml?)"
                )
        if len(errors) > 20:
            errors.append("... stopping geometry check after 20 errors")
            break

    # Every assembled position must have a finished run_concatenate task in the
    # LAST launch's trace (trace.txt is overwritten per launch; a resumed launch
    # records finished tasks as CACHED). A --max_positions smoke test leaves a
    # full-width plate with most positions never written, and this catches it.
    trace = output_dirpath / "nextflow" / "trace.txt"
    if not trace.exists():
        errors.append(f"{trace} missing, cannot confirm assemble finished")
    else:
        done = set()
        for line in trace.read_text().splitlines()[1:]:
            cols = line.split("\t")
            if len(cols) > 4 and ":run_concatenate (" in cols[3] and cols[4] in DONE_STATUSES:
                done.add(cols[3].split("(", 1)[1].rstrip(")"))
        not_done = [p for p in asm_positions if p not in done]
        if not_done:
            errors.append(
                f"{len(not_done)}/{len(asm_positions)} positions have no finished "
                f"run_concatenate in trace.txt, e.g. {not_done[:5]}"
            )

    # The whole run must be over, not just assemble: tracking reads the
    # assembled plate and QC writes tables into it. `.nextflow.log` is always the
    # latest launch; it ends with "Execution complete" once the head process has
    # exited, and logs "Session aborted" when that launch failed.
    log = output_dirpath / ".nextflow.log"
    if not log.exists():
        errors.append(f"{log} missing, cannot confirm the run finished")
    else:
        text = log.read_text(errors="replace")
        if "Execution complete -- Goodbye" not in text[-5_000:]:
            errors.append("the last Nextflow launch has not finished (still running?)")
        elif "Session aborted" in text:
            errors.append("the last Nextflow launch failed (Session aborted in .nextflow.log)")

    return errors, warnings, asm_positions


# ----------------------------------------------------------------------------
# verify: pixels, one position
# ----------------------------------------------------------------------------


def _compare_timepoint(t, src_arrays, asm_array, dtype):
    mismatches, empty = [], []
    floating = np.issubdtype(dtype, np.floating)
    c0 = 0
    for name, arr in src_arrays:
        for c in range(arr.shape[1]):
            where = {"t": t, "source": name, "source_channel": c, "assembled_channel": c0 + c}
            try:
                src = np.asarray(arr[t, c]).astype(dtype, copy=False)
                got = np.asarray(asm_array[t, c0 + c])
            except Exception as exc:  # torn shard, checksum, EIO: unverifiable
                mismatches.append({**where, "read_error": f"{type(exc).__name__}: {exc}"})
                continue
            if not np.array_equal(src, got, equal_nan=floating):
                if floating:
                    diff = ~((src == got) | (np.isnan(src) & np.isnan(got)))
                else:
                    diff = src != got
                mismatches.append(
                    {
                        **where,
                        "n_diff": int(diff.sum()),
                        "n_voxels": int(diff.size),
                        "assembled_all_zero": bool(not got.any()),
                    }
                )
            if not src.any() or (floating and np.isnan(src).all()):
                empty.append({"t": t, "source": name, "source_channel": c})
        c0 += arr.shape[1]
    return mismatches, empty


def verify_position(
    output_dirpath: Path,
    position: str,
    timepoints: int = 0,
    num_workers: int = 8,
    verify_dirpath: Path | None = None,
) -> dict:
    """Compare one position's source volumes with the assembled plate, voxel for voxel.

    Writes the result to ``<verify_dirpath>/verify/<position>.json``.

    Parameters
    ----------
    output_dirpath : Path
        The mantis-v2 project root.
    position : str
        Position key, e.g. ``"A/1/000"``.
    timepoints : int, optional
        Compare N evenly spaced timepoints (always first and last); 0 compares
        all of them. Default 0.
    num_workers : int, optional
        Timepoints compared in parallel. Default 8.
    verify_dirpath : Path, optional
        Where results go. Default ``<output_dirpath>/nextflow/clean_intermediates``.

    Returns
    -------
    dict
        The result record; ``status`` is ``"pass"`` when every compared voxel
        matched and every volume could be read.
    """
    output_dirpath = Path(output_dirpath).resolve()
    stores = _step_stores(output_dirpath)
    started = dt.datetime.now().isoformat(timespec="seconds")

    handles = [open_ome_zarr(stores[s] / position, mode="r") for s in SOURCES]
    asm = open_ome_zarr(stores["assemble"] / position, mode="r")
    asm_array = asm["0"]
    dtype = np.dtype(asm_array.dtype)
    src_arrays = [(s, h["0"]) for s, h in zip(SOURCES, handles, strict=True)]

    n_t = asm_array.shape[0]
    checked = _sample_timepoints(n_t, timepoints)
    mismatches, empty = [], []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for mm, em in pool.map(
            lambda t: _compare_timepoint(t, src_arrays, asm_array, dtype), checked
        ):
            mismatches += mm
            empty += em
    for h in [*handles, asm]:
        h.close()

    result = {
        "position": position,
        "status": "pass" if not mismatches else "fail",
        "n_timepoints": n_t,
        "timepoints_checked": checked,
        "assembled_shape": list(asm_array.shape),
        "assembled_dtype": str(dtype),
        "mismatches": mismatches,
        "empty_source_volumes": empty,
        "stores": {s: str(stores[s]) for s in [*SOURCES, "assemble"]},
        "started": started,
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
    }
    out = _result_path(_verify_dirpath(output_dirpath, verify_dirpath), position)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out)
    return result


# ----------------------------------------------------------------------------
# submit / status
# ----------------------------------------------------------------------------


def _verify_resources(position_dirpath: Path, n_timepoints: int, num_workers: int):
    """(mem_gb, time_minutes) for one position's verify job.

    Calibrated on 2026_09_22_A549_PCNA_DENV_ZIKV (91 x 5 x 86 x 1664 x 1193
    float32): 8 workers peaked at 22 GB, about 4 volumes per worker in flight
    (source, assembled, cast, diff), and the full position took 17 min alone
    and 37 min with 45 positions reading Lustre at once (~0.07 min per GB
    read). Requests keep ~2x headroom on both.
    """
    shape = _array_meta(position_dirpath)["shape"]
    vol_gb = int(np.prod(shape[2:])) * 4 / 2**30  # compared as up to float32
    mem_gb = int(np.ceil(max(16, 5 * vol_gb * num_workers + 4)))
    read_gb = 2 * vol_gb * shape[1] * n_timepoints  # source + assembled
    time_minutes = int(np.ceil(max(30, 0.15 * read_gb) / 10) * 10)
    return mem_gb, time_minutes


def submit_verification(
    output_dirpath: Path,
    timepoints: int = 0,
    reverify: bool = False,
    num_workers: int = 8,
    verify_dirpath: Path | None = None,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
) -> list:
    """Submit one verify job per position that has not already passed.

    Parameters
    ----------
    output_dirpath : Path
        The mantis-v2 project root.
    timepoints : int, optional
        Timepoints compared per position; 0 = all. Default 0.
    reverify : bool, optional
        Re-verify positions that already passed. Default False.
    num_workers : int, optional
        Timepoints compared in parallel within a job. Default 8.
    verify_dirpath : Path, optional
        Where results and job logs go. Default: results in
        ``<output_dirpath>/nextflow/clean_intermediates``, logs in
        ``<output_dirpath>/nextflow/slurm_output/clean_intermediates``.
    sbatch_filepath : str, optional
        SBATCH file overriding the default SLURM parameters.
    cluster : str, optional
        'slurm', 'local' or 'debug' (in-process). Default 'slurm'.

    Returns
    -------
    list
        The submitted submitit jobs (empty when there was nothing to do).
    """
    output_dirpath = Path(output_dirpath).resolve()
    reason = already_cleaned(output_dirpath)
    if reason:
        click.echo(reason)
        return []
    errors, _, positions = check_intermediates(output_dirpath)
    if errors:
        raise click.ClickException(
            "check failed, fix these before verifying pixels:\n  " + "\n  ".join(errors)
        )

    vdir = _verify_dirpath(output_dirpath, verify_dirpath)
    todo = positions
    if not reverify:
        passed, _, _ = _verify_summary(vdir, positions)
        done = {r["position"] for r in passed}
        todo = [p for p in positions if p not in done]
    if not todo:
        click.echo("every position already verified, run `status`")
        return []

    stores = _step_stores(output_dirpath)
    first = stores["assemble"] / todo[0]
    n_t = _array_meta(first)["shape"][0]
    mem_gb, time_minutes = _verify_resources(
        first, len(_sample_timepoints(n_t, timepoints)), num_workers
    )
    slurm_args = {
        "slurm_job_name": "verify_intermediates",
        "slurm_mem": f"{mem_gb}G",
        "slurm_cpus_per_task": num_workers,
        "slurm_array_parallelism": 50,  # each job streams hundreds of GB from Lustre
        "slurm_time": time_minutes,
        "slurm_partition": "preempted",
        # Idempotent per position: a preempted task simply reruns.
        "slurm_additional_parameters": {"requeue": True},
    }
    if sbatch_filepath:
        slurm_args.update(sbatch_to_submitit(sbatch_filepath))

    resolved_cluster = get_submitit_cluster(cluster=cluster)
    click.echo(f"Preparing jobs on cluster='{resolved_cluster}': {slurm_args}")
    log_dirpath = _log_dirpath(output_dirpath, verify_dirpath)
    executor = submitit.AutoExecutor(folder=log_dirpath, cluster=resolved_cluster)
    executor.update_parameters(**slurm_args)

    jobs = []
    with submitit.helpers.clean_env(), executor.batch():
        for position in todo:
            jobs.append(
                executor.submit(
                    verify_position,
                    output_dirpath,
                    position,
                    timepoints=timepoints,
                    num_workers=num_workers,
                    verify_dirpath=verify_dirpath,
                )
            )

    # submitit's DebugExecutor is lazy: run each job in the foreground.
    if resolved_cluster == "debug":
        for job, position in zip(jobs, todo, strict=True):
            click.echo(f"{position}: {job.result()['status']}")
        return jobs

    click.echo(
        f"submitted {len(todo)} verify jobs ({jobs[0].job_id.split('_')[0]}), "
        f"logs in {log_dirpath}; when they finish, run `status`"
    )
    return jobs


def verification_status(output_dirpath: Path, verify_dirpath: Path | None = None) -> bool:
    """Summarize the verify results of a project.

    Parameters
    ----------
    output_dirpath : Path
        The mantis-v2 project root.
    verify_dirpath : Path, optional
        Where results are. Default ``<output_dirpath>/nextflow/clean_intermediates``.

    Returns
    -------
    bool
        True when every assembled position is verified and passed.
    """
    output_dirpath = Path(output_dirpath).resolve()
    stores = _step_stores(output_dirpath)
    if "assemble" not in stores:
        raise click.ClickException("no assembled store")
    positions = _positions(stores["assemble"])
    passed, failed, missing = _verify_summary(
        _verify_dirpath(output_dirpath, verify_dirpath), positions
    )
    click.echo(
        f"verified {len(passed)}/{len(positions)} pass, {len(failed)} fail, "
        f"{len(missing)} not verified, {_coverage(passed + failed)}"
    )
    for r in failed:
        mm = r["mismatches"]
        click.echo(f"  FAIL {r['position']}: {len(mm)} volumes did not transfer")
        for m in mm[:5]:
            if "read_error" in m:
                click.echo(
                    f"       t={m['t']} {m['source']}[c{m['source_channel']}]: "
                    f"unreadable, {m['read_error']}"
                )
                continue
            click.echo(
                f"       t={m['t']} {m['source']}[c{m['source_channel']}] -> assembled "
                f"c{m['assembled_channel']}: {m['n_diff']}/{m['n_voxels']} voxels differ"
                + (" (assembled volume all zero)" if m["assembled_all_zero"] else "")
            )
    # Informational: blank wells and dropped frames are empty in the source and,
    # correctly, in the assembled plate too. Summarized per position and channel.
    for r in passed + failed:
        per_channel = {}
        for e in r["empty_source_volumes"]:
            per_channel.setdefault(f"{e['source']}[c{e['source_channel']}]", []).append(e["t"])
        for ch, ts in sorted(per_channel.items()):
            click.echo(
                f"  empty in source (copied as empty): {r['position']} {ch} "
                f"{len(ts)} timepoints, e.g. t={ts[:5]}"
            )
    if missing:
        click.echo(f"  not verified: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return not failed and not missing


# ----------------------------------------------------------------------------
# delete
# ----------------------------------------------------------------------------


def delete_intermediates(
    output_dirpath: Path, yes: bool = False, verify_dirpath: Path | None = None
) -> bool:
    """Delete the four intermediate step stores once every position passed verification.

    Without ``yes`` this is a dry run that lists what would be deleted and why
    it is safe, or why it is refused.

    Parameters
    ----------
    output_dirpath : Path
        The mantis-v2 project root.
    yes : bool, optional
        Actually delete. Default False (dry run).
    verify_dirpath : Path, optional
        Where results are. Default ``<output_dirpath>/nextflow/clean_intermediates``.

    Returns
    -------
    bool
        False when the delete is refused, True otherwise (including dry runs).
    """
    output_dirpath = Path(output_dirpath).resolve()
    vdir = _verify_dirpath(output_dirpath, verify_dirpath)

    # An interrupted delete: the stores were already renamed, i.e. the decision
    # was made and recorded. Finish removing them; nothing to re-check.
    # An interrupted delete: the marker is written, listing the stores to
    # remove, BEFORE any of them is renamed, so the decision is on record.
    # Finish it, renaming whatever the interruption left unrenamed; nothing to
    # re-check, since the verification it was based on is gone with the stores.
    leftovers = sorted(output_dirpath.glob("*-*/*.zarr.deleting-*"))
    stores = _step_stores(output_dirpath)
    targets = [stores[s] for s in DELETABLE if s in stores]
    marker = output_dirpath / MARKER
    if leftovers:
        if not marker.exists():
            raise click.ClickException(
                f"*.deleting-* stores but no {marker}: not left by this command, inspect by hand"
            )
        listed = {ln.strip() for ln in marker.read_text().splitlines()}
        unrenamed = [t for t in targets if str(t) in listed]
        click.echo("finishing an interrupted delete:")
        for d in [*unrenamed, *leftovers]:
            click.echo(f"  delete  {d}")
        if not yes:
            click.echo("delete: dry run, rerun with --yes")
            return True
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        for t in unrenamed:
            d = t.with_name(f"{t.name}.deleting-{stamp}")
            t.rename(d)
            leftovers.append(d)
        for d in leftovers:
            shutil.rmtree(d)
        click.echo("done")
        return True

    reason = already_cleaned(output_dirpath)
    if reason:
        click.echo(reason)
        return True
    errors, warnings, positions = check_intermediates(output_dirpath)
    passed, failed, missing = _verify_summary(vdir, positions) if positions else ([], [], [])
    if failed:
        errors.append(
            f"{len(failed)} positions have pixels that did not transfer, run `status`"
        )
    if missing:
        errors.append(f"{len(missing)} positions not pixel-verified, run `submit`")

    # A result only counts for the stores it read, and only if no pipeline launch
    # came after it: a relaunch rewrites trace.txt and may have rewritten data.
    trace = output_dirpath / "nextflow" / "trace.txt"
    trace_mtime = trace.stat().st_mtime if trace.exists() else 0
    for r in passed:
        for s, path in r["stores"].items():
            if stores.get(s) and str(stores[s]) != path:
                errors.append(
                    f"{r['position']}: verified against {path}, store is now {stores[s]}"
                )
        if dt.datetime.fromisoformat(r["started"]).timestamp() < trace_mtime:
            errors.append(
                f"{r['position']}: verified before the last pipeline launch, "
                "re-verify (`submit --reverify`)"
            )

    click.echo(f"output: {output_dirpath}")
    for t in targets:
        click.echo(f"  delete  {t}  (~{_human(_uncompressed_bytes(t))} uncompressed)")
    click.echo(f"  keep    {stores.get('assemble')}")
    if "track" in stores:
        click.echo(f"  keep    {stores['track']}")
    click.echo(
        "  keep    every step's slurm_output/, transfer_function.zarr, nextflow/, qc/, configs/"
    )
    click.echo(
        f"pixel verification: {len(passed)}/{len(positions)} positions pass, "
        f"{_coverage(passed)}"
    )
    for w in warnings:
        click.echo(f"  WARNING  {w}")
    for e in errors[:30]:
        click.echo(f"  ERROR    {e}")
    if len(errors) > 30:
        click.echo(f"  ... {len(errors) - 30} more errors")

    if errors:
        click.echo("delete: REFUSED")
        return False
    if not yes:
        click.echo("delete: dry run, rerun with --yes to delete")
        return True

    lines = [
        f"date      {dt.datetime.now().isoformat(timespec='seconds')}",
        f"user      {getpass.getuser()}",
        f"host      {socket.gethostname()}",
        "command   biahub nf clean-intermediates delete --yes",
        f"verified  {len(passed)}/{len(positions)} positions: every compared source "
        "voxel equals the assembled voxel",
        f"          in {stores['assemble']}, {_coverage(passed)}",
        "removed",
        *[f"          {t}" for t in targets],
        "",
        "-resume still runs the steps after assemble (track, QC): flat-field..assemble",
        "come back cached and never open the deleted stores. If any of them re-runs",
        "instead, stop the run.",
    ]
    # Marker first, then rename, then remove. The marker records the decision, so
    # an interruption anywhere after it is finished by the leftovers path above;
    # renaming means a half-deleted store never looks like a valid one.
    marker.write_text("\n".join(lines) + "\n")
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    doomed = []
    for t in targets:
        d = t.with_name(f"{t.name}.deleting-{stamp}")
        t.rename(d)
        doomed.append(d)

    for d in doomed:
        click.echo(f"  removing {d} ...")
        shutil.rmtree(d)
    click.echo(f"done, wrote {marker}")
    return True


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _output_dirpath() -> click.Argument:
    return click.argument(
        "output_dirpath",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
    )


def _verify_dirpath_option() -> click.Option:
    return click.option(
        "--verify-dirpath",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        help="Where verify results and job logs go, for trying the command on a "
        "project you should not write into. Default: results in "
        "<OUTPUT_DIRPATH>/nextflow/clean_intermediates, logs in "
        "<OUTPUT_DIRPATH>/nextflow/slurm_output/clean_intermediates.",
    )


@click.group("clean-intermediates")
def clean_intermediates_cli():
    """Delete a finished run's intermediate step stores after pixel-verifying assemble.

    Deletes only <N>-flatfield, <N>-deskew, <N>-reconstruct and <N>-virtual-stain
    <DATASET>.zarr; logs, transfer_function.zarr, nextflow/, qc/ and the assembled
    and tracking stores stay.

    \b
    >>> biahub nf clean-intermediates check  /hpc/projects/.../<DATASET>
    >>> biahub nf clean-intermediates submit /hpc/projects/.../<DATASET>
    >>> biahub nf clean-intermediates status /hpc/projects/.../<DATASET>
    >>> biahub nf clean-intermediates delete /hpc/projects/.../<DATASET>        # dry run
    >>> biahub nf clean-intermediates delete /hpc/projects/.../<DATASET> --yes
    """  # noqa: D301


@clean_intermediates_cli.command("check")
@_output_dirpath()
def check_cli(output_dirpath: Path):
    """Metadata checks only: stores, positions, geometry, trace.txt, run finished."""
    reason = already_cleaned(output_dirpath)
    if reason:
        click.echo(reason)
        return
    errors, warnings, positions = check_intermediates(output_dirpath)
    stores = _step_stores(output_dirpath)
    click.echo(f"output: {output_dirpath}")
    for name in DELETABLE:
        if name in stores:
            click.echo(
                f"  delete candidate  {stores[name]}  "
                f"(~{_human(_uncompressed_bytes(stores[name]))} uncompressed)"
            )
    if "assemble" in stores:
        click.echo(f"  verified against  {stores['assemble']}  ({len(positions)} positions)")
    for w in warnings:
        click.echo(f"  WARNING  {w}")
    for e in errors:
        click.echo(f"  ERROR    {e}")
    if errors:
        raise click.ClickException("check failed")
    click.echo("check: passed (metadata only, pixels not verified yet)")


@clean_intermediates_cli.command("submit")
@_output_dirpath()
@click.option(
    "--timepoints",
    type=int,
    default=0,
    show_default=True,
    help="Compare N evenly spaced timepoints per position (incl. first and last); 0 = all.",
)
@click.option("--reverify", is_flag=True, help="Re-verify positions that already passed.")
@click.option(
    "--num-workers",
    type=int,
    default=8,
    show_default=True,
    help="Timepoints compared at once.",
)
@_verify_dirpath_option()
@sbatch_filepath()
@cluster()
def submit_cli(
    output_dirpath: Path,
    timepoints: int = 0,
    reverify: bool = False,
    num_workers: int = 8,
    verify_dirpath: Path | None = None,
    sbatch_filepath: str | None = None,
    cluster: str = "slurm",
):
    """Pixel-verify every position against the assembled plate, one job per position."""
    submit_verification(
        output_dirpath=output_dirpath,
        timepoints=timepoints,
        reverify=reverify,
        num_workers=num_workers,
        verify_dirpath=verify_dirpath,
        sbatch_filepath=sbatch_filepath,
        cluster=cluster,
    )


@clean_intermediates_cli.command("status")
@_output_dirpath()
@_verify_dirpath_option()
def status_cli(output_dirpath: Path, verify_dirpath: Path | None = None):
    """Summarize verify results: pass/fail per position, empty source volumes."""
    if not verification_status(output_dirpath=output_dirpath, verify_dirpath=verify_dirpath):
        raise SystemExit(1)


@clean_intermediates_cli.command("delete")
@_output_dirpath()
@click.option("--yes", is_flag=True, help="Actually delete. Without it, a dry run.")
@_verify_dirpath_option()
def delete_cli(output_dirpath: Path, yes: bool = False, verify_dirpath: Path | None = None):
    """Delete the intermediate step stores (dry run without --yes)."""
    if not delete_intermediates(
        output_dirpath=output_dirpath, yes=yes, verify_dirpath=verify_dirpath
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    clean_intermediates_cli()
