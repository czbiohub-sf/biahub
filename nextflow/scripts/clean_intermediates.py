"""Delete a run's intermediate step stores once the assembled plate is proven to hold their pixels.

Deletes ONLY the four step zarrs::

    <OUTPUT>/<N>-flatfield/<DATASET>.zarr
    <OUTPUT>/<N>-deskew/<DATASET>.zarr
    <OUTPUT>/<N>-reconstruct/<DATASET>.zarr
    <OUTPUT>/<N>-virtual-stain/<DATASET>.zarr

Everything else stays: each step's ``slurm_output/``, ``.iohub-progress``,
``2-reconstruct/transfer_function.zarr``, ``nextflow/`` (trace, reports,
provenance), the assembled and tracking stores, QC.

Run with the biahub venv's python (needs iohub + numpy)::

    PY=<BIAHUB>/.venv/bin/python
    T=<BIAHUB>/nextflow/scripts/clean_intermediates.py

    $PY $T check  <OUTPUT>             # metadata checks, head node, seconds
    $PY $T submit <OUTPUT>             # SLURM array: pixel-verify every position
    $PY $T submit <OUTPUT> --timepoints 3   # faster: first, middle, last timepoint only
    $PY $T status <OUTPUT>             # summarize the verify results
    $PY $T delete <OUTPUT>             # dry run: what would go, and why it is safe
    $PY $T delete <OUTPUT> --yes       # actually delete

WHY PIXELS AND NOT SHAPES. ``concatenate --init`` scaffolds the assembled plate
at full shape before any data is copied, and an unwritten or torn shard reads
back as the fill value. So a plate whose copy failed has exactly the right
shape, dtype and channel count. The only proof the data arrived is to read it
back and compare it with the source.

WHAT ``verify`` COMPARES. For every position, timepoint and channel, the source
volume (deskew, reconstruct, virtual-stain, in ``-i`` order) cast to the
assembled dtype must equal the assembled volume exactly — NaN equal to NaN.
``concatenate`` copies pixels unchanged (it casts to float32 only when source
dtypes differ, which is lossless for uint16), so any difference is a failed
transfer, and so is a volume that cannot be read. Source volumes that are
entirely zero or NaN are reported but do not block: blank wells and dropped
frames are empty in the acquisition, and zarr skips writing all-zero shards, so
their gaps in the assembled plate are correct.

``--timepoints N`` compares N evenly spaced timepoints per position (always the
first and last) instead of all of them: about T/N times faster, since the check
is bound by Lustre reads. The trade-off is that a volume that transferred wrong
at an unchecked timepoint is not seen. The default is all timepoints, and
``status``, the ``delete`` dry run and the marker file all say which was used.

The delete also waits for the whole Nextflow run to finish, not just assemble:
tracking reads the assembled plate and QC writes tables into it.

Flat-field is not in the assembled plate. It is deleted on the strength of
deskew — which is computed from it — being verified complete.

Channels are matched by POSITION, not name: ``rename_channels.py`` renames the
assembled channels after the run, so names no longer agree with the sources.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import re
import shutil
import socket
import subprocess
import sys

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STEP_RE = re.compile(r"^(\d+)-(flatfield|deskew|reconstruct|virtual-stain|assemble|track)$")
# Deleted, in pipeline order. The last three are the assembled plate's sources,
# in the order `concatenate` receives them as `-i` groups.
DELETABLE = ["flatfield", "deskew", "reconstruct", "virtual-stain"]
SOURCES = ["deskew", "reconstruct", "virtual-stain"]
MARKER = "INTERMEDIATES_CLEANED.txt"
DONE_STATUSES = {"COMPLETED", "CACHED"}


# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------


def step_stores(output: Path) -> dict[str, Path]:
    """Map step name -> its <DATASET>.zarr (transfer_function.zarr excluded)."""
    stores = {}
    for d in sorted(output.iterdir()):
        m = STEP_RE.match(d.name)
        if not m or not d.is_dir():
            continue
        zarrs = [z for z in d.glob("*.zarr") if z.name != "transfer_function.zarr"]
        if len(zarrs) > 1:
            sys.exit(f"{d}: expected one step store, found {[z.name for z in zarrs]}")
        if zarrs:
            stores[m.group(2)] = zarrs[0]
    return stores


def work_dir(output: Path) -> Path:
    # CLEAN_INTERMEDIATES_WORKDIR redirects the verify results and logs, for
    # trying the script against a project you should not write into.
    override = os.environ.get("CLEAN_INTERMEDIATES_WORKDIR")
    return Path(override) if override else output / "nextflow" / "clean_intermediates"


def result_path(output: Path, position: str) -> Path:
    return work_dir(output) / "verify" / (position.replace("/", "_") + ".json")


def array_meta(position_dir: Path) -> dict:
    """shape/dtype of a position's full-resolution array, from metadata only."""
    v3 = position_dir / "0" / "zarr.json"
    if v3.exists():
        m = json.loads(v3.read_text())
        return {"shape": m["shape"], "dtype": m["data_type"]}
    m = json.loads((position_dir / "0" / ".zarray").read_text())
    return {"shape": m["shape"], "dtype": m["dtype"]}


def positions(store: Path) -> list[str]:
    found = set(store.glob("*/*/*/0/zarr.json")) | set(store.glob("*/*/*/0/.zarray"))
    return sorted(str(p.parent.parent.relative_to(store)) for p in found)


def uncompressed_bytes(store: Path) -> int:
    import numpy as np

    total = 0
    for pos in positions(store):
        meta = array_meta(store / pos)
        total += int(np.prod(meta["shape"])) * np.dtype(meta["dtype"]).itemsize
    return total


def sample_timepoints(n_t: int, n: int) -> list[int]:
    """N evenly spaced timepoints, always the first and last; all when n <= 0 or n >= n_t."""
    if n <= 0 or n >= n_t:
        return list(range(n_t))
    if n == 1:
        return [0]
    return sorted({round(i * (n_t - 1) / (n - 1)) for i in range(n)})


def coverage(results: list[dict]) -> str:
    """'all timepoints' or how many were sampled, over a set of verify results."""

    # Results without the field predate sampling and compared every timepoint.
    def n_checked(r):
        return len(r.get("timepoints_checked", range(r["n_timepoints"])))

    sampled = [r for r in results if n_checked(r) < r["n_timepoints"]]
    if not sampled:
        return "all timepoints"
    counts = sorted({n_checked(r) for r in sampled})
    return (
        f"SAMPLED ({len(sampled)}/{len(results)} positions checked at "
        f"{'/'.join(map(str, counts))} timepoints, not all)"
    )


def human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} EB"


# ----------------------------------------------------------------------------
# check — metadata only
# ----------------------------------------------------------------------------


def run_check(output: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (errors, warnings, assembled positions). Reads no pixels."""
    errors, warnings = [], []

    if (output / MARKER).exists():
        errors.append(f"already cleaned: {output / MARKER}")
    if list(output.glob("*-*/*.zarr.deleting-*")):
        warnings.append(
            "an interrupted delete left *.deleting-* stores; `delete --yes` finishes them"
        )

    stores = step_stores(output)
    if "assemble" not in stores:
        errors.append("no <N>-assemble/<DATASET>.zarr — nothing to verify against")
        return errors, warnings, []
    missing = [s for s in SOURCES if s not in stores]
    if missing:
        errors.append(f"source stores missing, cannot verify: {missing}")
        return errors, warnings, []

    asm = stores["assemble"]
    asm_positions = positions(asm)
    if not asm_positions:
        errors.append(f"{asm}: no positions")
        return errors, warnings, []

    # Every position in every source must be in the assembled plate, and vice versa.
    for s in SOURCES:
        src_positions = positions(stores[s])
        if set(src_positions) != set(asm_positions):
            only_src = sorted(set(src_positions) - set(asm_positions))
            only_asm = sorted(set(asm_positions) - set(src_positions))
            errors.append(
                f"{s}: positions differ from assemble "
                f"(only in {s}: {only_src[:5]}, only in assemble: {only_asm[:5]})"
            )

    # Geometry: assembled C = sum of source Cs, identical T/Z/Y/X. A crop or a
    # time/channel subset in concatenate.yml means the intermediates hold data
    # the assembled plate does not — refuse.
    for pos in asm_positions:
        a = array_meta(asm / pos)["shape"]
        src_shapes = {s: array_meta(stores[s] / pos)["shape"] for s in SOURCES}
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
            errors.append("… stopping geometry check after 20 errors")
            break

    # Every assembled position must have a finished run_concatenate task in the
    # LAST launch's trace (trace.txt is overwritten per launch; a resumed launch
    # records finished tasks as CACHED). A --max_positions smoke test leaves a
    # full-width plate with most positions never written — this catches it.
    trace = output / "nextflow" / "trace.txt"
    if not trace.exists():
        errors.append(f"{trace} missing — cannot confirm assemble finished")
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
    log = output / ".nextflow.log"
    if not log.exists():
        errors.append(f"{log} missing — cannot confirm the run finished")
    else:
        text = log.read_text(errors="replace")
        if "Execution complete -- Goodbye" not in text[-5_000:]:
            errors.append("the last Nextflow launch has not finished (still running?)")
        elif "Session aborted" in text:
            errors.append("the last Nextflow launch failed (Session aborted in .nextflow.log)")

    return errors, warnings, asm_positions


def cmd_check(args) -> int:
    output = args.output.resolve()
    errors, warnings, asm_positions = run_check(output)
    stores = step_stores(output)
    print(f"output: {output}")
    for name in DELETABLE:
        if name in stores:
            print(
                f"  delete candidate  {stores[name]}  (~{human(uncompressed_bytes(stores[name]))} uncompressed)"
            )
    if "assemble" in stores:
        print(f"  verified against  {stores['assemble']}  ({len(asm_positions)} positions)")
    for w in warnings:
        print(f"  WARNING  {w}")
    for e in errors:
        print(f"  ERROR    {e}")
    print(
        "check: "
        + ("FAILED" if errors else "passed (metadata only — pixels not verified yet)")
    )
    return 1 if errors else 0


# ----------------------------------------------------------------------------
# verify — pixels, one position
# ----------------------------------------------------------------------------


def _compare_timepoint(t, src_arrays, asm_array, dtype):
    import numpy as np

    mismatches, empty = [], []
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
            floating = np.issubdtype(dtype, np.floating)
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


def cmd_verify(args) -> int:
    import numpy as np

    from iohub.ngff import open_ome_zarr

    output = args.output.resolve()
    stores = step_stores(output)
    pos = args.position
    if pos is None:
        # SLURM array task: index into the position list written by `submit`.
        plist = json.loads((work_dir(output) / "positions.json").read_text())
        pos = plist[int(os.environ["SLURM_ARRAY_TASK_ID"])]

    started = dt.datetime.now().isoformat(timespec="seconds")
    handles = [open_ome_zarr(stores[s] / pos, mode="r") for s in SOURCES]
    asm = open_ome_zarr(stores["assemble"] / pos, mode="r")
    asm_array = asm["0"]
    dtype = np.dtype(asm_array.dtype)
    src_arrays = [(s, h["0"]) for s, h in zip(SOURCES, handles, strict=True)]

    mismatches, empty = [], []
    n_t = asm_array.shape[0]
    timepoints = sample_timepoints(n_t, args.timepoints)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for mm, em in pool.map(
            lambda t: _compare_timepoint(t, src_arrays, asm_array, dtype), timepoints
        ):
            mismatches += mm
            empty += em

    result = {
        "position": pos,
        "status": "pass" if not mismatches else "fail",
        "n_timepoints": n_t,
        "timepoints_checked": timepoints,
        "assembled_shape": list(asm_array.shape),
        "assembled_dtype": str(dtype),
        "mismatches": mismatches,
        "empty_source_volumes": empty,
        "stores": {s: str(stores[s]) for s in [*SOURCES, "assemble"]},
        "started": started,
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
    }
    out = result_path(output, pos)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out)
    print(
        f"{pos}: {result['status']}  mismatches={len(mismatches)}  empty={len(empty)}  -> {out}"
    )
    return 0 if result["status"] == "pass" else 1


# ----------------------------------------------------------------------------
# submit / status
# ----------------------------------------------------------------------------


def cmd_submit(args) -> int:
    output = args.output.resolve()
    errors, _, asm_positions = run_check(output)
    if errors:
        print("check failed — fix these before verifying pixels:")
        for e in errors:
            print(f"  ERROR  {e}")
        return 1

    todo = asm_positions
    if not args.all:
        # Skip positions that already passed; re-verify failures and missing ones.
        todo = [
            p
            for p in asm_positions
            if not (
                result_path(output, p).exists()
                and json.loads(result_path(output, p).read_text())["status"] == "pass"
            )
        ]
    if not todo:
        print("every position already verified — run `status`")
        return 0

    wd = work_dir(output)
    (wd / "slurm").mkdir(parents=True, exist_ok=True)
    (wd / "positions.json").write_text(json.dumps(todo))
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name=verify_assemble",
        f"--array=0-{len(todo) - 1}%{args.max_jobs}",
        f"--partition={args.partition}",
        f"--cpus-per-task={args.workers}",
        f"--mem={args.mem}",
        f"--time={args.time}",
        "--requeue",  # idempotent per position: a preempted task just reruns
        f"--output={wd}/slurm/%x_%A_%a.out",
        f"--error={wd}/slurm/%x_%A_%a.err",
        f"--wrap={sys.executable} {Path(__file__).resolve()} verify {output} --workers {args.workers} --timepoints {args.timepoints}",
    ]
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    job = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    print(f"submitted array job {job}: {len(todo)} positions, logs in {wd}/slurm/")
    print(f"when `squeue -j {job}` is empty, run: status {output}")
    return 0


def verify_summary(output: Path, asm_positions: list[str]):
    passed, failed, missing = [], [], []
    for p in asm_positions:
        rp = result_path(output, p)
        if not rp.exists():
            missing.append(p)
            continue
        r = json.loads(rp.read_text())
        (passed if r["status"] == "pass" else failed).append(r)
    return passed, failed, missing


def cmd_status(args) -> int:
    output = args.output.resolve()
    stores = step_stores(output)
    if "assemble" not in stores:
        sys.exit("no assembled store")
    asm_positions = positions(stores["assemble"])
    passed, failed, missing = verify_summary(output, asm_positions)
    print(
        f"verified {len(passed)}/{len(asm_positions)} pass, {len(failed)} fail, "
        f"{len(missing)} not verified — {coverage(passed + failed) if passed or failed else '-'}"
    )
    for r in failed:
        mm = r["mismatches"]
        print(f"  FAIL {r['position']}: {len(mm)} volumes did not transfer")
        for m in mm[:5]:
            if "read_error" in m:
                print(
                    f"       t={m['t']} {m['source']}[c{m['source_channel']}]: "
                    f"unreadable — {m['read_error']}"
                )
                continue
            print(
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
            print(
                f"  empty in source (copied as empty): {r['position']} {ch} "
                f"{len(ts)}/{r['n_timepoints']} timepoints, e.g. t={ts[:5]}"
            )
    if missing:
        print(f"  not verified: {missing[:10]}{' …' if len(missing) > 10 else ''}")
    return 0 if not failed and not missing else 1


# ----------------------------------------------------------------------------
# delete
# ----------------------------------------------------------------------------


def cmd_delete(args) -> int:
    output = args.output.resolve()

    # An interrupted delete: the stores were already renamed, i.e. the decision
    # was made and recorded. Finish removing them; nothing to re-check.
    leftovers = sorted(output.glob("*-*/*.zarr.deleting-*"))
    stores = step_stores(output)
    targets = [stores[s] for s in DELETABLE if s in stores]
    if leftovers and not targets:
        print("finishing an interrupted delete:")
        for d in leftovers:
            print(f"  delete  {d}")
        if not args.yes:
            print("delete: dry run — rerun with --yes")
            return 0
        for d in leftovers:
            shutil.rmtree(d)
        print("done")
        return 0

    errors, warnings, asm_positions = run_check(output)
    passed, failed, missing = (
        verify_summary(output, asm_positions) if asm_positions else ([], [], [])
    )
    if failed:
        errors.append(
            f"{len(failed)} positions have pixels that did not transfer — run `status`"
        )
    if missing:
        errors.append(f"{len(missing)} positions not pixel-verified — run `submit`")

    # A result only counts for the stores it read, and only if no pipeline launch
    # came after it: a relaunch rewrites trace.txt and may have rewritten data.
    trace = output / "nextflow" / "trace.txt"
    trace_mtime = trace.stat().st_mtime if trace.exists() else 0
    for r in passed:
        for s, path in r["stores"].items():
            if stores.get(s) and str(stores[s]) != path:
                errors.append(
                    f"{r['position']}: verified against {path}, store is now {stores[s]}"
                )
        if dt.datetime.fromisoformat(r["started"]).timestamp() < trace_mtime:
            errors.append(
                f"{r['position']}: verified before the last pipeline launch — "
                "re-verify (`submit --all`)"
            )

    print(f"output: {output}")
    for t in targets:
        print(f"  delete  {t}")
    print(f"  keep    {stores.get('assemble')}")
    if "track" in stores:
        print(f"  keep    {stores['track']}")
    print(
        "  keep    every step's slurm_output/, transfer_function.zarr, nextflow/, qc/, configs/"
    )
    print(
        f"pixel verification: {len(passed)}/{len(asm_positions)} positions pass, "
        f"{coverage(passed) if passed else '-'}"
    )
    for w in warnings:
        print(f"  WARNING  {w}")
    for e in errors[:30]:
        print(f"  ERROR    {e}")
    if len(errors) > 30:
        print(f"  … {len(errors) - 30} more errors")

    if errors:
        print("delete: REFUSED")
        return 1
    if not args.yes:
        print("delete: dry run — rerun with --yes to delete")
        return 0

    # Rename first: a half-deleted store must never look like a valid one, and an
    # interrupted delete is found and finished by the leftovers path above.
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    doomed = []
    for t in targets:
        d = t.with_name(f"{t.name}.deleting-{stamp}")
        t.rename(d)
        doomed.append(d)

    lines = [
        f"date      {dt.datetime.now().isoformat(timespec='seconds')}",
        f"user      {getpass.getuser()}",
        f"host      {socket.gethostname()}",
        f"script    {Path(__file__).resolve()}",
        f"verified  {len(passed)}/{len(asm_positions)} positions: every source "
        "voxel equals the assembled voxel",
        f"          in {stores['assemble']}, {coverage(passed)}",
        "removed",
        *[f"          {t}" for t in targets],
        "",
        "-resume still runs the steps after assemble (track, QC): flat-field..assemble",
        "come back cached and never open the deleted stores. If any of them re-runs",
        "instead, stop the run.",
    ]
    (output / MARKER).write_text("\n".join(lines) + "\n")

    for d in doomed:
        print(f"  removing {d} …", flush=True)
        shutil.rmtree(d)
    print(f"done — wrote {output / MARKER}")
    return 0


# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="metadata checks only")
    p.add_argument("output", type=Path)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("verify", help="pixel-compare one position (SLURM array task)")
    p.add_argument("output", type=Path)
    p.add_argument("--position", help="e.g. A/1/000; default: from SLURM_ARRAY_TASK_ID")
    p.add_argument("--workers", type=int, default=8, help="timepoints compared in parallel")
    p.add_argument(
        "--timepoints",
        type=int,
        default=0,
        help="compare N evenly spaced timepoints per position (incl. first and last); "
        "0 = all (default)",
    )
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("submit", help="SLURM array verifying every position")
    p.add_argument("output", type=Path)
    p.add_argument(
        "--all", action="store_true", help="re-verify positions that already passed"
    )
    p.add_argument("--partition", default="preempted")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--timepoints",
        type=int,
        default=0,
        help="compare N evenly spaced timepoints per position (incl. first and last); "
        "0 = all (default)",
    )
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="4:00:00")
    p.add_argument("--max-jobs", type=int, default=50, help="array tasks running at once")
    p.add_argument("--dry-run", action="store_true", help="print the sbatch command only")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="summarize verify results")
    p.add_argument("output", type=Path)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("delete", help="delete the step stores (dry run without --yes)")
    p.add_argument("output", type=Path)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_delete)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
