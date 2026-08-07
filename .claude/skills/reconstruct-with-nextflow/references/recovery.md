# Error handling and recovery

## Triage: read before you act

Never delete zarr data or relaunch before classifying the failure. Start here:

```bash
cd <OUTPUT>
tail -n 200 .nextflow.log                       # rotates to .1, .2, ...
grep -n "ERROR\|Caused by\|Command error\|terminated with" .nextflow.log | tail -40
```

Nextflow names the failing process, the position, and a `Work dir:`. Go read it:

```bash
cat <WORKDIR>/.command.sh       # the exact biahub command, -i / -o / -p
cat <WORKDIR>/.exitcode
```

The real traceback is **not** in the work dir — see `caveats.md` §9. It is in
`<OUTPUT>/nextflow/slurm_output/<step>/*_<jobid>.out`.

## Decision table

| signature | cause | action |
|---|---|---|
| exit 143 / 137 / 140, no traceback | SLURM preemption, timeout, or OOM | none — Nextflow retries (exit 130–145, `maxRetries = 5`). Expected on the `preempted` partition. |
| exit 143 repeatedly on the *same* position until retries exhaust | task too slow or too large for its resource envelope | check `trace.txt` for that step's realized time/RSS; the run may need a wall-time or memory bump |
| `RuntimeError: the checksum is invalid` | torn shard from a killed write | **job-io-error-repair agent** — see below |
| `RuntimeError: The encoded shard is smaller than the expected size of its index.` | same | same |
| `RuntimeError: blosc encoded value is invalid` | same | same |
| `OSError` / `IOError` / **"Input/output error"** on a `.zarr` path | Lustre EIO — usually the same torn-shard condition surfacing through the filesystem layer; occasionally a genuine Lustre hiccup | **job-io-error-repair agent** |
| truncated / short read while decoding a chunk | same | same |
| `FileNotFoundError: Dataset directory not found at .../<zarr>/ROW/COL/FOV` | a previous cleanup deleted the metadata scaffold too | **job-io-error-repair agent** (it recreates the scaffold from a sibling) |
| exit 1/2 with a Python traceback (pydantic validation, `TypeError`, `KeyError`) | bad config or a real bug | Nextflow terminates deliberately. Fix, then relaunch with `-resume`. |
| `Expected a 'RESOURCES:' line in command output but none was found` | the underlying biahub CLI crashed during its init step | read the init step's `slurm_output` log; usually a config validation error |
| `list_positions` returns nothing | the input has no HCS plate | build `0-convert` — `caveats.md` §1 |

## Lustre EIO / torn shards — the important one

**What happens.** A per-position task is killed mid-write (preemption → SIGTERM,
exit 143). The shard file it was writing is left truncated. Every output array in
this pipeline overhangs in Z, Y and X (e.g. Z 86 → shard Z 96), so part of every
shard lies outside the array bound and **no write can ever cover a whole shard**.
That makes every write a read-modify-write. The next attempt reads the truncated
shard back, the CRC32C check fails, and it aborts.

**Why retries don't help.** The condition is deterministic. Nextflow's retry and
`nextflow -resume` both re-run the same read-modify-write against the same
corrupt file and fail identically, forever. The run makes no further progress.
A position stuck in a retry loop with the same error each time is this.

**Fix.** Delete only the corrupt **chunk data**, keep the metadata scaffold, then
resume. Delegate to the **job-io-error-repair** agent — it has the full
procedure. The rule it enforces, worth knowing so you can sanity-check it:

- delete `<output.zarr>/ROW/COL/FOV/0/c/` — the zarr-v3 chunk directory
- **keep** `<output.zarr>/ROW/COL/FOV/zarr.json` and
  `<output.zarr>/ROW/COL/FOV/0/zarr.json`

The scaffold is created by a *separate, cached* `init-*` Nextflow task. `-resume`
will not re-run it, and the worker opens the output FOV in `mode="r"` expecting
it to be there. Deleting the FOV directory wholesale turns a checksum error into
a `FileNotFoundError` and makes things worse.

Also delete the failed attempts' work dirs for those positions, but **keep the
successful positions' work dirs and `.nextflow/`** so `-resume` reuses finished
work.

Confirm the volume with the user before a large `rm -rf` (`du -sh` first); these
deletes can be hundreds of GB, and on Lustre they are slow — run them in the
background.

**Prevention.** Newer iohub/biahub carry unlink-before-write repair plus per-unit
resume markers under `<position>/0/.iohub-write-progress/`, which make a retry
skip already-written units and replace torn ones wholesale instead of reading
them back. A live preemption test on `2026_07_14_A549_MAP4_ZIKV_rerun` confirmed
it works: the retry skipped 120 units, recomputed 81, and produced zero codec
errors. If the biahub checkout predates that work, expect torn shards to need
manual repair. Full write-up:
`/hpc/projects/intracellular_dashboard/organelle_dynamics/2026_07_14_A549_MAP4_ZIKV_rerun/HANDOFF_torn_shard_resume.md`.

Note that the graceful-drain SIGTERM handler is **inert under Nextflow** —
Nextflow's generated `.command.run` does not wait for the payload after the
trap fires, so the partition's 30 s grace period is unusable in this path. Repair
plus resume is the load-bearing protection, not the signal handler.

**Corrupt *input* is different.** If the checksum/IO error is raised while reading
the *input* zarr rather than writing the output, the source is damaged. Do not
clean the output. Report it — the input may need re-export from the instrument.

## Restarting

Always the same command; `-resume` is already in the script:

```bash
cd <OUTPUT> && bash ./run_mantis_v2.sh
```

Relaunch in the tmux session so the user can still watch it.

`-resume` reuses cached tasks by hash. It is invalidated by: editing the biahub
checkout, editing a config file, or changing a `--param`. If a resume unexpectedly
recomputes everything, one of those three changed.

To force one step to recompute, delete that step's output for the affected
positions and remove their work dirs — not the whole `.nextflow/` cache.

## When to escalate to the user

- Retries exhausted on any step.
- The same position failing identically more than twice after a repair.
- Any deletion over ~100 GB.
- A traceback you cannot map to the table above.
- A step running more than ~3× the reference run's time for that step.
