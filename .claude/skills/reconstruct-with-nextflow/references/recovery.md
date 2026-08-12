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
| `RuntimeError: the checksum is invalid` | torn shard from a killed write | **rerun with `-resume` first** — see below. Only propose a repair if it fails identically again. |
| `RuntimeError: The encoded shard is smaller than the expected size of its index.` | same | same |
| `RuntimeError: blosc encoded value is invalid` | same | same |
| `OSError` / `IOError` / **"Input/output error"** on a `.zarr` path | Lustre EIO — usually the same torn-shard condition surfacing through the filesystem layer; occasionally a genuine Lustre hiccup | **rerun with `-resume` first**; propose a repair only if it recurs identically |
| truncated / short read while decoding a chunk | same | same |
| `FileNotFoundError: Dataset directory not found at .../<zarr>/ROW/COL/FOV` | a previous cleanup deleted the metadata scaffold too | **job-io-error-repair agent** (it recreates the scaffold from a sibling) — this is why the skill never deletes zarr data itself |
| exit 1/2 with a Python traceback (pydantic validation, `TypeError`, `KeyError`) | bad config or a real bug | Nextflow terminates deliberately. Fix, then relaunch with `-resume`. |
| `Expected a 'RESOURCES:' line in command output but none was found` | the underlying biahub CLI crashed during its init step | read the init step's `slurm_output` log; usually a config validation error |
| `list_positions` returns nothing | the input has no HCS plate | build `0-convert` — `caveats.md` §1 |
| task shows a `(Pdb)` prompt in `.command.out` and never exits | `--cluster debug` drops into a post-mortem debugger inside the SLURM job | **kill the run** — see below. Do not wait for it. |

## A task stuck on `(Pdb)` — kill it, don't wait

`biahub track` (and any step run with `--cluster debug`) can enter a post-mortem
debugger *inside the SLURM job* when it raises. Nothing is attached to stdin, so
the task sits at the prompt until the wall-time limit kills it, holding its
allocation — a GPU allocation, for track. Signature in
`nextflow/slurm_output/<step>/*.out` or the work dir's `.command.out`:

```
TypeError: ...
> /.../work/80/9d98.../cupy/_core/core.pyx(1699)...
(Pdb)
```

`errorStrategy` cannot save you: the task reports no exit status until SLURM kills
it, and a time-limit kill lands in the retryable 130–145 range, so Nextflow retries
a deterministic failure up to `maxRetries` times — 6 × the wall limit of wasted
GPU time per position.

Stop the run rather than letting it burn out:

```bash
tmux send-keys -t nf_<DATASET> C-c     # Nextflow cancels its own SLURM jobs
squeue -u "$USER" -h -o "%j" | grep -c nf-   # confirm 0 remain
```

Then read the traceback *above* the `(Pdb)` line — that is the real error, and it
is a genuine bug or config error, not an infrastructure blip. Fix it and `-resume`;
completed upstream steps are cached, so only the failing step re-runs (verified:
changing only `track.yml` re-ran track alone). Tracked in biahub#309.

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

### Step 1 — rerun with `-resume` first

**This is now the expected fix, and usually the only one needed.**
[iohub#455](https://github.com/czbiohub-sf/iohub/pull/455) — on `main` since the
iohub git pin — replaces a torn shard instead of reading it back, and records
per-unit progress in a `.iohub-progress/` sibling of the output store. A retry
skips the units that finished and recomputes only the ones that did not, so the
read-modify-write that used to fail forever no longer happens.

```bash
bash ./run_mantis_v2.sh          # -resume is already in the script
```

A live preemption test on `2026_07_14_A549_MAP4_ZIKV_rerun` confirmed it: the
retry skipped 120 units, recomputed 81, zero codec errors.

Note the progress records sit **beside** the store, not inside it, so copying or
deleting a store no longer carries or clears its progress state.

### Step 2 — if it still fails identically, propose a repair; do not perform it

**Never delete zarr data from this skill.** If a position still fails with the
same checksum error after a clean `-resume`, stop and hand the user a written
plan. Report:

- the exact error and which position(s) it names,
- the specific paths you believe are corrupt,
- `du -sh` of each path so the user knows the volume at risk,
- what would be lost, and what `-resume` would recompute afterwards.

Then let the user decide and run it, or delegate to the **job-io-error-repair**
agent, which owns this procedure. The rule either must follow — worth knowing so
you can sanity-check a proposal:

- delete `<output.zarr>/ROW/COL/FOV/0/c/` — the zarr-v3 chunk directory **only**
- **keep** `<output.zarr>/ROW/COL/FOV/zarr.json` and
  `<output.zarr>/ROW/COL/FOV/0/zarr.json`

The scaffold comes from a *separate, cached* `init-*` Nextflow task. `-resume`
will not re-run it, and the worker opens the output FOV in `mode="r"` expecting
it to be there. Deleting the FOV directory wholesale turns a checksum error into
a `FileNotFoundError` and makes things worse.

Failed attempts' work dirs for those positions can go too, but **keep the
successful positions' work dirs and `.nextflow/`** so `-resume` reuses finished
work. These deletes can be hundreds of GB and are slow on Lustre — run them in
the background.

If a repair is needed at all on a current checkout, say so explicitly in the
report: it means the per-unit resume did not cover this case, which is worth
knowing upstream.

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
