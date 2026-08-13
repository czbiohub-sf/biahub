# Error handling and recovery

## Triage: read before you act

Never delete zarr data or relaunch before classifying the failure:

```bash
cd <OUTPUT>
tail -n 200 .nextflow.log                       # rotates to .1, .2, ...
grep -n "ERROR\|Caused by\|Command error\|terminated with" .nextflow.log | tail -40
```

Nextflow names the failing process, the position, and a `Work dir:`.
`<WORKDIR>/.command.sh` has the exact biahub command, `.exitcode` the status.
The real traceback is **not** in the work dir (`caveats.md` §9) — it is in
`<OUTPUT>/nextflow/slurm_output/<step>/*_<jobid>.out`.

## Decision table

| signature | cause | action |
|---|---|---|
| exit 143 / 137 / 140, no traceback | SLURM preemption, timeout, or OOM | none — Nextflow retries (exit 130–145, `maxRetries = 5`). Expected on the `preempted` partition. |
| exit 143 repeatedly on the *same* position until retries exhaust | task too slow or too large for its resource envelope | check `trace.txt` for realized time/RSS; the run may need a wall-time or memory bump |
| `RuntimeError: the checksum is invalid` / `encoded shard is smaller than ... its index` / `blosc encoded value is invalid` | torn shard from a killed write | **rerun with `-resume` first** — see below |
| `OSError` / `IOError` / "Input/output error" on a `.zarr` path, or a truncated read while decoding a chunk | Lustre EIO — usually the same torn-shard condition | same |
| `FileNotFoundError: Dataset directory not found at .../<zarr>/ROW/COL/FOV` | a previous cleanup deleted the metadata scaffold too | **job-io-error-repair agent** (recreates the scaffold from a sibling) |
| exit 1/2 with a Python traceback (pydantic validation, `TypeError`, `KeyError`) | bad config or a real bug | Nextflow terminates deliberately. Fix, relaunch with `-resume`. |
| `Expected a 'RESOURCES:' line in command output but none was found` | the biahub CLI crashed during its init step | read the init step's `slurm_output` log; usually a config validation error |
| `list_positions` returns nothing | input has no HCS plate | build `0-convert` — `caveats.md` §1 |
| a `(Pdb)` prompt in the task output, task never exits | `--cluster debug` post-mortem debugger inside the SLURM job | **kill the run** — see below |

## A task stuck on `(Pdb)` — kill it, don't wait

Steps run with `--cluster debug` (track, concatenate) drop into a post-mortem
debugger inside the SLURM job when they raise. Nothing is attached to stdin,
so the task holds its allocation (a GPU, for track) until the wall-time kill —
which lands in the retryable 130–145 range, so Nextflow retries the
deterministic failure up to 5 more times. Stop the run instead:

```bash
tmux send-keys -t nf_<DATASET> C-c           # Nextflow cancels its SLURM jobs
squeue -u "$USER" -h -o "%j" | grep -c nf-   # confirm 0 remain
```

The real error is the traceback *above* the `(Pdb)` line — a genuine bug or
config error. Fix it and `-resume`; completed steps are cached, only the
failing step re-runs. Tracked in biahub#309.

## Lustre EIO / torn shards

A task killed mid-write leaves a truncated shard. Every output array overhangs
its shards in Z/Y/X, so every write is a read-modify-write: the next attempt
reads the truncated shard back, the CRC32C check fails, and it aborts — and
plain retries fail identically, forever.

### Step 1 — rerun with `-resume` first

**This is the expected fix, and usually the only one needed.**
[iohub#455](https://github.com/czbiohub-sf/iohub/pull/455) (in the pinned
iohub) replaces a torn shard instead of reading it back and records per-unit
progress in a `.iohub-progress/` *sibling* of the output store, so a retry
recomputes only the unfinished units.

```bash
bash ./run_mantis_v2.sh          # -resume is already in the script
```

### Step 2 — if it still fails identically, propose a repair; do not perform it

**Never delete zarr data from this skill.** Hand the user a written plan: the
exact error and position(s), the paths believed corrupt, `du -sh` of each,
what would be lost, and what `-resume` recomputes afterwards. Let the user run
it or delegate to the **job-io-error-repair** agent. Sanity-check any proposal
against this rule:

- delete `<output.zarr>/ROW/COL/FOV/0/c/` — the chunk directory **only**
- **keep** both `zarr.json` files — the scaffold comes from a separate cached
  `init-*` task that `-resume` will not re-run; deleting it turns a checksum
  error into a `FileNotFoundError`

Keep the successful positions' work dirs and `.nextflow/` so `-resume` reuses
finished work. If a repair is needed at all on a current checkout, say so —
it means the per-unit resume did not cover the case, which matters upstream.

**Corrupt *input* is different.** If the error is raised while *reading* the
input zarr, the source is damaged — do not clean the output; report it. The
input may need re-export from the instrument.

## Restarting

Always: `cd <OUTPUT> && bash ./run_mantis_v2.sh`, relaunched in the tmux
session. `-resume` is invalidated by editing the biahub checkout, editing a
config, or changing a `--param` — if a resume unexpectedly recomputes
everything, one of those changed. To force one step to recompute, delete that
step's output for the affected positions and their work dirs — not the whole
`.nextflow/` cache.

## When to escalate to the user

- Retries exhausted on any step.
- The same position failing identically more than twice after a repair.
- Any deletion over ~100 GB.
- A traceback you cannot map to the table above.
- A step running more than ~3× the reference run's time.
