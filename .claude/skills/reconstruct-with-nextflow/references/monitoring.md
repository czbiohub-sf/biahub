# Running, monitoring, and notifying

## Where the Nextflow head process runs

The head process only submits SLURM jobs and waits — it is I/O-bound and cheap,
so a **login node is the right place for it**. It must survive disconnection,
hence tmux.

Reserve a compute node instead only if the user asks, or if a login-node process
reaper is killing the run:

```bash
salloc --partition=cpu --cpus-per-task=2 --mem=16G --time=3-00:00:00
```

Start the tmux session *inside* the allocation in that case, and note that the
run dies when the allocation ends — size `--time` for the whole pipeline, not
for one step.

## tmux

One session per dataset, created detached, command sent in the foreground of the
pane so the user sees live Nextflow output on attach:

```bash
SESSION="nf_<DATASET>"
tmux has-session -t "$SESSION" 2>/dev/null && echo "session exists — attach or kill it first"
tmux new-session -d -s "$SESSION" -c "<OUTPUT>"
tmux send-keys -t "$SESSION" \
  "bash ./run_mantis_v2.sh 2>&1 | tee -a nextflow/run_$(date +%Y%m%dT%H%M%S).log" Enter
```

Tell the user: `tmux attach -t nf_<DATASET>`, detach with `Ctrl-b d`.

**Do not attach yourself** — attaching from a tool call gives you a terminal you
cannot read usefully and can steal the user's pane. Read the tee'd log instead:

```bash
tail -n 50 <OUTPUT>/nextflow/run_*.log
tmux capture-pane -p -t "$SESSION" | tail -40    # if the log is not yet flushed
```

To confirm the run is alive:

```bash
tmux list-sessions
pgrep -fa "nextflow run .*mantis-v2.nf"
```

## What to poll

Poll every few minutes, not seconds — these runs take hours to days, and each
poll costs the user context.

```bash
squeue -u "$USER" -o "%.10i %.30j %.8T %.10M %.6D %R" | head -40   # SLURM view
squeue -u "$USER" -h -t RUNNING,PENDING | wc -l                    # queue depth
tail -n 30 <OUTPUT>/.nextflow.log                                  # head-process view
column -t <OUTPUT>/nextflow/trace.txt | tail -20                   # per-task record
```

`trace.txt` is the authoritative per-task record: process name, tag (the
position), status, exit code, realized time and RSS, and attempt number. Count
failures and retries from there, not from the console.

Per-step task logs (remember `.out` holds the output, `.err` is empty — see
`caveats.md` §9):

```bash
ls -t <OUTPUT>/nextflow/slurm_output/*/ | head
tail -n 40 <OUTPUT>/nextflow/slurm_output/run_flat_field/*_<jobid>.out
```

Position count per step. The plate nests `<store>.zarr/<row>/<col>/<fov>`, so glob
three levels **inside** the named store:

```bash
DS=<DATASET>
for d in 0-flatfield 1-deskew 2-reconstruct 3-virtual-stain 5-assemble 4-track; do
  printf "  %-16s %s\n" "$d" "$(ls -d "<OUTPUT>/$d/$DS.zarr"/*/*/*/ 2>/dev/null | wc -l)"
done
```

Two traps, both hit for real:

- **Name the store.** A bare `$d/*.zarr/*/*/*/` overcounts `2-reconstruct`, which
  also holds `transfer_function.zarr` (10 instead of 8 for an 8-position plate).
- **This counts scaffolds, not finished work.** Each step's cached `init-*` task
  creates every position's metadata up front, so a step shows its full position
  count from the moment it starts — a step whose every task *failed* still reads
  8/8 here. Use it for "has this step started", never for "is this step done".
  `trace.txt` is the only authoritative source for completion:

```bash
awk -F'\t' 'NR>1{split($4,a,":"); print a[length(a)], $5}' <OUTPUT>/nextflow/trace.txt \
  | sort | uniq -c | sort -k2
```

## Rough expectations

Steps run strictly in sequence; within a step, positions fan out with
`maxForks = 30`. Order and relative cost:

| step | shape | note |
|---|---|---|
| `run_flat_field` | fan-out, CPU (`preempted`) | first, so it absorbs most preemption |
| `run_deskew` | fan-out, CPU | CPU on current main, not GPU |
| `compute_transfer_function` + `run_apply_inv_tf` | one-shot + fan-out, CPU | TF is quick; apply is the bulk |
| `run_virtual_stain_preprocess` + `run_virtual_stain` | one-shot + fan-out, **GPU** | `gpu` partition, effectively not preempted; longest per-position step |
| `run_concatenate` | **single job**, one reserved node | longest single job; whole-plate I/O |
| `run_track` | fan-out, GPU (cellpose) | reads the assembled plate |

Per-step wall-time is derived from data volume (`estimate_time_minutes()` in
`biahub/cli/utils.py`) and passed through the init step's `RESOURCES:` payload,
so it scales with the dataset rather than being hardcoded. If a step is far
outside the reference run's time for a comparable dataset, say so.

## Notifications

Use `templates/notify.sh`. It posts to the Slack webhook in
`$BIAHUB_SLACK_WEBHOOK` and, when that is unset, prints to the terminal so
nothing is lost.

```bash
export BIAHUB_SLACK_WEBHOOK="https://hooks.slack.com/services/..."   # user sets this
bash <SKILL>/templates/notify.sh "✅ <DATASET>: mantis-v2 complete — 5-assemble written"
```

If the variable is unset, mention it once in the plan so the user can export it
before launch; do not block on it.

Send a message on:

- **Completion** — with the summary from the wrap-up step.
- **A terminating error** — with the step, the position, and the first lines of
  the traceback. Do not wait to finish diagnosing.
- **Retries past a threshold** — e.g. a step whose retry count exceeds ~20% of
  its tasks, or any position on attempt 4 of 5. This is the early warning for a
  torn shard.

Do not notify on individual preemptions; they are routine.

Alongside Slack, surface the same information in the conversation, since the
user may be watching the terminal instead.

## Final summary

From `trace.txt`, report per step: tasks submitted, succeeded, failed, retried,
total and max wall time. Then:

```bash
uv run --project <BIAHUB> python -c "
from iohub.ngff import open_ome_zarr
with open_ome_zarr('<OUTPUT>/5-assemble/<DATASET>.zarr', mode='r') as p:
    pos = list(p.positions())
    print(len(pos), 'positions'); print(p.channel_names)
    print(pos[0][1].data.shape, pos[0][1].data.dtype)"
du -sh <OUTPUT>/5-assemble/<DATASET>.zarr
```

Point the user at `<OUTPUT>/nextflow/report.html` (resource usage per process)
and `timeline.html` (where the wall time went). They are plain files on Lustre —
the user opens them locally; do not try to render them.
