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
tmux send-keys -t "$SESSION" "bash ./run_mantis_v2.sh" Enter
```

Tell the user: `tmux attach -t nf_<DATASET>`, detach with `Ctrl-b d`.

**Send the command bare — no `| tee`, no redirect.** A pipe buys nothing that
`.nextflow.log` and `nextflow/provenance.txt` don't already record. The console is
the user's view; yours is `.nextflow.log`.

**The pane inherits `CLAUDECODE=1` from the tool call that created the session**,
which puts Nextflow 26.04 into agent mode: one static `[PROCESS …]` line per task,
no live table, for a run the user watches for days. `run_mantis_v2.sh` handles it
with `unset CLAUDECODE`; if you ever launch Nextflow by hand, prefix it with
`env -u CLAUDECODE`. `NXF_AGENT_MODE=false` and `-ansi-log true` do **not** work —
see SKILL.md §8 and [nextflow#7478](https://github.com/nextflow-io/nextflow/issues/7478).

**Do not attach yourself** — attaching from a tool call gives you a terminal you
cannot read usefully and can steal the user's pane. **Read `.nextflow.log`**,
which Nextflow writes in the launch directory regardless of what the console
shows:

```bash
tail -n 50 <OUTPUT>/.nextflow.log
```

It carries what you need to follow progress: the launch command line, each task's
submission and completion with its work dir hash, exit statuses, retries, and the
final `WorkflowStats[succeededCount=...; failedCount=...; retriesCount=...]`
summary. Rotated per run as `.nextflow.log.1`, `.log.2`, … so the previous run's
log survives a relaunch.

`tmux capture-pane -p -t "$SESSION" | tail -40` still works as a last resort, but
prefer the log: the pane holds only what fits on screen, and with the progress
table redrawing in place it is not a transcript.

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

**`ProcessFailedException` in the log is not a failure signal.** It is written
for every *retried* task, immediately before the matching
`NOTE: ... -- Execution is retried (N)` line — one healthy run's log holds 33 of
them. Do not report a run as broken on the strength of that string. The terminal
marker is `Session aborted -- Cause: ...` (which also covers a Ctrl-C).

`trace.txt` is the authoritative per-task record: process name, tag (the
position), status, exit code, realized time and RSS, and — as of the `attempt`
field added to `trace.fields` in `nextflow.config` — the attempt number, as
column **15**. Count failures and retries from there, not from the console.

The 14 default columns keep their original positions, so `$4` is still the name,
`$5` the status and `$6` the exit code. Retried tasks also appear as repeated
`name` rows (a `FAILED` row and then a `COMPLETED` one), which is how retries had
to be counted before the column existed.

Per-step task logs (remember `.out` holds the output, `.err` is empty — see
`caveats.md` §9):

The directory is named for the STEP, not the process: `flat_field`, `deskew`,
`reconstruct`, `virtual_stain`, `assemble`, `track` (see `slurm_log_dir()` in
`nextflow/modules/common.nf`). There is no `run_flat_field/` directory.

```bash
ls -t <OUTPUT>/nextflow/slurm_output/*/ | head
tail -n 40 <OUTPUT>/nextflow/slurm_output/flat_field/*_<jobid>.out
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
awk -F'\t' 'NR>1 && $4 !~ /notify_step/ {split($4,a,":"); print a[length(a)], $5}' \
  <OUTPUT>/nextflow/trace.txt | sort | uniq -c | sort -k2
```

The `notify_step` filter matters: the pipeline's Slack notifications are ordinary
tasks, so six of them appear in `trace.txt` and in `report.html`/`timeline.html`
alongside the real work. They are local, sub-second, and not part of any step's
position count.

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
`biahub/utils/cluster.py`) and passed through the init step's `RESOURCES:` payload,
so it scales with the dataset rather than being hardcoded. If a step is far
outside the reference run's time for a comparable dataset, say so.

## Notifications

**The pipeline sends these itself — do not duplicate them.** `mantis-v2.nf` posts
to Slack via `biahub nf notify` (`nextflow/modules/notify.nf`):

| when | message | pings |
|---|---|---|
| run start | dataset, operator, pipeline, position count, the 6 step names, input, output, host, `max_positions` if capped | no |
| each step completes | dataset, step name, `[n/6]`, output path | no |
| run end | succeeded / cached / restarted (with SLURM's cause) / failed, wall time, assembled path | **yes** |
| run end, failed | the error message, plus a truncated `errorReport` tail | **yes** |

Only the run-end message @-mentions the operator, so a ping always means "this
needs you". Step messages are informational and stay quiet. The mention sits at
the END of the title, so every message opens with its emoji and dataset whether or
not it pings; position within `text` does not affect delivery.

Each kind of message has its own emoji, so the one that needs you is not six
lookalikes deep in a channel: :rocket: to start, :white_check_mark: per step,
:checkered_flag: complete, :x: failed, :warning: aborted.

The position count and the step list are reported once, at run start — they are
the same for every step, so repeating them per message would add nothing. Run
start is therefore sent once `list_positions` has produced the count (about 40s
in), not at graph-construction time; a config error that kills `list_positions`
itself yields only the failure message.

### Setting it up

**Both variables are optional.** They only decide whether messages reach Slack.
The pipeline runs identically without them: every message is printed instead of
posted, every exit status is still 0, and no step behaves differently. Never
block a launch on them, and never treat a missing one as an error.

Two environment variables, read from the launching shell and inherited by every
task. Neither is a pipeline parameter:

```bash
# ~/.bashrc
export BIAHUB_SLACK_WEBHOOK="https://hooks.slack.com/services/T.../B.../..."
export BIAHUB_SLACK_ID="U0A2ZH9CS8S"
```

`BIAHUB_SLACK_WEBHOOK` — the incoming-webhook URL for the channel the run should
report to. **This is a credential**: it lets anyone holding it post to that
channel, so it belongs in `~/.bashrc` only, never in the repo, a config file, a
command line, or a commit. Users do not create it themselves — **ask a biahub
developer (Ivan, Taylla) for the webhook URL.**

`BIAHUB_SLACK_ID` — the member ID of the person to @-mention when a run finishes
or fails. To find it: open Slack, click your avatar (or your name in a message)
→ **View full profile** → the **⋮ / More** button → **Copy member ID**. In the
Slack web app it is also the last path segment of your profile URL
(`.../team/U0A2ZH9CS8S`).

The format matters, and getting it wrong fails silently:

| | |
|---|---|
| ✅ `U0A2ZH9CS8S` | a member ID — 9+ characters, starts with `U` (or `W`) |
| ❌ `@Ivan Ivanov` | a display name; never pings via the API |
| ❌ `ivan.ivanov` | a username or email local part |
| ❌ `<@U0A2ZH9CS8S>` | already-wrapped mention — accepted, but write the bare ID |

A display name posts as literal text and notifies nobody, which is why the
notifier validates the ID against `^[UW][A-Z0-9]{6,}$` and warns rather than
posting something that silently reaches no one.

**If either variable is unset, offer to add it to the user's `~/.bashrc`** — say
what each one does, that both are optional, and where to get the webhook. Only
write the file if the user agrees, append rather than rewrite, and have them
`source ~/.bashrc` (or open a new shell) before launching, since the pipeline
reads the value from the launching shell:

```bash
cat >> ~/.bashrc <<'EOF'

# biahub Nextflow pipeline -> Slack notifications (both optional)
export BIAHUB_SLACK_WEBHOOK="https://hooks.slack.com/services/..."   # ask Ivan or Taylla
export BIAHUB_SLACK_ID="U0A2ZH9CS8S"                                # Slack profile -> Copy member ID
EOF
```

Check what is already set before offering, so an existing value is never
clobbered:

```bash
printenv BIAHUB_SLACK_WEBHOOK >/dev/null && echo "webhook set" || echo "webhook MISSING"
printenv BIAHUB_SLACK_ID      >/dev/null && echo "slack id set" || echo "slack id MISSING"
grep -c 'BIAHUB_SLACK' ~/.bashrc
```

`run_mantis_v2.sh` warns about a missing variable at launch and records the Slack
ID in `nextflow/provenance.txt`, so which was in effect is part of the run record.

The *name* in the run-start message is separate from the mention: it comes from
the account database on the cluster (the GECOS field, via `--operator`), not from
Slack. Turning a member ID into a display name needs a `users.info` call and a bot
token, which an incoming webhook cannot make — and an `<@U…>` mention would ping,
while run start is deliberately silent.

If either is unset, say so once in the plan so the user can export it before
launch, but **do not block on it**: without a webhook every message still prints
and every exit status is still 0. Note the asymmetry when reporting that — run
start and run end print to the console, but step messages run as tasks, so their
text goes only to `nextflow/slurm_output/<step>/*.out`. The pipeline warns about
this at launch.

If a notification itself fails, the reason is in
`<OUTPUT>/nextflow/.notify/notify.log` — the run-end message is sent during
session teardown, after Nextflow's console renderer is gone, so that file is the
only durable record. Silence there means every message was delivered.

### What is left for you to send

`templates/notify.sh` is now only for messages the pipeline cannot know about:

- **A terminating error you have diagnosed** — the step, the position, and the
  first lines of the traceback. The automatic run-end message reports *that* the
  run failed; you add *why*. Send it before you finish investigating.
- **A step retrying past a reasonable threshold** — any position on attempt 4 of
  `maxRetries = 5`, now readable straight from `trace.txt`'s `attempt` column.
  This is the early warning for a torn shard. Do **not** use a retry-rate
  percentage: retries measure preemption pressure on the `preempted` partition,
  not dataset health, and a busy day crosses any such threshold routinely.
- **The wrap-up** — channel rename result, iohub verification, shape/channels and
  size on disk. The pipeline knows none of this.

```bash
bash <SKILL>/templates/notify.sh --level error --ping \
  "❌ <DATASET>: run_deskew failed on C/4/001001" "$(tail -20 <OUTPUT>/nextflow/slurm_output/deskew/*_<jobid>.out)"
```

It takes `--level info|good|warn|error` and `--ping`, and delegates to
`biahub nf notify`. Also use it for hand-rolled reruns that bypass Nextflow.

Do not notify on individual preemptions; they are routine. The run-end message
already reports them as `restarted: N`, with the cause from `sacct` — a restarted
attempt was retried and is not a failure, and `sacct` is what distinguishes
preemption from a wall-time kill, since both exit 143.

Alongside Slack, surface the same information in the conversation, since the user
may be watching the terminal instead.

## Final summary

From `trace.txt`, report per step: tasks submitted, succeeded, failed, retried,
total and max wall time. Then:

```bash
<BIAHUB>/.venv/bin/python -c "
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
