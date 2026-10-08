# Error handling and recovery

## Triage: read before you act

| Symptom | Likely cause | Action |
|---|---|---|
| many records with `job failed: ... CANCELLED ... DUE TO TIME LIMIT` | the partition is busy, or volumes are large | relaunch with `--resume` and an sbatch file setting `#SBATCH --time=...` |
| the sequential (`previous_timepoint`) job cancelled part-way | its time limit was too short for T | relaunch with `--resume` and a longer limit; finished timepoints are kept and the chain continues |
| `Input/output error` / `Stale file handle` reading the stores | the filesystem, not biahub | check the mount on the node and a compute node (`srun ... ls <path>`); report to HPC; nothing to fix in the run |
| `--resume: settings_sha256 changed` | the config or inputs changed since the run | rerun without `--resume` into a fresh output directory |
| Nextflow stops with `Error executing process > '...'` | a task failed with a non-signal exit (a real error, not preemption) | read that task's `.command.err` (work dir in the message) or its log under `<out>/nextflow/slurm_output/`; fix, then relaunch the same script (`-resume` reruns only what failed) |
| Nextflow `the config changed since --init` / `run with --init first` | the config file was edited after the run started, or the run folder was removed | relaunch into a new output directory (or drop `-resume` so init runs again) |
| `Manual registration is interactive` at launch | `method: manual` given to `registration.nf` | run manual with the plain CLI in a session with a display, then apply its file with `TRANSFORMS=` |
| a timepoint `unreliable` with note `job failed: <error>` (e.g. `RuntimeError: blosc encoded value is invalid`) | that timepoint's job raised (unreadable chunk, I/O fault); the run carried on with a stand-in (Nextflow retried the task twice first) | check the data at that timepoint; after fixing it, relaunch with resume (`-resume` / `--resume`): the failed timepoint is redone, finished ones are kept |
| a timepoint `unreliable` with note `empty frame (no data)` | the raw acquisition is empty there | expected; nothing moves at that timepoint |
| a timepoint `unreliable` with `too few matches` / `too few nodes` | too few beads detected | check the bead thresholds on that timepoint; or substitute another method's result for it |
| every timepoint `unreliable` with a correlation-based method | thresholds tuned for the bead-overlap score | read `methods.md`; check a few timepoints visually |

## Rules

- Keep the number of jobs you have on the queue modest: an overloaded partition makes jobs
  hit their time limit, and a cancelled timepoint looks like a failed estimate.
- An sbatch time limit wins over every phase's default (plain CLI). Nextflow retries a task
  that hit its time limit with twice the time, and a preempted one with the same request.
- Never delete a run folder's `timepoints/` or `repairs/` (`<out>/transforms/` for
  `-o <out>/transforms.yml`) to "clean up" a run that may be resumed; a fresh run (without
  `--resume`) clears them itself, and only its own.
- Results at timepoints with very few beads can differ by node (floating point): compare
  runs at the score's resolution.
