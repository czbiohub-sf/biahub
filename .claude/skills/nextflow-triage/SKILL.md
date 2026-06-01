---
name: nextflow-triage
description: Diagnose a failed Nextflow / Slurm / AWS Batch run without jumping to the first hypothesis. Pulls logs, classifies against a failure playbook (NODE_FAIL, OOM, time limit, cache invalidation, S3 throttle, container pull), checks memory/cpu against actual node specs, enumerates ≥2 hypotheses with evidence, and flags cache-invalidation impact before any config edit. Use when a Nextflow pipeline failed on Bruno HPC or AWS ParallelCluster.
---

# Nextflow Triage

Evidence-first diagnosis. Do NOT propose a fix until classification and ≥2 hypotheses are written down.

## 1. Gather evidence

From the run directory:

```bash
tail -200 .nextflow.log
awk -F'\t' 'NR==1 || $5 != "COMPLETED"' trace.txt   # non-completed tasks
ls -la work/<task-hash>/.command.*                    # per-task logs
tail -100 work/<task-hash>/.command.err
```

From Slurm (Bruno / ParallelCluster):

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,ReqMem,MaxRSS,Elapsed,NodeList
scontrol show job <jobid> 2>/dev/null || echo "(job aged out)"
```

From AWS Batch:

```bash
aws batch describe-jobs --jobs <jobid>
# log stream ID → CloudWatch
aws logs get-log-events --log-group-name /aws/batch/job --log-stream-name <stream>
```

## 2. Classify against playbook

Map the evidence to one (or more) categories:

| Category | Signal | First check |
|---|---|---|
| NODE_FAIL | Slurm state=NODE_FAIL, no task stdout | `sinfo` — partition healthy? |
| OOM | MaxRSS ≈ ReqMem, `.command.err` has `Killed` | bump `memory`, verify against node spec |
| Time limit | state=TIMEOUT, Elapsed ≈ ReqTime | `time` directive, or chunk the input |
| S3 throttle | `SlowDown` / 503 in stderr | add retry/backoff, reduce parallel GETs |
| Cache invalidation | Everything re-running from scratch | diff against last successful `.nextflow/cache` |
| Container pull | `ErrImagePull` / `toomanyrequests` | registry throttle, or wrong image URI |
| Auth | `AccessDenied` on S3 / EFS | IAM role, `AWS_PROFILE` override, EFS mount |

## 3. Check memory/cpus vs node specs

Before proposing to bump resources, verify the node can actually serve them:

```bash
# Slurm
sinfo -o "%P %c %m %l"         # partition, cpus, memory, maxtime
# AWS ParallelCluster: check the instance-type spec
#   c5.2xlarge  = 8 vCPU / 16 GB
#   r5.4xlarge  = 16 vCPU / 128 GB
```

Leave ~10% headroom. **Undersized requests silently fail on ParallelCluster** rather than erroring clearly — always verify.

## 4. Hypotheses (minimum 2)

Write down ≥2 candidate root causes with the evidence supporting each. Example:

> **H1 (OOM)** — strong. `MaxRSS=14.8G`, `ReqMem=16G`, `.command.err` ends with "Killed". Node is `c5.2xlarge` (16 GB). Need r5 class or chunk the input.
>
> **H2 (S3 throttle)** — weak. All 12 parallel tasks hit stderr at the same timestamp, but only one task has stderr content and it's not `SlowDown`. Timing is coincidental.

Propose only the strongest hypothesis's fix unless the user asks for both.

## 5. Cache invalidation impact

Before editing any config / process script / work-dir path, list which cache keys the change invalidates:

> Changing `process.memory` on `QC_METRICS` invalidates QC_METRICS and all downstream: SEGMENT, TRACK, REPORT.
> Expected rerun cost: ~6h on 384 FOVs.

Wait for user confirmation before editing.

## 6. Apply minimal fix and retry

```bash
# edit only the necessary config
nextflow run <main.nf> -resume -profile <profile>
```

Report the new trace summary and whether the hypothesis held.

## 7. Commit hygiene

Config fix goes in its own commit. Do not bundle with unrelated changes. If the fix required a Nextflow version bump or major refactor, flag that explicitly.
