// Registration subworkflows: estimate-transform (init, then estimate -> flag ->
// repair / sweep -> finalize) and apply-transform (init, then one task per position).
//
// This module is PATH-AGNOSTIC, like every other step module: callers pass the
// stores, the position glob, the config and the output paths explicitly. The
// standalone entry point (registration.nf) owns the layout.
//
// THE CLI OWNS THE LOGIC, NEXTFLOW OWNS THE FAN-OUT. `biahub estimate-transform
// --init` checks the config, plans every position and prints a PLAN line: the
// positions, the timepoints, whether they are estimated in one sequential task
// (`seed_from: previous_timepoint`), and each phase's resources. Each task then
// runs one `--step` in-process (`--cluster debug`): estimate one timepoint (or one
// position's whole series), flag one position (its PLAN line names the timepoints
// to repair and sweep), repair / sweep one timepoint, and finalize, which writes
// the transforms file. Every step writes a small record in the run folder (next to
// the transforms file, named after it), and `--resume` makes a retried task skip
// the timepoints it already finished -- what matters for the sequential task on the
// preemptible partition.
//
// CACHE KEYS ON CONTENT, NOT NAMES. Nextflow caches a task by its inputs, and a path
// passed as `val` is just a string: a config or transforms file rewritten in place (e.g.
// `substitute-transforms -o <same file>`) would let -resume reuse every task. So the
// estimate tasks take the config's content hash, and the apply tasks the transforms
// file's (finalize prints the hash of the file it writes).
//
// apply-transform follows deskew: `--init` creates the output plate and prints
// RESOURCES for one position's task; each task writes one position into it. Its
// init reads the transforms file, so it runs after estimation, not in an up-front
// init phase.

include {
    parse_resources; slurm_logs; slurm_log_dir; slurm_output_readme; retry_time; retry_memory
} from './common'


// The JSON payload of the last `PLAN:` line the CLI printed.
//
// JsonSlurperClassic, not JsonSlurper: the latter returns LazyMaps, which are filled
// in on first access and are not thread-safe, and Nextflow hashes the inputs of
// concurrent tasks that share one -- an intermittent "error while creating task
// hash" with a corrupted map in the log.
def parse_plan(stdout_text) {
    def matching = stdout_text.trim().readLines().findAll { line -> line.startsWith('PLAN:') }
    if (!matching) {
        error "Expected a 'PLAN:' line in estimate-transform output but none was found."
    }
    return new groovy.json.JsonSlurperClassic().parseText(matching.last().replace('PLAN:', '').trim())
}

// A step whose jobs raised exits 3 after recording them: retry it (a transient read error
// recovers), then let the run finish with those timepoints as stand-ins ('ignore': the run
// goes on, and a failed task is never cached, so a later -resume redoes it). Every other
// exit follows main's rule (nextflow.config): retry signal exits, terminate on errors.
def step_error_strategy(task) {
    if (task.exitStatus == 3) {
        return task.attempt <= 2 ? 'retry' : 'ignore'
    }
    return task.exitStatus in (130..145) + [Integer.MAX_VALUE] ? 'retry' : 'terminate'
}

// One task's resources as a plain map of ints, the shape parse_resources returns.
def task_resources(res) {
    return [cpus: res.cpus as int, mem_gb: res.mem_gb as int, time_minutes: res.time_minutes as int]
}

// `-r <reference store>/<positions>` for a cross registration; nothing for a
// stabilization (an empty reference_zarr). One reference position serves every
// moving position; several are paired with them by row/col/fov.
def reference_arg(reference_zarr, positions) {
    return reference_zarr ? "-r \"${reference_zarr}\"/${positions}" : ''
}

// One estimate-transform step for one position, in this task's process.
def step_command(step, position, t, moving_zarr, reference_zarr, positions, config, transforms) {
    def timepoints = t >= 0 ? "--timepoints ${t}" : ''
    return """
    biahub estimate-transform --step ${step} ${timepoints} --cluster debug --resume \\
        -m "${moving_zarr}/${position}" ${reference_arg(reference_zarr, positions)} \\
        -c "${config}" \\
        -o "${transforms}"
    """
}


process init_estimate_transform {
    label 'cpu_local'

    input:
    val moving_zarr
    val reference_zarr
    val positions
    val config
    val config_hash
    val transforms
    val trigger

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('estimate_transform')}"
    biahub estimate-transform --init \\
        -m "${moving_zarr}"/${positions} ${reference_arg(reference_zarr, positions)} \\
        -c "${config}" \\
        -o "${transforms}"
    """
}

// t = -1: the position's whole series (a propagated run).
process estimate_timepoints {
    tag "${position} t=${t >= 0 ? t : 'all'}"
    label 'cpu'
    errorStrategy { step_error_strategy(task) }
    clusterOptions { slurm_logs('estimate_transform') }
    cpus { meta.cpus }
    memory { retry_memory(meta.mem_gb, task) }
    time { retry_time(meta.time_minutes, task) }

    input:
    tuple val(position), val(t), val(meta)
    val moving_zarr
    val reference_zarr
    val positions
    val config
    val config_hash
    val transforms

    output:
    val position

    script:
    step_command('estimate', position, t, moving_zarr, reference_zarr, positions, config, transforms)
}

// Reads only the position's records: cheap, on the head node.
process flag_position {
    tag "${position}"
    label 'cpu_local'
    // Never cached: it reads the estimates, which a -resume may have redone. Cheap.
    cache false

    input:
    val position
    val moving_zarr
    val reference_zarr
    val positions
    val config
    val config_hash
    val transforms

    output:
    tuple val(position), stdout

    script:
    """
    biahub estimate-transform --step flag \\
        -m "${moving_zarr}/${position}" ${reference_arg(reference_zarr, positions)} \\
        -c "${config}" \\
        -o "${transforms}"
    """
}

// step = repair or sweep, one flagged timepoint.
process refine_timepoint {
    tag "${step} ${position} t=${t}"
    label 'cpu'
    errorStrategy { step_error_strategy(task) }
    // Never cached: it depends on the estimates, which a -resume may have redone. It runs
    // with --resume, so a timepoint already repaired / swept returns at once.
    cache false
    clusterOptions { slurm_logs('estimate_transform') }
    cpus { meta.cpus }
    memory { retry_memory(meta.mem_gb, task) }
    time { retry_time(meta.time_minutes, task) }

    input:
    tuple val(step), val(position), val(t), val(meta)
    val moving_zarr
    val reference_zarr
    val positions
    val config
    val config_hash
    val transforms

    output:
    val position

    script:
    step_command(step, position, t, moving_zarr, reference_zarr, positions, config, transforms)
}

process finalize_estimate_transform {
    label 'cpu_local'
    // Never cached: it reads every record, which a -resume may have changed. Cheap.
    cache false

    input:
    val gate
    val moving_zarr
    val reference_zarr
    val positions
    val config
    val config_hash
    val transforms

    output:
    stdout

    script:
    // the last line is the written file's content hash, for the apply tasks' cache key
    """
    biahub estimate-transform --step finalize \\
        -m "${moving_zarr}"/${positions} ${reference_arg(reference_zarr, positions)} \\
        -c "${config}" \\
        -o "${transforms}"
    sha256sum "${transforms}" | cut -d' ' -f1
    """
}


process init_apply_transform {
    label 'cpu_local'

    input:
    val moving_zarr
    val reference_zarr
    val positions
    val transforms
    val transforms_hash
    val output_zarr
    val extra_args

    output:
    stdout

    script:
    """
    mkdir -p "${slurm_log_dir('apply_transform')}"
    ${slurm_output_readme('apply_transform', output_zarr)}
    biahub apply-transform --init \\
        -m "${moving_zarr}"/${positions} ${reference_arg(reference_zarr, positions)} \\
        -c "${transforms}" \\
        -o "${output_zarr}" ${extra_args}
    """
}

process run_apply_transform {
    tag "${position}"
    label 'cpu'
    clusterOptions { slurm_logs('apply_transform') }
    cpus { meta.cpus }
    memory { retry_memory(meta.mem_gb, task) }
    time { retry_time(meta.time_minutes, task) }

    input:
    tuple val(position), val(meta)
    val moving_zarr
    val reference_zarr
    val positions
    val transforms
    val transforms_hash
    val output_zarr
    val extra_args

    output:
    val position

    script:
    // --resume: a preempted task finishes the write it is in and stops early, so the
    // retry (or a later `nextflow -resume`) rewrites only the (t, c) units it had not
    // finished; the completion records are keyed by the transforms and apply options.
    """
    biahub apply-transform --cluster debug --resume \\
        -m "${moving_zarr}/${position}" ${reference_arg(reference_zarr, positions)} \\
        -c "${transforms}" \\
        -o "${output_zarr}" ${extra_args}
    """
}


// Check the estimate config and plan the run. Metadata-only.
//
// take:
//   moving_zarr     plate holding the moving channel
//   reference_zarr  plate holding the reference channel ('' to stabilize)
//   positions       position glob to estimate on, e.g. 'C/1/000000' (one shared
//                   transform list, e.g. the beads well) or '*/*/*' (one list each)
//   config          estimate-transform settings YAML
//   config_hash     its content hash (the tasks' cache key; see the header)
//   transforms      the transforms file to write
//   trigger         gating channel -- init starts once this emits
// emit:
//   plan            the PLAN payload (positions, time_indices, propagated, resources)
workflow estimate_transform_init_wf {
    take:
    moving_zarr
    reference_zarr
    positions
    config
    config_hash
    transforms
    trigger

    main:
    init_out = init_estimate_transform(
        moving_zarr, reference_zarr, positions, config, config_hash, transforms,
        trigger.collect().map { 'done' }
    )
    plan = init_out.map { stdout_text ->
        def p = parse_plan(stdout_text)
        if (p.interactive) {
            error "Manual registration is interactive (napari and a terminal): run " +
                "`biahub estimate-transform` in a session with a display, then apply its " +
                "file with --transforms."
        }
        p
    }

    emit:
    plan = plan
}


// Estimate every position's timepoints, flag, repair / sweep, and write the file.
//
// take:
//   plan, moving_zarr, reference_zarr, positions, config, config_hash, transforms  as above
//   prev_done   gating channel -- estimation starts once this emits
// emit:
//   done        the written transforms file's content hash (the apply tasks' cache key)
workflow estimate_transform_run_wf {
    take:
    plan
    moving_zarr
    reference_zarr
    positions
    config
    config_hash
    transforms
    prev_done

    main:
    // Gates are mapped to a token before combining; see deskew_run_wf.
    estimate_items = plan
        .combine(prev_done.map { 'done' })
        .flatMap { p, _gate ->
            p.positions.collectMany { pos ->
                p.propagated
                    ? [[pos, -1, task_resources(p.resources.estimate)]]
                    : p.time_indices.collect { t -> [pos, t, task_resources(p.resources.estimate)] }
            }
        }
    estimated = estimate_timepoints(
        estimate_items, moving_zarr, reference_zarr, positions, config, config_hash, transforms
    ) | collect

    // Flagging reads the whole run's scores, so it waits for every estimate. Some or all
    // estimate tasks may have been ignored (their jobs raised, the failure recorded): then
    // `collect` emits nothing, hence ifEmpty -- the run still flags and writes the file.
    to_flag = plan
        .combine(estimated.ifEmpty(['none']).map { 'done' })
        .flatMap { p, _gate -> p.positions }
    flags = flag_position(
        to_flag, moving_zarr, reference_zarr, positions, config, config_hash, transforms
    )

    refine_items = flags
        .combine(plan)
        .flatMap { pos, stdout_text, p ->
            def f = parse_plan(stdout_text)[pos]
            f.repair.collect { t -> ['repair', pos, t, task_resources(p.resources.repair)] } +
                f.sweep.collect { t -> ['sweep', pos, t, task_resources(p.resources.sweep)] }
        }
    refined = refine_timepoint(
        refine_items, moving_zarr, reference_zarr, positions, config, config_hash, transforms
    )

    // Finalize after every position is flagged and every repair / sweep is done
    // (there may be none: `collect` of an empty channel emits nothing, hence ifEmpty).
    gate = flags.collect().map { 'flagged' }
        .combine(refined.collect().ifEmpty(['none']).map { 'refined' })
        .map { _flagged, _refined -> 'done' }
    written = finalize_estimate_transform(
        gate, moving_zarr, reference_zarr, positions, config, config_hash, transforms
    )

    emit:
    done = written.map { stdout_text -> stdout_text.trim().readLines().last().trim() }
}


// Create the output plate. Reads the transforms file, so it runs once it exists.
//
// take:
//   moving_zarr, reference_zarr  as above ('' reference: stabilization)
//   positions     position glob to apply to, e.g. '*/*/*'
//   transforms       the transforms file
//   transforms_hash  its content hash, once it exists (the cache key; see the header)
//   output_zarr      the output plate
//   extra_args       further apply-transform options (e.g. '--crop-to-overlap')
// emit:
//   resources     the RESOURCES payload sizing one position's task
//   done          fires once the output plate exists
workflow apply_transform_init_wf {
    take:
    moving_zarr
    reference_zarr
    positions
    transforms
    transforms_hash
    output_zarr
    extra_args

    main:
    init_out = init_apply_transform(
        moving_zarr, reference_zarr, positions, transforms, transforms_hash, output_zarr,
        extra_args
    )

    emit:
    resources = init_out.map { stdout_text -> parse_resources(stdout_text) }
    done      = init_out.map { 'done' }
}


// Fan out one apply-transform task per position.
//
// take:
//   position_keys  collected channel of the position keys to write
//   moving_zarr, reference_zarr, positions, transforms, transforms_hash, output_zarr,
//   extra_args     as above
//   resources      RESOURCES payload from apply_transform_init_wf
//   prev_done      gating channel -- compute starts once this emits
workflow apply_transform_run_wf {
    take:
    position_keys
    moving_zarr
    reference_zarr
    positions
    transforms
    transforms_hash
    output_zarr
    extra_args
    resources
    prev_done

    main:
    pos_meta = position_keys
        .flatMap { items -> items }
        .combine(resources)
        .combine(prev_done.map { 'done' })
        .map { pos, meta, _gate -> [pos, meta] }

    applied = run_apply_transform(
        pos_meta, moving_zarr, reference_zarr, positions, transforms,
        transforms_hash.first(), output_zarr, extra_args
    ) | collect

    emit:
    done = applied
}
