// Cleanup of a finished run's intermediates (biahub#292).
//
// Opt-in via `--cleanup_intermediates auto|true|false` (default false).
// cleanup_decision() turns that into on/off at launch; the pipeline then hands
// this module the directories to delete and a trigger that fires only once the
// run's last step has finished. This module knows nothing about which steps
// those are.
//
// NO VERIFICATION, ON PURPOSE. The issue proposed checking the assembled plate
// against each source before deleting it, but most of what could be checked —
// positions, channel names, shapes — is written by `--init` before a single
// pixel is computed, so it passes on an empty store. Pixel spot-checks are
// fragile because concatenate's crop and channel selection decide what the
// assembled plate holds. The gate is the run itself instead: every compute
// step terminates the run once its retries are spent, so the trigger can only
// fire when every task of every step succeeded.
//
// A CLEANED RUN IS FINAL. The pipeline deletes the intermediate stores, the
// resume markers (`.iohub-progress`) beside the final stores, and — through
// Nextflow's `cleanup`, which it switches on alongside — the work directory.
// A later run in the same output directory therefore recomputes everything
// from the raw input rather than resuming: no Nextflow task is cached, and no
// concatenate write unit is skipped as already written.


// The concatenate settings in `concatenate_config` that select a SUBSET of the
// source data. Anything other than "all" (or a per-source list made only of
// "all") means the assembled plate holds less than the intermediates, so they
// are not duplicates and `auto` keeps them. Mirrors ConcatenateSettings, where
// every one of these defaults to "all".
def cropped_fields(concatenate_config) {
    def config = new org.yaml.snakeyaml.Yaml().load(new File(concatenate_config as String).text) ?: [:]
    def fields = ['time_indices', 'channel_names', 'X_slice', 'Y_slice', 'Z_slice']
    return fields.findAll { field ->
        def value = config[field]
        def takes_all = value == null || value == 'all' ||
            (value instanceof List && !value.isEmpty() && value.every { entry -> entry == 'all' })
        !takes_all
    }
}

// Resolve --cleanup_intermediates into [on: boolean, reason: String] at launch.
//
//   false (default)  off
//   true             on, even if concatenate.yml crops; needs --concatenate_config
//   auto             on only when concatenate.yml takes ALL the data, i.e. the
//                    intermediates are fully duplicated in the assembled plate;
//                    off if it crops or there is no assemble step
//
// The value is compared as text, never truth-tested: a param given on the
// command line arrives as a String, and the String "false" is truthy in Groovy.
// Decided while the graph is built rather than in a task, because the step
// list, the run-start message and the work-directory cleanup all depend on it,
// and a typo should fail before anything is submitted.
def cleanup_decision(value, concatenate_config) {
    def requested = value == null ? 'false' : value.toString().trim().toLowerCase()
    if (!(requested in ['auto', 'true', 'false'])) {
        error "--cleanup_intermediates must be auto, true or false, not '${value}'"
    }
    if (requested == 'false') {
        return [on: false, reason: 'off: --cleanup_intermediates false']
    }
    if (!concatenate_config) {
        if (requested == 'true') {
            error "--cleanup_intermediates true needs --concatenate_config: without assemble the reconstruction stores are the output."
        }
        return [on: false, reason: 'auto: no --concatenate_config, so the reconstruction stores are the output']
    }
    if (requested == 'true') {
        return [on: true, reason: 'on: --cleanup_intermediates true']
    }
    def cropped = cropped_fields(concatenate_config)
    if (cropped) {
        return [on: false, reason: "auto: concatenate.yml crops ${cropped.join(', ')}"]
    }
    return [on: true, reason: 'auto: concatenate.yml takes all the data']
}


// Validate and normalise the paths to delete. Called while the graph is built,
// so a bad path fails the run at launch rather than after days of compute.
//
// Every target must sit strictly INSIDE `root` (the run's --output): never the
// output directory itself, never anything outside it. That is the only thing
// standing between a wiring mistake and an `rm -rf` of someone else's data.
def cleanup_targets(targets, root) {
    if (!root) error "cleanup_targets: no output directory to confine the cleanup to"
    def root_path = java.nio.file.Paths.get(root as String).toAbsolutePath().normalize()
    return targets.collect { target ->
        if (!target) error "cleanup_targets: empty path in ${targets}"
        def path = java.nio.file.Paths.get(target as String).toAbsolutePath().normalize()
        if (path == root_path || !path.startsWith(root_path)) {
            error "cleanup_targets: refusing to delete ${path}: not inside the output directory ${root_path}"
        }
        path.toString()
    }
}


process cleanup_intermediates {
    label 'cpu_local'

    input:
    val targets
    val trigger

    output:
    stdout

    script:
    // No size report: these stores run to terabytes, and walking them to add up
    // their size costs longer than deleting them.
    def quoted = targets.collect { target -> "\"${target}\"" }.join(' ')
    """
    for target in ${quoted}; do
        if [ -e "\$target" ]; then
            rm -rf "\$target"
            echo "removed \$target"
        else
            echo "absent  \$target"
        fi
    done
    """
}


// Delete `targets` once `trigger` has emitted everything it will emit.
//
// take:
//   targets   list of paths, already checked by cleanup_targets()
//   trigger   gating channel — the mix of every final step's `done`
// emit:
//   done      fires once every target is gone
workflow cleanup_intermediates_wf {
    take:
    targets
    trigger

    main:
    cleanup_out = cleanup_intermediates(targets, trigger.collect().map { 'done' })
    cleanup_out.subscribe { stdout_text -> log.info "cleanup_intermediates:\n${stdout_text.trim()}" }

    emit:
    done = cleanup_out.map { 'done' }
}
