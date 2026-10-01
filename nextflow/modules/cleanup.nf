// Cleanup of a finished run's intermediates (biahub#292).
//
// Opt-in via `--cleanup_intermediates`. The pipeline hands this module the
// directories to delete and a trigger that fires only once the run's last step
// has finished; this module knows nothing about which steps those are.
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
// nextflow.config's `cleanup` — the work directory. A later run in the same
// output directory therefore recomputes everything from the raw input rather
// than resuming: no Nextflow task is cached, and no concatenate write unit is
// skipped as already written.


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
