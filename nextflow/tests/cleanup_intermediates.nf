#!/usr/bin/env nextflow

// Asserts what --cleanup_intermediates deletes and what it refuses to (biahub#292).
//
// WHY THIS EXISTS. The cleanup is an `rm -rf` on paths the pipeline computes,
// so two things stand between a mistake and deleting data nothing else holds:
// cleanup_decision(), which must turn cleanup OFF for `false` given as the
// String the command line delivers and for a concatenate config that crops;
// and the guard in cleanup_targets(), which keeps every path inside --output.
// The first two blocks check those directly; the third runs
// cleanup_intermediates_wf on a stand-in run directory and checks that the
// targets are gone and the final store next to them is not. Runs locally in
// seconds:
//     nextflow run nextflow/tests/cleanup_intermediates.nf
// It exits non-zero if either regresses.

nextflow.enable.dsl = 2

params.output = "${workflow.workDir}/cleanup_intermediates_test/run"

include { cleanup_decision; cleanup_targets; cleanup_intermediates_wf } from '../modules/cleanup'

def check(label, ok) {
    if (!ok) {
        error "FAIL: ${label}"
    }
    println "ok  ${label}"
}

def refused(targets, root) {
    try {
        cleanup_targets(targets, root)
        return false
    }
    catch (Exception _e) {
        return true
    }
}

def decision_refused(value, config) {
    try {
        cleanup_decision(value, config)
        return false
    }
    catch (Exception _e) {
        return true
    }
}

// Write a concatenate config and return its path.
def concatenate_yml(name, text) {
    def file = new File(new File(params.output as String).parentFile, "configs/${name}.yml")
    file.parentFile.mkdirs()
    file.text = text
    return file.path
}

def touch(path) {
    def file = new File(path)
    file.parentFile.mkdirs()
    file.text = 'x'
}

workflow {
    def root = params.output

    // The decision. Command-line values arrive as Strings, so each is checked
    // both ways; "false" must never count as on.
    def full     = concatenate_yml('full',     'time_indices: all\nchannel_names: all\nchunks_czyx: [1, 16, 256, 256]\n')
    def per_src  = concatenate_yml('per_src',  'channel_names: [all, all, all]\nZ_slice: all\n')
    def bare     = concatenate_yml('bare',     'chunks_czyx: [1, 16, 256, 256]\n')
    def cropped  = concatenate_yml('cropped',  'time_indices: [0, 1, 2]\nZ_slice: [10, 40]\n')
    def channels = concatenate_yml('channels', 'channel_names: [all, [Phase3D], all]\n')
    check('false (String) is off',              !cleanup_decision('false', full).on)
    check('false (Boolean) is off',             !cleanup_decision(false, full).on)
    check('unset is off',                       !cleanup_decision(null, full).on)
    check('true (String) is on',                cleanup_decision('true', full).on)
    check('true (Boolean) is on',               cleanup_decision(true, full).on)
    check('true is on even when cropping',      cleanup_decision('true', cropped).on)
    check('auto: all data is on',               cleanup_decision('auto', full).on)
    check('auto: per-source "all" is on',       cleanup_decision('auto', per_src).on)
    check('auto: unset crop fields is on',      cleanup_decision('auto', bare).on)
    check('auto: cropped T and Z is off',       !cleanup_decision('auto', cropped).on)
    check('auto: names the cropped fields',     cleanup_decision('auto', cropped).reason.contains('time_indices, Z_slice'))
    check('auto: channel subset is off',        !cleanup_decision('auto', channels).on)
    check('auto: no concatenate config is off', !cleanup_decision('auto', null).on)
    check('true without concatenate refused',   decision_refused('true', null))
    check('an unknown value is refused',        decision_refused('yes', full))

    // The guard: inside the output directory only, and never the directory itself.
    check('accepts a step directory',      cleanup_targets(["${root}/0-flatfield"], root) == ["${root}/0-flatfield"])
    check('normalises the path',           cleanup_targets(["${root}/x/../0-flatfield/"], root) == ["${root}/0-flatfield"])
    check('refuses the output directory',  refused([root], root))
    check('refuses "<output>/."',          refused(["${root}/."], root))
    check('refuses a parent escape',       refused(["${root}/../elsewhere"], root))
    check('refuses a sibling prefix',      refused(["${root}_rerun/0-flatfield"], root))
    check('refuses an empty path',         refused([''], root))
    check('refuses with no output dir',    refused(["${root}/0-flatfield"], null))

    // End to end on a stand-in run: two intermediates and the assembled store's
    // resume markers go, the assembled store itself stays, and a target that was
    // never created is reported rather than failing the task.
    touch("${root}/0-flatfield/ds.zarr/zarr.json")
    touch("${root}/0-flatfield/slurm_output/README.md")
    touch("${root}/1-deskew/.iohub-progress/ds.zarr/A/1/0/t0-4_c0-0_abc.done")
    touch("${root}/4-assemble/ds.zarr/zarr.json")
    touch("${root}/4-assemble/.iohub-progress/ds.zarr/A/1/0/t0-4_c0-0_abc.done")

    def targets = cleanup_targets(["${root}/0-flatfield", "${root}/1-deskew",
                                   "${root}/4-assemble/.iohub-progress", "${root}/9-never-made"], root)
    cleanup = cleanup_intermediates_wf(targets, channel.of('assemble', 'qc'))
    cleanup.done.subscribe { _token ->
        check('deletes a whole step directory',         !new File("${root}/0-flatfield").exists())
        check('deletes a hidden-file-only directory',   !new File("${root}/1-deskew").exists())
        check('deletes the final store\'s markers',     !new File("${root}/4-assemble/.iohub-progress").exists())
        check('keeps the final store',                  new File("${root}/4-assemble/ds.zarr/zarr.json").exists())
    }
}
