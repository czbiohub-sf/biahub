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
// targets are gone, the final store next to them is not, a target reached
// through a symlink leading outside the run is refused, a path full of shell
// syntax is deleted rather than run, and the record file says so. Runs locally in seconds:
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
    // slurm_output/ and resume markers go, the assembled store itself stays, a
    // target that was never created is recorded rather than failing the task,
    // and the record is appended to, not overwritten.
    touch("${root}/0-flatfield/ds.zarr/zarr.json")
    touch("${root}/0-flatfield/slurm_output/README.md")
    touch("${root}/1-deskew/.iohub-progress/ds.zarr/A/1/0/t0-4_c0-0_abc.done")
    touch("${root}/4-assemble/ds.zarr/zarr.json")
    touch("${root}/4-assemble/slurm_output/DEBUG_1_0_log.out")
    touch("${root}/4-assemble/.iohub-progress/ds.zarr/A/1/0/t0-4_c0-0_abc.done")
    // A symlinked directory that leads outside the run passes the lexical
    // check at launch; the task must refuse it and leave the outside data be.
    touch("${root}_outside/slurm_output/keep.txt")
    def link = java.nio.file.Paths.get("${root}/5-link")
    java.nio.file.Files.deleteIfExists(link)  // refused, so a previous run left it
    java.nio.file.Files.createSymbolicLink(link, java.nio.file.Paths.get("${root}_outside"))
    // A name full of shell syntax must be deleted as written, never run.
    def odd_name = '6-odd "$(echo INJECTED)" `echo x` \'q\' $HOME'
    touch("${root}/${odd_name}/zarr.json")
    def record = new File("${root}/nextflow/intermediates_cleaned.txt")
    record.parentFile.mkdirs()
    record.text = "=== an earlier cleanup ===\n"

    def targets = cleanup_targets(["${root}/0-flatfield", "${root}/1-deskew",
                                   "${root}/4-assemble/slurm_output", "${root}/4-assemble/.iohub-progress",
                                   "${root}/5-link/slurm_output", "${root}/${odd_name}",
                                   "${root}/9-never-made"], root)
    cleanup = cleanup_intermediates_wf(targets, root, record.path, 'on (auto: concatenate.yml takes all the data)',
                                       channel.of('assemble', 'qc'))
    cleanup.done.subscribe { _token ->
        check('deletes a whole step directory',         !new File("${root}/0-flatfield").exists())
        check('deletes a hidden-file-only directory',   !new File("${root}/1-deskew").exists())
        check('deletes the final store\'s slurm_output', !new File("${root}/4-assemble/slurm_output").exists())
        check('deletes the final store\'s markers',     !new File("${root}/4-assemble/.iohub-progress").exists())
        check('keeps the final store',                  new File("${root}/4-assemble/ds.zarr/zarr.json").exists())

        def lines = record.readLines()
        check('record keeps the earlier section',       lines[0] == '=== an earlier cleanup ===')
        check('record opens a timestamped section',     lines.count { line -> line.startsWith('=== 20') } == 1)
        check('record states the decision',             lines.contains('decision  on (auto: concatenate.yml takes all the data)'))
        check('record lists each removed target',
              ['0-flatfield', '1-deskew', '4-assemble/slurm_output', '4-assemble/.iohub-progress']
                  .every { t -> lines.contains("removed   ${root}/${t}".toString()) })
        check('record marks a missing target absent',   lines.contains("absent    ${root}/9-never-made".toString()))
        check('keeps data behind a symlink escape',     new File("${root}_outside/slurm_output/keep.txt").exists())
        check('record marks a symlink escape refused',
              lines.any { line -> line.startsWith("refused   ${root}/5-link/slurm_output (".toString()) })
        check('deletes a name with shell syntax',       !new File("${root}/${odd_name}").exists())
        check('records that name verbatim',             lines.contains("removed   ${root}/${odd_name}".toString()))
    }
}
