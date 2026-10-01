#!/usr/bin/env nextflow

// Asserts what --cleanup_intermediates deletes and what it refuses to (biahub#292).
//
// WHY THIS EXISTS. The cleanup is an `rm -rf` on paths the pipeline computes,
// so the guard in cleanup_targets() is the only thing between a wiring mistake
// and deleting the run's deliverables, or data outside it. The first block
// checks the guard directly; the second runs cleanup_intermediates_wf on a
// stand-in run directory and checks that the targets are gone and the final
// store next to them is not. Runs locally in seconds:
//     nextflow run nextflow/tests/cleanup_intermediates.nf
// It exits non-zero if either regresses.

nextflow.enable.dsl = 2

params.output = "${workflow.workDir}/cleanup_intermediates_test/run"

include { cleanup_targets; cleanup_intermediates_wf } from '../modules/cleanup'

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
    catch (Exception e) {
        return true
    }
}

def touch(path) {
    def file = new File(path)
    file.parentFile.mkdirs()
    file.text = 'x'
}

workflow {
    def root = params.output

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
    cleanup.done.subscribe { token ->
        check('deletes a whole step directory',         !new File("${root}/0-flatfield").exists())
        check('deletes a hidden-file-only directory',   !new File("${root}/1-deskew").exists())
        check('deletes the final store\'s markers',     !new File("${root}/4-assemble/.iohub-progress").exists())
        check('keeps the final store',                  new File("${root}/4-assemble/ds.zarr/zarr.json").exists())
    }
}
