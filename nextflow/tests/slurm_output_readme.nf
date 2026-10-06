#!/usr/bin/env nextflow

// Asserts that slurm_output_readme() writes the pointer README a step's init
// process leaves in `<step_dir>/slurm_output/` (biahub#276).
//
// WHY THIS EXISTS. Tasks run the CLI with `--cluster debug`, so the per-step
// slurm_output/ only holds submitit's DebugJob placeholders and the real logs
// land in nextflow/slurm_output/<step>/. The README is emitted as a heredoc
// inside an indented process script, so this runs it the same way the init
// processes do. Runs locally in seconds:
//     nextflow run nextflow/tests/slurm_output_readme.nf
// It exits non-zero if the README is missing or points at the wrong place.

nextflow.enable.dsl = 2

params.output = "${workflow.workDir}/slurm_output_readme_test"

include { slurm_output_readme } from '../modules/common'

process init_step {
    input:
    val output_zarr

    output:
    stdout

    script:
    """
    ${slurm_output_readme('deskew', output_zarr)}
    echo "RESOURCES:{}"
    """
}

workflow {
    def output_zarr = "${params.output}/1-deskew/dataset.zarr"
    init_step(output_zarr).view { stdout_text ->
        def readme = new File("${params.output}/1-deskew/slurm_output/README.md")
        def expected_dir = "${params.output}/nextflow/slurm_output/deskew/"
        def failures = []
        if (!readme.exists()) {
            failures << "README not written at ${readme}"
        } else {
            def text = readme.text
            if (!text.contains(expected_dir)) failures << "README does not point at ${expected_dir}"
            if (!text.contains('--cluster debug')) failures << "README does not explain --cluster debug"
        }
        if (!stdout_text.trim().readLines().any { line -> line.startsWith('RESOURCES:') }) {
            failures << "snippet swallowed the RESOURCES line"
        }
        if (failures) error "FAIL:\n  " + failures.join("\n  ")
        return "PASS: ${readme}"
    }
}
