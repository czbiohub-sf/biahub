#!/usr/bin/env nextflow
//
// Standalone registration / stabilization: estimate a transform series with
// `biahub estimate-transform`, then (optionally) apply it with `biahub
// apply-transform` -- without the mantis-v2 pipeline. The CLI owns the logic
// (see modules/registration.nf); Nextflow owns the fan-out, retries, resources
// and -resume.
//
//   source <BIAHUB>/.venv/bin/activate
//   cd <BIAHUB>/nextflow
//
//   # registration: estimate on the beads well, apply to every position
//   nextflow run registration.nf -profile slurm \
//       --moving <deskewed.zarr> --reference <reconstructed.zarr> \
//       --estimate_config estimate-transform-beads.yml --estimate_positions 'C/1/000000' \
//       --apply --output <dir> -resume
//
//   # stabilization: one transform list per position, applied to the same positions
//   nextflow run registration.nf -profile slurm \
//       --moving <data.zarr> --estimate_config estimate-transform-focus-finding.yml \
//       --estimate_positions '*/*/*' --apply --output <dir> -resume
//
//   # apply an existing transforms file (e.g. after substitute-transforms)
//   nextflow run registration.nf -profile slurm \
//       --moving <deskewed.zarr> --reference <reconstructed.zarr> \
//       --transforms final.yml --output <dir> -resume
//
// Writes <output>/transforms.yml (and its run folder <output>/transforms/) and,
// when applying, <output>/<moving store name>.zarr; per-task logs go to
// <output>/nextflow/slurm_output/{estimate_transform,apply_transform}/.
// Manual registration is interactive and cannot run here: run it with the CLI and
// apply its file with --transforms.

nextflow.enable.dsl = 2

include { collect_positions; check_environment } from './modules/common'
include {
    estimate_transform_init_wf; estimate_transform_run_wf;
    apply_transform_init_wf; apply_transform_run_wf
} from './modules/registration'


// A path as given, made absolute against the launch directory (null stays null).
def absolute_path(path) {
    return path ? file(path.toString()).toAbsolutePath().toString() : path
}


workflow {
    check_environment(['biahub'])

    if (!params.moving) {
        error "Provide --moving (the plate holding the moving channel)"
    }
    if (!params.output) {
        error "Provide --output (the run directory)"
    }
    if (!params.estimate_config && !params.transforms) {
        error "Provide --estimate_config (estimate, then --apply to apply) or --transforms " +
            "(apply an existing transforms file)"
    }
    if (params.estimate_config && params.transforms) {
        error "--estimate_config and --transforms are alternatives: estimate a new file, " +
            "or apply an existing one"
    }
    // Tasks run in their own work directory, so a relative path would point there. The
    // output must be absolute (common.nf's log paths and the work dir read it as given);
    // the inputs are resolved here, against the launch directory.
    if (!new File(params.output.toString()).isAbsolute()) {
        error "--output must be an absolute path (got '${params.output}'): e.g. \$(realpath ${params.output})"
    }
    ['estimate_config', 'transforms'].each { name ->
        if (params[name] && !file(params[name].toString()).exists()) {
            error "--${name} not found: ${params[name]}"
        }
    }
    if (params.estimate_config && !params.estimate_positions) {
        error "Provide --estimate_positions: e.g. the beads well ('C/1/000000') for a " +
            "registration shared by every position, or '*/*/*' to stabilize each position"
    }

    def out = params.output
    def moving = absolute_path(params.moving)
    def reference = absolute_path(params.reference) ?: ''
    def estimate_config = absolute_path(params.estimate_config)
    def apply = params.apply || params.transforms
    def transforms = absolute_path(params.transforms) ?: "${out}/transforms.yml"
    def start = channel.value('start')

    // An edited config reruns the estimate (staged file, main's #397); the transforms file
    // is keyed on its content (see the module header).
    if (params.estimate_config) {
        estimate_init = estimate_transform_init_wf(
            moving, reference, params.estimate_positions, estimate_config,
            file(estimate_config), transforms, start
        )
        estimated = estimate_transform_run_wf(
            estimate_init.plan, moving, reference, params.estimate_positions,
            estimate_config, file(estimate_config), transforms, start
        )
        transforms_hash = estimated.done
    } else {
        transforms_hash = channel.value(file(transforms).text.md5())
    }

    if (apply) {
        def store_name = new File(moving).name.replaceAll(/(\.ome)?\.zarr$/, '')
        def output_zarr = absolute_path(params.apply_output) ?: "${out}/${store_name}.zarr"
        def extra = []
        if (params.crop_to_overlap) extra << '--crop-to-overlap'
        if (params.channels) {
            params.channels.toString().split(',').each { c -> extra << "--channels '${c.trim()}'" }
        }
        def extra_args = extra.join(' ')
        def matcher = java.nio.file.FileSystems.getDefault()
            .getPathMatcher("glob:${params.apply_positions}")

        apply_positions = collect_positions(moving)
            .map { keys -> keys.findAll { key -> matcher.matches(java.nio.file.Paths.get(key)) } }
        apply_init = apply_transform_init_wf(
            moving, reference, params.apply_positions, transforms, transforms_hash,
            output_zarr, extra_args
        )
        apply_transform_run_wf(
            apply_positions, moving, reference, params.apply_positions, transforms,
            transforms_hash, output_zarr, extra_args, apply_init.resources, apply_init.done
        )
    }
}
