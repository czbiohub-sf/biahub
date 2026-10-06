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
    if (params.estimate_config && !params.estimate_positions) {
        error "Provide --estimate_positions: e.g. the beads well ('C/1/000000') for a " +
            "registration shared by every position, or '*/*/*' to stabilize each position"
    }

    def out = params.output
    def reference = params.reference ?: ''
    def apply = params.apply || params.transforms
    def transforms = params.transforms ?: "${out}/transforms.yml"
    def start = channel.value('start')

    if (params.estimate_config) {
        estimate_init = estimate_transform_init_wf(
            params.moving, reference, params.estimate_positions, params.estimate_config,
            transforms, start
        )
        estimated = estimate_transform_run_wf(
            estimate_init.plan, params.moving, reference, params.estimate_positions,
            params.estimate_config, transforms, start
        )
        transforms_ready = estimated.done
    } else {
        transforms_ready = start
    }

    if (apply) {
        def store_name = new File(params.moving.toString()).name.replaceAll(/(\.ome)?\.zarr$/, '')
        def output_zarr = params.apply_output ?: "${out}/${store_name}.zarr"
        def extra = []
        if (params.crop_to_overlap) extra << '--crop-to-overlap'
        if (params.channels) {
            params.channels.toString().split(',').each { c -> extra << "--channels '${c.trim()}'" }
        }
        def extra_args = extra.join(' ')
        def matcher = java.nio.file.FileSystems.getDefault()
            .getPathMatcher("glob:${params.apply_positions}")

        apply_positions = collect_positions(params.moving)
            .map { keys -> keys.findAll { key -> matcher.matches(java.nio.file.Paths.get(key)) } }
        apply_init = apply_transform_init_wf(
            params.moving, reference, params.apply_positions, transforms, output_zarr,
            extra_args, transforms_ready
        )
        apply_transform_run_wf(
            apply_positions, params.moving, reference, params.apply_positions, transforms,
            output_zarr, extra_args, apply_init.resources, apply_init.done
        )
    }
}
