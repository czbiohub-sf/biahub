#!/usr/bin/env nextflow
//
// Registration benchmark: every pipeline (an estimate config, e.g. a stack of methods) on
// every case (a dataset, its estimate and held-out positions), each run evaluated in µm
// by `biahub evaluate-transform` -- its transform and each block of its stack, against the
// unregistered pair. The CLI owns the logic (modules/registration.nf); Nextflow owns the
// fan-out, retries and -resume.
//
//   nextflow run registration_benchmark_wf.nf -profile slurm \
//       --benchmark benchmark.yml --output $(realpath runs/benchmark) -resume
//
// benchmark.yml:
//   cases:
//     - name: dragonfly_0418
//       moving: /path/0-convert/Cells_2.zarr
//       reference: /path/2-VS/VS_Cells2.zarr      # omit to stabilize
//       moving_channel: 2-Prime_DAPI
//       reference_channel: nuclei_prediction
//       estimate_positions: 0/12/001001             # one, or a list
//       evaluate_positions: [0/9/001001, 0/19/001001]
//       landmarks: nuclei                           # optional: nuclei, beads
//       tile_px: 256                                # optional
//   pipelines:
//     - name: tiles_nuclei
//       config: tiles_nuclei.yml                    # its channels come from each case
//
// Writes <output>/<case>/<pipeline>/{estimate.yml, transforms.yml, transforms/,
// evaluation.csv, evaluation/} and <output>/benchmark.csv (every table joined).

nextflow.enable.dsl = 2

// Each task's resources (set before the includes, so the module sees them).
params.benchmark_cpus = 4
params.benchmark_mem_gb = 48
params.benchmark_time_minutes = 120

include { check_environment } from './modules/common'
include { benchmark_estimate; benchmark_evaluate } from './modules/registration'

def as_list(value) {
    return value instanceof List ? value : [value]
}

// A path in the benchmark file, made absolute against the file's own directory.
def resolve(path, base) {
    def f = new File(path.toString())
    return (f.isAbsolute() ? f : new File(base, path.toString())).getCanonicalPath()
}

// The pipeline's config with the case's channels: written once per case x pipeline.
def case_config(pipeline_config, the_case, target) {
    def yaml = new org.yaml.snakeyaml.Yaml()
    def config = yaml.load(new File(pipeline_config).text) as Map
    config.moving = (config.moving ?: [:]) + [channel: the_case.moving_channel]
    if (the_case.reference) {
        config.reference = (config.reference ?: [frame: 'cross']) + [channel: the_case.reference_channel]
    }
    def out = new File(target)
    out.parentFile.mkdirs()
    def options = new org.yaml.snakeyaml.DumperOptions()
    options.setDefaultFlowStyle(org.yaml.snakeyaml.DumperOptions.FlowStyle.BLOCK)
    out.text = new org.yaml.snakeyaml.Yaml(options).dump(config)
    return out.getCanonicalPath()
}


workflow {
    check_environment(['biahub'])
    if (!params.benchmark) {
        error "Provide --benchmark (the cases and pipelines YAML)"
    }
    if (!params.output || !new File(params.output.toString()).isAbsolute()) {
        error "Provide --output as an absolute path: e.g. \$(realpath ${params.output ?: 'runs'})"
    }
    def spec_file = file(params.benchmark)
    def base = spec_file.parent.toString()
    def spec = new org.yaml.snakeyaml.Yaml().load(spec_file.text) as Map
    ['cases', 'pipelines'].each { key ->
        if (!spec[key]) error "${params.benchmark}: needs a '${key}' list"
    }
    def required = ['name', 'moving', 'moving_channel', 'estimate_positions', 'evaluate_positions']
    spec.cases.each { c ->
        def missing = required.findAll { k -> !c[k] }
        if (c.reference && !c.reference_channel) missing << 'reference_channel'
        if (missing) error "case ${c.name ?: '?'}: missing ${missing}"
    }

    def runs = []
    spec.cases.each { c ->
        spec.pipelines.each { p ->
            def dir = "${params.output}/${c.name}/${p.name}"
            def config = case_config(resolve(p.config, base), c, "${dir}/estimate.yml")
            runs << [
                c.name, p.name, resolve(c.moving, base),
                c.reference ? resolve(c.reference, base) : '',
                as_list(c.estimate_positions), config, file(config), "${dir}/transforms.yml",
            ]
        }
    }
    def evaluation = spec.cases.collectEntries { c ->
        def options = []
        if (c.landmarks) options << "--landmarks ${c.landmarks}"
        if (c.tile_px) options << "--tile-px ${c.tile_px}"
        [(c.name): [as_list(c.evaluate_positions), options.join(' ')]]
    }

    estimated = benchmark_estimate(channel.fromList(runs))
    evaluated = benchmark_evaluate(
        estimated.map { case_name, pipeline, transforms ->
            def r = runs.find { it[0] == case_name && it[1] == pipeline }
            def (positions, options) = evaluation[case_name]
            [case_name, pipeline, transforms, r[2], r[3], positions, options]
        }
    )
    evaluated.collectFile(
        name: 'benchmark.csv', storeDir: params.output, keepHeader: true, skip: 1
    )
}
