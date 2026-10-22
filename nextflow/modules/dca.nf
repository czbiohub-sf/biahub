// DCA finalize: write the DCA v0.2 metadata onto every FOV of a finished plate,
// then validate the plate against the spec. The last step of a run, after QC,
// because finalize reads the plate's QC stage-5 statistics.
//
// `biahub dca-finalize` is metadata-only (it edits each position's zarr.json
// attributes and reads no pixel), so one task per plate covers every position.
// It is two-pass: if any FOV is missing a required value it writes nothing,
// lists every gap and exits non-zero, which terminates the run. Fill the plate's
// `<plate>.dca-confirmed.toml` and `-resume`; a failed task is never cached.
//
// THE PLATE IS A `val`, NEVER A STAGED `path`. finalize finds the confirmed TOML
// and the sample sheet NEXT TO the plate by default, so a plate staged as a
// symlink under the work dir would send it looking there. The same holds for
// the config: the CLI reads the absolute `val`, so the YAML's relative
// `base_config` resolves against its own directory, and the staged copy only
// keys the task hash (see common.nf, #397).
//
// `dca-validate` exiting non-zero fails the task: Nextflow runs the script under
// `bash -e`, and a redirect, unlike a pipe, keeps the command's exit status. The
// JSON report lands next to the plate either way, so a failing plate keeps the
// findings that failed it.

include { slurm_logs; slurm_log_dir; retry_memory; retry_time } from './common'


process dca_finalize_store {
    tag "${zarr}"
    label 'cpu'
    clusterOptions { slurm_logs('dca') }
    memory { retry_memory(8, task) }
    time { retry_time(30, task) }

    input:
    tuple val(zarr), val(config), path(config_file)  // config_file: staged only for the task hash, see common.nf (#397)

    output:
    val zarr

    script:
    """
    biahub dca-finalize -i "${zarr}"/*/*/* -c "${config}"
    dca-validate --spec-version 0.2 --show-warnings --output json "${zarr}" > "${zarr}.dca-validate.json"
    """
}


// Finalize and validate each plate once the gate opens.
//
// take:
//   stores   Groovy list of plate.zarr paths to finalize
//   config   path to the dca_finalize settings YAML
//   gate     gating channel: the plates' QC statistics are final
// emit:
//   done     fires once every plate is finalized and valid
workflow dca_finalize_wf {
    take:
    stores
    config
    gate

    main:
    // No init task creates this step's log directory, and SLURM cannot open a
    // job's --output under a directory that does not exist.
    file(slurm_log_dir('dca')).mkdirs()

    def config_path = file(config).toString()
    // Mapped to a token before combining, as in assembly.nf: a gate's payload
    // shape must not reach the tuple.
    finalize_inputs = channel.fromList(stores.collect { zarr -> file(zarr).toString() })
        .combine(gate.collect().map { 'done' })
        .map { zarr, _gate -> tuple(zarr, config_path, file(config_path)) }

    finalized = dca_finalize_store(finalize_inputs) | collect

    emit:
    done = finalized
}
