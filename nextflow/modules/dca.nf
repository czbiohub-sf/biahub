// DCA finalize: write the DCA v0.2 metadata onto every FOV of a finished plate,
// then validate the plate against the spec. The last step of a run, after QC,
// because finalize reads the plate's QC stage-5 statistics.
//
// `biahub dca-finalize` is metadata-only (it edits each position's zarr.json
// attributes and reads no pixel), so one task per plate covers every position.
// It is two-pass: if any FOV is missing a required value it writes nothing,
// lists every gap and exits non-zero, which terminates the run. Fill the plate's
// `<stem>.dca-confirmed.toml` and `-resume`.
//
// NEVER CACHED. What finalize reads (the base TOML, the confirmed TOML, the
// sample sheet, the QC tables) is not in the task hash, so a cached task could
// keep metadata built from files edited since. The step is metadata-only and
// last, so `-resume` always reruns it.
//
// THE PLATE IS A `val`, NEVER A STAGED `path`. finalize finds the confirmed TOML
// and the sample sheet NEXT TO the plate by default, so a plate staged as a
// symlink under the work dir would send it looking there. The same holds for
// the config: the CLI reads the absolute `val`, so the YAML's relative
// `base_config` resolves against its own directory. The staged copy follows the
// common.nf convention (#397) and is never read.
//
// `dca-validate` exiting non-zero fails the task: Nextflow runs the script under
// `bash -e`, and a redirect, unlike a pipe, keeps the command's exit status. The
// JSON report lands next to the plate as `<stem>.dca-validate.json` either way,
// so a failing plate keeps the findings that failed it. The stem drops the last
// suffix only, as finalize's own defaults do (`plate.zarr` -> `plate`). The task
// deletes the previous run's report first, so a failed finalize cannot leave a
// passing report beside the plate.

include { slurm_logs; slurm_log_dir; retry_memory; retry_time } from './common'


process dca_finalize_store {
    tag "${zarr}"
    label 'cpu'
    clusterOptions { slurm_logs('dca') }
    memory { retry_memory(8, task) }
    time { retry_time(30, task) }
    cache false

    input:
    tuple val(zarr), val(config), path(config_file)  // config_file: staged per common.nf (#397), never read

    output:
    val zarr

    script:
    def plate = file(zarr)
    def report = "${plate.parent}/${plate.baseName}.dca-validate.json"
    """
    rm -f "${report}"
    biahub dca-finalize -i "${zarr}"/*/*/* -c "${config}"
    dca-validate --spec-version 0.2 --show-warnings --output json "${zarr}" > "${report}"
    """
}


// Finalize and validate each plate once the gate opens.
//
// take:
//   stores   Groovy list of plate.zarr paths to finalize
//   config   path to the dca_finalize settings YAML
//   gate     gating channel: the plates' QC statistics are final and no other
//            step still reads the plates; may mix several producers
// emit:
//   done     fires once every plate is finalized and valid
workflow dca_finalize_wf {
    take:
    stores
    config
    gate

    main:
    // Created at launch on purpose, as qc_report_spec() does (qc.nf): no init
    // task exists for this step, and SLURM cannot open a job's --output under a
    // directory that does not exist.
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
