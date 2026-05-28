#!/usr/bin/env nextflow

/*
 * Toy pipeline demonstrating submitit DebugExecutor failure modes under Nextflow.
 *
 * Usage:
 *   Set --python_bin to the python interpreter in your venv that has submitit
 *   installed. Defaults to "python" (whatever is on PATH).
 *
 *   # Using a specific venv:
 *   nextflow run main.nf --python_bin /path/to/venv/bin/python --mode success
 *
 *   # Concern 1: pdb on failure (with submitit) — pollutes logs; hangs if stdin is a TTY
 *   nextflow run main.nf --python_bin /path/to/venv/bin/python --mode fail
 *
 *   # Concern 1: clean failure (bypass submitit) — exits cleanly with traceback
 *   nextflow run main.nf --python_bin /path/to/venv/bin/python --mode fail --bypass_submitit true
 *
 *   # Concern 2: shadow logs (with submitit) — check submitit_logs/ in work dirs after
 *   nextflow run main.nf --python_bin /path/to/venv/bin/python --mode success
 *
 *   # Concern 2: no shadow logs (bypass submitit)
 *   nextflow run main.nf --python_bin /path/to/venv/bin/python --mode success --bypass_submitit true
 *
 *   # Concern 3: env var pollution (with submitit)
 *   NXF_DEMO_VAR=hello nextflow run main.nf --python_bin /path/to/venv/bin/python --mode success
 */

params.python_bin = 'python'
params.mode = 'success'
params.bypass_submitit = false

process TOY_PROCESS {
    /*
     * Time limit: 30 seconds. With submitit debug + failure mode on a TTY stdin,
     * the process will hang in pdb.post_mortem() until this limit kills it.
     * Under Nextflow's pipe stdin, pdb exits on EOF but pollutes the logs.
     */
    time '30s'
    errorStrategy 'terminate'

    input:
    val position_id

    output:
    stdout

    script:
    def bypass_flag = params.bypass_submitit ? '--bypass-submitit' : ''
    """
    # Set a Nextflow-injected env var to demonstrate Concern 3
    export NXF_TASK_WORKDIR="\${PWD}"
    export CUDA_VISIBLE_DEVICES="0"

    ${params.python_bin} ${projectDir}/toy_process.py --mode ${params.mode} ${bypass_flag}
    """
}


workflow {
    // Fan out over 3 toy "positions"
    positions = Channel.of('pos_0', 'pos_1', 'pos_2')
    TOY_PROCESS(positions)
    TOY_PROCESS.out.view { "STDOUT from task: ${it.trim()}" }
}
