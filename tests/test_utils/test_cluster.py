import pytest

from biahub.utils.cluster import apply_sbatch_overrides, local_executor_parameters

COMPUTED = {"slurm_mem": "28G", "slurm_cpus_per_task": 4, "slurm_time": 30}


@pytest.mark.parametrize("key", ["slurm_mem_per_cpu", "slurm_mem_per_gpu"])
def test_a_memory_override_replaces_the_computed_total(key):
    """SLURM rejects jobs that set more than one of --mem, --mem-per-cpu, --mem-per-gpu."""
    merged = apply_sbatch_overrides(COMPUTED, {key: "16G"})

    assert merged[key] == "16G"
    assert "slurm_mem" not in merged
    assert merged["slurm_cpus_per_task"] == 4


def test_a_total_memory_override_replaces_a_computed_per_cpu_value():
    merged = apply_sbatch_overrides({"slurm_mem_per_cpu": "7G"}, {"slurm_mem": "64G"})

    assert merged == {"slurm_mem": "64G"}


def test_other_overrides_keep_the_computed_memory():
    merged = apply_sbatch_overrides(COMPUTED, {"slurm_time": "3:00:00"})

    assert merged == {**COMPUTED, "slurm_time": "3:00:00"}
    assert COMPUTED["slurm_time"] == 30  # the input is not mutated


def test_local_executor_parameters_give_a_gpu_and_the_slurm_time_budget():
    assert local_executor_parameters(90) == {"timeout_min": 90, "gpus_per_node": 1}


def test_local_executor_parameters_without_a_gpu_only_set_the_timeout():
    assert local_executor_parameters(90, gpu=False) == {"timeout_min": 90}


def test_sbatch_overrides_win_over_local_executor_parameters():
    slurm_args = apply_sbatch_overrides({"slurm_time": 90}, {"timeout_min": 5})
    assert {**local_executor_parameters(90), **slurm_args}["timeout_min"] == 5
