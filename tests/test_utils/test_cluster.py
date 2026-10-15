import pytest

from biahub.utils.cluster import apply_sbatch_overrides

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
