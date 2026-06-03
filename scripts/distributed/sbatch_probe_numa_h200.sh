#!/usr/bin/env bash
# Probe H200 node NUMA topology for Monarch ``proc_bind`` planning.
#
# Captures: numactl --hardware, lscpu, nvidia-smi topology (CPU affinity
# per GPU), NIC mapping. Writes to log; no state changes.
#
# Submit:
#   sbatch scripts/distributed/sbatch_probe_numa_h200.sh
#
#SBATCH --job-name=probe-numa-h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=%x_%j.log

set +e   # diagnostic: keep going on missing tools

banner() { echo; echo "===== $* ====="; }

banner "hostname / slurm"
hostname
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"

banner "numactl --hardware"
numactl --hardware 2>&1 | head -40

banner "lscpu (NUMA section)"
lscpu | grep -E "Socket|NUMA|CPU\\(s\\)|Model name"

banner "nvidia-smi topo -m (GPU-CPU affinity + IB)"
nvidia-smi topo -m 2>&1 | head -30

banner "nvidia-smi topo -c (CPU affinity per GPU)"
for i in 0 1; do
    echo "GPU $i CPU affinity:"
    nvidia-smi topo -c $i 2>&1 | head -5
done

banner "Mellanox IB → NUMA mapping"
for hca in /sys/class/infiniband/*/device/numa_node; do
    nic=$(basename $(dirname $(dirname "$hca")))
    node=$(cat "$hca" 2>/dev/null)
    echo "$nic -> NUMA $node"
done

banner "PCI topology for GPUs + IB HCAs"
lspci -tv 2>&1 | grep -A 1 -B 1 -E "NVIDIA|Mellanox" | head -30

banner "done"
date
