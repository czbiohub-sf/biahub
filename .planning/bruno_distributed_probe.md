# Bruno distributed-recon probe — 2026-05-22

Source: `sbatch scripts/distributed/sbatch_probe_gpu_node.sh` (job 33230482, ran on `gpu-h-4`, H200).

## Hardware
- 8× ConnectX-7 + 1× ConnectX-6 InfiniBand HCAs per node (mlx5_0/1 Active @ 400 Gb/s, IB link)
- H100 partition: `gpu-f-[1-6]` (8× H100/node)
- H200 partition: `gpu-h-[1-8]` (8× H200/node, 144 GB VRAM each)

## OS / RDMA
- MLNX-OFED 24.10 (`libibverbs-2410mlnx54`, `rdma-core-2410mlnx54`)
- UCX 1.18.0 system, built with `--with-cuda --with-gdrcopy --with-verbs --with-mlx5` → GPUDirect RDMA ready

## MPI / NCCL stack
- `module load hpcx/2.19` → OpenMPI 4.1.7a1 + UCX + UCC + HCOLL + SHARP + clusterkit
- Alternative CUDA-aware: `openmpi/5.0.7-cuda12.8`
- Recommended bootstrap for distributed FFT: `module load cuda/13.1.0_590.44.01 hpcx/2.19`

## NVSHMEM
- **Not installed system-wide.** No `nvshmem-info`, `NVSHMEM_HOME` unset.
- Plan: rely on the `nvidia-nvshmem-cu13` PyPI wheel bundled with `nvmath-python[cu13]`. To be verified on first `uv sync --extra tilestitch-distributed`.

## Python env state (biahub feature-tile-stitch venv)
- `torch 2.11.0+cu130` ✓
- `tilestitch-gpu` extras (cupy, rmm, dask_cuda, ucxx) **not yet synced** — `uv run --no-sync --extra X` does NOT install. Run `uv sync --extra tilestitch-gpu --extra tilestitch-distributed` explicitly before first sbatch.

## Implications for experiment plan
- **E2** (single-node 2-GPU distributed FFT): use hpcx mpirun, NVSHMEM via PyPI wheel.
- **E3** (2-node × 4-GPU): GDRCopy + ConnectX-7 + CUDA UCX = strong path. Use `mpirun --mca pml ucx --mca osc ucx`.
- **E0** (UCXX dask protocol): UCX wheel needs to be installed first.
