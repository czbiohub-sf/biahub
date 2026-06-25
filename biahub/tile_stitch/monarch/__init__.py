"""Monarch actor-based tile-stitch engine.

PyTorch Monarch single-controller / actor-mesh distributed framework — the
only distributed backend for tile-stitch (multi-host + RDMABuffer).

Modules:

- ``tile_worker`` — one actor per GPU: recon (Stage A), RDMA handoff, CPU blend
  + zarr write (Stage B), per-TP volume swap.
- ``backend`` — ``MonarchBackend``: mesh bring-up, per-TP pipelined drive, swap,
  teardown. Driven by ``biahub.tile_stitch.cli``.
- ``worker_loop`` — per-node ``run_worker_loop_forever`` entry for multi-host.
"""
