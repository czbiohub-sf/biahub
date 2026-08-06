r"""Generic Monarch host-worker loop for multi-node runs.

One instance runs per node (the tile-stitch wrapper launches it via
``srun --ntasks-per-node=1`` when the allocation spans >1 node). It derives its
TCP port + the shared ready-dir from the SLURM allocation — the same
``_slurm_topology`` the driver uses — so no flags are passed: it binds, signals
readiness, and serves as a Monarch host the driver attaches to.

The listen address must be a resolvable hostname (wildcard ``tcp://*:PORT`` is
rejected by Monarch) so the driver's connect string matches.
"""

import logging
import os
import socket
import sys


def main() -> None:
    import monarch.actor as ma

    from biahub.tile_stitch.monarch.backend import _slurm_topology

    logging.basicConfig(level=logging.INFO, format="%(asctime)s WORKER %(message)s")
    log = logging.getLogger("worker_loop")

    _hosts, port, ready_dir = _slurm_topology()
    if not port:
        log.error("no multi-node SLURM allocation detected; nothing to serve")
        sys.exit(1)

    ma.enable_transport("tcp")
    host = socket.gethostname()
    address = f"tcp://{host}:{port}"
    # Signal readiness on the shared FS just before the blocking bind — a tiny
    # race remains (file exists ~ms before the socket listens), so the driver
    # adds a small margin after all files appear.
    if ready_dir:
        os.makedirs(ready_dir, exist_ok=True)
        with open(os.path.join(ready_dir, f"{host}.ready"), "w") as f:
            f.write(address)
    log.info("starting host worker loop at %s", address)
    ma.run_worker_loop_forever(address=address, ca="trust_all_connections")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
