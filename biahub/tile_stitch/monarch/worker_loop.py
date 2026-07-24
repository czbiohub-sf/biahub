r"""Monarch host-worker entry point for multi-node tile-stitch runs.

Each node derives the same address and readiness directory as the driver, then
serves a host the driver can attach to.
"""

import logging
import os
import socket
import sys


def main() -> None:
    """Start the Monarch host worker for the active SLURM node.

    Raises
    ------
    SystemExit
        If no multi-node SLURM allocation is active.
    """
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
    # Publish the address before entering Monarch's blocking worker loop.
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
