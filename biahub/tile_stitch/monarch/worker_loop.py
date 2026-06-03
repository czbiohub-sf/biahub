r"""Generic Monarch host-worker loop for multi-node runs.

One instance runs per node (launched via ``srun --ntasks-per-node=1``).
It binds a TCP server on the node's own hostname; the driver attaches to
all of them via ``attach_to_workers`` to form a HostMesh.

Wildcard ``tcp://*:PORT`` is rejected by Monarch — the listen address
must be a resolvable hostname so the driver's connect string matches.
"""

import logging
import socket
import sys

import click

logger = logging.getLogger("worker_loop")


@click.command()
@click.option("--port", type=int, default=26000, show_default=True)
@click.option(
    "--ready-dir",
    type=str,
    default=None,
    help="Shared dir; worker touches <hostname>.ready just before binding "
    "so the driver can gate its attach on all workers being up.",
)
def main(port: int, ready_dir: str | None) -> None:
    import os

    import monarch.actor as ma

    logging.basicConfig(level=logging.INFO, format="%(asctime)s WORKER %(message)s")
    ma.enable_transport("tcp")
    host = socket.gethostname()
    address = f"tcp://{host}:{port}"
    # Signal readiness on the shared FS. Written just before the blocking
    # bind — a tiny race remains (file exists ~ms before the socket
    # listens), so the driver adds a small margin after all files appear.
    if ready_dir:
        os.makedirs(ready_dir, exist_ok=True)
        with open(os.path.join(ready_dir, f"{host}.ready"), "w") as f:
            f.write(address)
    logger.info("starting host worker loop at %s", address)
    ma.run_worker_loop_forever(address=address, ca="trust_all_connections")


if __name__ == "__main__":
    try:
        main(standalone_mode=False)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
