"""Vendored subset of ahillsley/stitching@jen.

Upstream lives at https://github.com/ahillsley/stitching. Only the files
actually imported by biahub are vendored here (connect, graph, tile). The
``dexp`` dependency in upstream ``tile.py`` is replaced by a small shim in
``_dexp_shim`` using numpy/scipy so this package adds no runtime deps.
"""

from .tile import optimal_positions, pairwise_shifts

__all__ = ["optimal_positions", "pairwise_shifts"]
