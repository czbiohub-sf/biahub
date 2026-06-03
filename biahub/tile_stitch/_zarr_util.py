"""Shared helpers for the Monarch tile-stitch driver."""


def parse_timepoints(spec: str) -> list[int]:
    """Parse '0-9' (inclusive range), '0,3,7' (list), or '5' (single)."""
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",")]
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]
