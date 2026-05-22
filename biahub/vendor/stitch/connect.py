import numpy as np
import yaml

from numpy.typing import NDArray


def parse_positions(fovs: list) -> NDArray:
    """Parse FOV names into integer ``(x, y)`` grid coordinates.

    FOV names are assumed to follow the ``XXXYYY`` zero-padded convention
    (e.g. ``"003001"`` -> ``(3, 1)``). Names may optionally be prefixed by
    ``row/col/`` paths; only the trailing token is parsed.

    Parameters
    ----------
    fovs : list[str]
        FOV names, each formatted as ``XXXYYY`` or ``row/col/XXXYYY``.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(len(fovs), 2)`` with one ``(x, y)`` row
        per input FOV, preserving input order.
    """
    fov_positions = []
    for p in fovs:
        p = p.split("/")[-1]  # handle positions either with row/column info or without
        x_cord = int(p[:3])
        y_cord = int(p[3:])
        fov_positions.append((x_cord, y_cord))

    return np.asarray(fov_positions)


def pos_to_name(pos: tuple) -> str:
    """Convert an ``(x, y)`` grid position to a zero-padded ``XXXYYY`` name.

    Parameters
    ----------
    pos : tuple[int, int]
        Two-element ``(x, y)`` grid coordinate. Each component must fit in
        three digits (i.e. ``0 <= x, y <= 999``).

    Returns
    -------
    str
        Six-character FOV name with three-digit zero-padded ``x`` and ``y``
        concatenated.
    """
    return f"{pos[0]:03d}{pos[1]:03d}"


def read_shifts_biahub(shifts_path: str) -> dict:
    """Load the ``total_translation`` block from a biahub stitch settings YAML.

    Parameters
    ----------
    shifts_path : str | Path
        Path to a YAML file produced by ``biahub estimate-stitch``.

    Returns
    -------
    dict
        Mapping from ``"{well}/{fov_name}"`` to a 3-element ``[z, y, x]``
        translation in pixel coordinates.
    """
    with open(shifts_path) as file:
        raw_settings = yaml.safe_load(file)

    return raw_settings["total_translation"]
