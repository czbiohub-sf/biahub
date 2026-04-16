import numpy as np
import yaml


def parse_positions(fovs: list) -> np.array:
    """Parse FOV names assumed to follow the ``XXXYYY`` convention into (X, Y) coordinates."""
    fov_positions = []
    for p in fovs:
        p = p.split("/")[-1]  # handle positions either with row/column info or without
        x_cord = int(p[:3])
        y_cord = int(p[3:])
        fov_positions.append((x_cord, y_cord))

    return np.asarray(fov_positions)


def pos_to_name(pos: tuple) -> str:
    """Convert a position tuple to a zero-padded ``XXXYYY`` name."""
    return f"{pos[0]:03d}{pos[1]:03d}"


def read_shifts_biahub(shifts_path: str) -> dict:

    with open(shifts_path) as file:
        raw_settings = yaml.safe_load(file)

    return raw_settings["total_translation"]
