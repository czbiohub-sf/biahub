"""Final channel names for the assembled plate.

Implements the convention in czbiohub-sf/biahub#291. ``biahub concatenate`` applies
it with ``rename_channels: true``, so QC, tracking and later steps read the final
names. Patterns are matched in order and the first match wins.

===========================================  ==========================
incoming                                     final name
===========================================  ==========================
``BF - Oblique``                             ``BF``
``Phase3D*``                                 ``Phase3D``
``nuclei``                                   ``nuclei_prediction``
``membrane``                                 ``membrane_prediction``
already canonical                            left alone
anything else (e.g. ``mCherry EX561 ...``)   ``raw <original>``
===========================================  ==========================

The ``raw`` prefix marks unprocessed detector output, so raw and derived channels
are distinguishable at a glance. The mapping is idempotent: a name that already
starts with ``raw `` is not prefixed again.

``nuclei`` becomes ``nuclei_prediction`` because ``biahub virtual-stain`` names its
outputs verbatim from ``target_channel``, dropping the ``_prediction`` suffix that
viscy's ``HCSPredictionWriter`` used to append (biahub#288), while the rest of
biahub still keys off the suffix.
"""

from collections import defaultdict
from fnmatch import fnmatch

RENAMES = {
    "BF - Oblique": "BF",
    "Phase3D*": "Phase3D",
    "nuclei": "nuclei_prediction",
    "membrane": "membrane_prediction",
}

PASSTHROUGH = [
    "BF",
    "Phase3D",
    "nuclei_prediction",
    "membrane_prediction",
]

UNMATCHED_PREFIX = "raw "


def canonical_channel_name(name: str) -> str:
    """Return the final name for a channel.

    Parameters
    ----------
    name : str
        Channel name as it appears in the source store.

    Returns
    -------
    str
        The final name. Equal to ``name`` when it is already canonical.
    """
    for pattern, new_name in RENAMES.items():
        if fnmatch(name, pattern):
            return new_name
    if name in PASSTHROUGH or name.startswith(UNMATCHED_PREFIX):
        return name
    return f"{UNMATCHED_PREFIX}{name}"


def canonical_channel_names(names: list[str]) -> list[str]:
    """Map channel names to final names, refusing mappings that merge channels.

    Parameters
    ----------
    names : list[str]
        Distinct source channel names.

    Returns
    -------
    list[str]
        Final names, in the order of ``names``.

    Raises
    ------
    ValueError
        If two source names map to the same final name.
    """
    final_names = [canonical_channel_name(name) for name in names]
    sources = defaultdict(list)
    for name, final in zip(names, final_names, strict=True):
        sources[final].append(name)
    collisions = {final: srcs for final, srcs in sources.items() if len(srcs) > 1}
    if collisions:
        detail = "; ".join(f"{srcs} -> {final!r}" for final, srcs in collisions.items())
        raise ValueError(
            f"rename_channels maps several source channels to one final name: {detail}. "
            "Take only one of them with channel_names."
        )
    return final_names
