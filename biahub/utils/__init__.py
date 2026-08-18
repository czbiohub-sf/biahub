"""Helpers shared across the step CLIs, grouped by concern.

* :mod:`biahub.utils.array_ops` -- NumPy cropping and empty-frame detection.
* :mod:`biahub.utils.cellpose` -- cellpose device policy and weight caching.
* :mod:`biahub.utils.cluster` -- per-position resource sizing, executor choice.
* :mod:`biahub.utils.config` -- YAML configs <-> pydantic settings models.
* :mod:`biahub.utils.ngff` -- OME-Zarr store paths, versions, and provenance.

Import from the submodule rather than from this package. These helpers share no
theme beyond being shared, and a flat namespace is what turned their previous
home, ``biahub.cli.utils``, into a grab-bag.
"""
