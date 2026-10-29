"""
ANTs-based intensity registration module.

Provides functions for registering volumetric imaging data using the ANTsPy
library's optimization-based registration. This complements bead-based
registration by directly optimizing image similarity metrics.

Key conventions
---------------
- Coordinates are in ZYX order for 3D data.
- "mov" / "moving" refers to the source channel being aligned.
- "ref" / "reference" refers to the fixed target channel.
- Transforms are 4x4 homogeneous matrices stored as Transform objects.
"""

from __future__ import annotations

import os
import tempfile

import ants
import click
import numpy as np

from numpy.typing import ArrayLike

from biahub.core.transform import Transform
from biahub.registration.estimators import EstimationError
from biahub.settings import AffineTransformSettings, AntsRegistrationSettings

DEFAULT_ANTS_KWARGS = {
    "type_of_transform": "Similarity",
    "aff_shrink_factors": (6, 3, 1),
    "aff_iterations": (2100, 1200, 50),
    "aff_smoothing_sigmas": (2, 1, 0),
}


# ANTs samples voxels at random in its affine stage (aff_random_sampling_rate): a
# fixed seed makes a run repeatable for a given thread count.
ANTS_RANDOM_SEED = 1  # as ANTsPy's reproducible mode; 0 would mean a random seed


def _itk_threads() -> str:
    """Return the threads ITK may use: the user's setting, else this task's CPUs.

    ANTs parallelizes inside a task on ITK threads; left alone, ITK sizes its pool from
    the cores it can see, which on a cluster node can exceed the CPUs reserved for the
    task (and changes the result between machines).
    """
    if os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        return os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return os.environ["SLURM_CPUS_PER_TASK"]
    if hasattr(os, "sched_getaffinity"):  # not on macOS
        return str(len(os.sched_getaffinity(0)))
    return str(os.cpu_count() or 1)


def use_task_threads() -> None:
    """Give ITK this task's CPUs (`_itk_threads`). Call before any ITK operation in the process.

    ITK reads the thread count once, at its first threaded operation (e.g. warping the
    moving volume onto the seed), and keeps it.
    """
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = _itk_threads()


def estimate(
    ref: np.ndarray,
    mov: np.ndarray,
    initial: Transform,
    ref_mask: np.ndarray | None = None,
    mov_mask: np.ndarray | None = None,
    verbose: bool = False,
    ants_kwargs: dict | None = None,
) -> Transform:
    """Register `mov` onto `ref` with ANTs, starting from `initial`.

    Works for 2D (Y, X) and 3D (Z, Y, X) arrays. `initial` and the result are forward
    (moving -> reference, the engine's direction); ANTs works in the inverse direction
    (reference -> moving, the one it resamples with), converted here and nowhere else.
    ANTs holds `initial` fixed and optimizes on top of it; the result includes it.

    `ref_mask` / `mov_mask` (boolean, the shape of `ref` / `mov`) restrict the comparison
    to where they are True; points outside either volume never count.
    """
    if ref.ndim not in (2, 3) or mov.ndim not in (2, 3):
        raise ValueError(
            f"Images must be 2D or 3D, got ref.ndim={ref.ndim}, mov.ndim={mov.ndim}"
        )
    if ref.ndim != mov.ndim:
        raise ValueError(f"Dimension mismatch: ref.ndim={ref.ndim}, mov.ndim={mov.ndim}")

    ants_kwargs = dict(DEFAULT_ANTS_KWARGS if ants_kwargs is None else ants_kwargs)
    ants_kwargs.setdefault("random_seed", ANTS_RANDOM_SEED)
    use_task_threads()

    def as_ants_mask(m):
        return None if m is None else ants.from_numpy(np.asarray(m, dtype=np.float32))

    if verbose:
        click.echo(f"Optimizing registration parameters using ANTs with kwargs: {ants_kwargs}")
    with tempfile.TemporaryDirectory() as tmp:
        # Without an initial transform ANTsPy would first align the intensity centres of
        # mass, replacing where `initial` starts.
        initial_path = os.path.join(tmp, "initial.mat")
        ants.write_transform(initial.invert().to_ants(), initial_path)
        try:
            reg = ants.registration(
                fixed=ants.from_numpy(np.asarray(ref, dtype=np.float32)),
                moving=ants.from_numpy(np.asarray(mov, dtype=np.float32)),
                initial_transform=[initial_path],
                mask=as_ants_mask(ref_mask),
                moving_mask=as_ants_mask(mov_mask),
                verbose=verbose,
                **ants_kwargs,
            )
        except (RuntimeError, ValueError) as error:  # ITK's failures: a failed estimate
            raise EstimationError(f"ANTs registration failed: {error}") from error
    inverse = Transform.from_ants(ants.read_transform(reg["fwdtransforms"][0]))
    return inverse.invert()


_ANTS_TRANSFORM_TYPE = {
    "euclidean": "Rigid",
    "rigid": "Rigid",
    "similarity": "Similarity",
    "affine": "Affine",
}


class AntsEstimator:
    """TransformEstimator using ANTs intensity-based optimization.

    One pass: ANTs registers `mov` onto `ref` starting from the seed (passed as its
    initial transform, so the moving volume is never warped first) and returns the full
    forward transform, seed included. Preparation -- Sobel, clipping, masks -- is the
    config's preprocess blocks, applied before this; `ref_mask` / `mov_mask` restrict the
    comparison.
    """

    def __init__(self, ants_kwargs: dict | None = None, verbose: bool = False):
        self.ants_kwargs = dict(ants_kwargs) if ants_kwargs else dict(DEFAULT_ANTS_KWARGS)
        self.verbose = verbose

    @classmethod
    def from_settings(
        cls,
        ants_registration_settings: AntsRegistrationSettings,
        affine_transform_settings: AffineTransformSettings,
        verbose: bool = False,
    ) -> AntsEstimator:
        """Transform family from the affine settings."""
        ants_kwargs = dict(DEFAULT_ANTS_KWARGS)
        ants_kwargs["type_of_transform"] = _ANTS_TRANSFORM_TYPE[
            affine_transform_settings.transform_type
        ]
        return cls(ants_kwargs=ants_kwargs, verbose=verbose)

    def estimate(
        self,
        mov: ArrayLike,
        ref: ArrayLike,
        seed: Transform | None = None,
        ref_mask: np.ndarray | None = None,
        mov_mask: np.ndarray | None = None,
    ) -> Transform:
        ref = np.asarray(ref, dtype=np.float32)
        return estimate(
            ref=ref,
            mov=np.asarray(mov, dtype=np.float32),
            initial=seed if seed is not None else Transform.identity(ref.ndim),
            ref_mask=ref_mask,
            mov_mask=mov_mask,
            verbose=self.verbose,
            ants_kwargs=self.ants_kwargs,
        )
