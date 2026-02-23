"""
Geometric transform class for 2D/3D volumes.

This module provides an immutable Transform class that wraps homogeneous
transformation matrices and provides methods for application, inversion,
composition, and conversion between different representations.

Coordinate convention: ZYX ordering for 3D, YX for 2D.
"""

from __future__ import annotations
from typing import Literal, Tuple

import ants
import click
import numpy as np
import scipy

from numpy.typing import NDArray
from scipy.ndimage import affine_transform

from biahub.registration.utils import convert_transform_to_ants

TransformType = Literal["affine", "similarity", "euclidean", "rigid"]
Backend = Literal["scipy", "ants"]


class Transform:
    """
    Geometric transform for 2D/3D volumes.

    Wraps a homogeneous transformation matrix (3×3 for 2D, 4×4 for 3D)
    and provides methods for application, inversion, composition, and conversion.

    This class is immutable - all operations return new Transform instances.

    Coordinate convention: ZYX ordering for 3D, YX for 2D.

    Parameters
    ----------
    matrix : np.ndarray
        Homogeneous transformation matrix. Shape (3, 3) for 2D or (4, 4) for 3D.
    transform_type : TransformType
        Type of transform. Affects estimation constraints but not application.
        - "affine": Full affine (12 DOF in 3D)
        - "similarity": Rotation + translation + uniform scale (7 DOF in 3D)
        - "euclidean" / "rigid": Rotation + translation only (6 DOF in 3D)

    Examples
    --------
    >>> t = Transform.identity(ndim=3)
    >>> t_shifted = Transform.from_translation([0, 10, 20])  # ZYX
    >>> composed = t_shifted @ t
    >>> inverted = t_shifted.invert()
    >>> points_transformed = t_shifted.apply_points(points)
    >>> volume_transformed = t_shifted.apply(volume, output_shape=(64, 128, 128))
    """

    def __init__(
        self,
        matrix: NDArray[np.floating],
        transform_type: TransformType = "affine",
    ):
        matrix = np.asarray(matrix, dtype=np.float64)

        if matrix.shape == (3, 3):
            self._ndim = 2
        elif matrix.shape == (4, 4):
            self._ndim = 3
        else:
            raise ValueError(
                f"Matrix must be (3, 3) for 2D or (4, 4) for 3D, got {matrix.shape}"
            )

        # Store as immutable
        self._matrix = matrix
        self._matrix.flags.writeable = False
        self._type = transform_type

    # ==================== Properties ====================

    @property
    def matrix(self) -> NDArray[np.floating]:
        """The homogeneous transformation matrix (copy)."""
        return self._matrix.copy()

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return self._ndim

    @property
    def transform_type(self) -> TransformType:
        """Type of transform."""
        return self._type

    @property
    def translation(self) -> NDArray[np.floating]:
        """Translation component. ZYX for 3D, YX for 2D."""
        return self._matrix[:-1, -1].copy()

    @property
    def linear(self) -> NDArray[np.floating]:
        """Linear component (rotation, scale, shear)."""
        return self._matrix[:-1, :-1].copy()

    @property
    def is_identity(self) -> bool:
        """Check if this is an identity transform."""
        return np.allclose(self._matrix, np.eye(self._ndim + 1))

    # ==================== Constructors ====================

    @classmethod
    def identity(cls, ndim: int = 3) -> Transform:
        """
        Create an identity transform.

        Parameters
        ----------
        ndim : int
            Number of dimensions (2 or 3).

        Returns
        -------
        Transform
            Identity transform.
        """
        if ndim == 2:
            matrix = np.eye(3, dtype=np.float64)
        elif ndim == 3:
            matrix = np.eye(4, dtype=np.float64)
        else:
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")

        return cls(matrix, transform_type="affine")

    @classmethod
    def from_translation(cls, offset: NDArray[np.floating]) -> Transform:
        """
        Create a pure translation transform.

        Parameters
        ----------
        offset : array-like
            Translation vector. ZYX for 3D, YX for 2D.

        Returns
        -------
        Transform
            Translation transform.

        Examples
        --------
        >>> t = Transform.from_translation([5, 10, 15])  # 3D: Z=5, Y=10, X=15
        >>> t = Transform.from_translation([10, 15])     # 2D: Y=10, X=15
        """
        offset = np.asarray(offset, dtype=np.float64)
        ndim = len(offset)

        if ndim == 2:
            matrix = np.eye(3, dtype=np.float64)
            matrix[:2, 2] = offset
        elif ndim == 3:
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, 3] = offset
        else:
            raise ValueError(f"offset must be 2D or 3D, got shape {offset.shape}")

        return cls(matrix, transform_type="euclidean")

    @classmethod
    def from_skimage(
        cls,
        skimage_transform,
        ndim: int = 3,
    ) -> Transform:
        """
        Create Transform from a scikit-image transform.

        Parameters
        ----------
        skimage_transform : skimage.transform._geometric.GeometricTransform
            A scikit-image transform (EuclideanTransform, SimilarityTransform,
            AffineTransform, etc.)
        ndim : int
            Number of dimensions for the output transform. If the skimage
            transform is 2D and ndim=3, it will be embedded in 3D (YX plane).

        Returns
        -------
        Transform
            New Transform instance.

        Examples
        --------
        >>> from skimage.transform import SimilarityTransform
        >>> sk_tform = SimilarityTransform(scale=0.9, rotation=0.1, translation=(10, 20))
        >>> t = Transform.from_skimage(sk_tform, ndim=3)
        """
        params = skimage_transform.params

        # Determine transform type from skimage class name
        class_name = type(skimage_transform).__name__.lower()
        if "euclidean" in class_name:
            transform_type = "euclidean"
        elif "similarity" in class_name:
            transform_type = "similarity"
        elif "affine" in class_name:
            transform_type = "affine"
        else:
            transform_type = "affine"

        if params.shape == (3, 3):
            # 2D transform
            if ndim == 2:
                return cls(params, transform_type=transform_type)
            elif ndim == 3:
                # Embed 2D transform in 3D (YX plane, Z unchanged)
                matrix_3d = np.eye(4, dtype=np.float64)
                matrix_3d[1:3, 1:3] = params[:2, :2]  # YX linear part
                matrix_3d[1:3, 3] = params[:2, 2]  # YX translation
                return cls(matrix_3d, transform_type=transform_type)
        elif params.shape == (4, 4):
            # 3D transform
            if ndim == 3:
                return cls(params, transform_type=transform_type)
            else:
                raise ValueError("Cannot convert 3D skimage transform to 2D")
        else:
            raise ValueError(f"Unexpected skimage transform shape: {params.shape}")

    # ==================== Algebraic Operations ====================

    def invert(self) -> Transform:
        """
        Return the inverse transform.

        Returns
        -------
        Transform
            New transform that undoes this one.
        """
        return Transform(
            np.linalg.inv(self._matrix),
            transform_type=self._type,
        )

    def compose(self, other: Transform) -> Transform:
        """
        Compose this transform with another: self @ other.

        The result applies `other` first, then `self`.
        For points: composed.apply_points(p) == self.apply_points(other.apply_points(p))

        Parameters
        ----------
        other : Transform
            Transform to compose with.

        Returns
        -------
        Transform
            New composed transform.
        """
        if self.ndim != other.ndim:
            raise ValueError(f"Cannot compose {self.ndim}D and {other.ndim}D transforms")

        # Result type: most general of the two
        type_hierarchy = ["euclidean", "rigid", "similarity", "affine"]
        self_idx = type_hierarchy.index(self._type)
        other_idx = type_hierarchy.index(other._type)
        result_type = type_hierarchy[max(self_idx, other_idx)]

        return Transform(
            self._matrix @ other._matrix,
            transform_type=result_type,
        )

    def __matmul__(self, other: Transform) -> Transform:
        """Compose transforms using @ operator."""
        return self.compose(other)

    # ==================== Application Methods ====================

    def apply_points(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Apply transform to a point cloud.

        Parameters
        ----------
        points : NDArray
            (N, D) array of points where D is ndim (2 or 3).
            Coordinates in ZYX order for 3D, YX for 2D.

        Returns
        -------
        NDArray
            (N, D) transformed points.

        Examples
        --------
        >>> t = Transform.from_translation([0, 5, 10])
        >>> points = np.array([[0, 0, 0], [1, 1, 1]])
        >>> transformed = t.apply_points(points)
        """
        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2:
            raise ValueError(f"points must be 2D array (N, D), got shape {points.shape}")

        if points.shape[1] != self._ndim:
            raise ValueError(f"points must have {self._ndim} columns, got {points.shape[1]}")

        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])

        # Apply transform
        transformed = (self._matrix @ points_homogeneous.T).T

        # Convert back from homogeneous
        return transformed[:, :-1]

    def apply(
        self,
        moving: NDArray,
        reference: NDArray | None = None,
        order: int = 1,
        mode: str = "constant",
        cval: float = 0.0,
        backend: Backend = "scipy",
    ) -> NDArray:
        """
        Apply transform to align moving image with reference space.

        Parameters
        ----------
        moving : NDArray
            Image to transform (ZYX for 3D, YX for 2D).
        reference : NDArray, optional
            Reference image defining output space. If None, uses moving image shape.
        order : int, default=1
            Interpolation order (0=nearest, 1=linear, 3=cubic).
        mode : str, default='constant'
            How to handle boundaries ('constant', 'edge', 'reflect', 'wrap').
        cval : float, default=0.0
            Fill value for constant mode.
        backend : {'scipy', 'ants'}, default='scipy'
            Backend to use for transformation.

        Returns
        -------
        NDArray
            Transformed image in reference space.

        Examples
        --------
        >>> t = Transform.from_translation([0, 5, 10])
        >>> aligned = t.apply(moving, reference=fixed)
        """
        moving = np.asarray(moving)

        # Validate dimensions
        if moving.ndim != self._ndim:
            raise ValueError(f"Expected {self._ndim}D array, got {moving.ndim}D")

        # Determine output shape
        output_shape = reference.shape if reference is not None else moving.shape

        if backend == "scipy":
            return self._apply_scipy(moving, output_shape, order, mode, cval)
        elif backend == "ants":
            return self._apply_ants(moving, reference)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _apply_scipy(
        self,
        moving: NDArray,
        output_shape: tuple,
        order: int,
        mode: str,
        cval: float,
    ) -> NDArray:
        """Apply transform using scipy.ndimage."""
        # scipy applies inverse transform: output[i] = input[inv_matrix @ i]
        inv_matrix = np.linalg.inv(self._matrix)
        affine_matrix = inv_matrix[:-1, :-1]
        offset = inv_matrix[:-1, -1]

        return affine_transform(
            moving,
            matrix=affine_matrix,
            offset=offset,
            output_shape=output_shape,
            order=order,
            mode=mode,
            cval=cval,
        )

    def _apply_ants(self, moving: NDArray, reference: NDArray | None) -> NDArray:
        """Apply transform using ANTs."""
        try:
            import ants
        except ImportError:
            raise ImportError(
                "ANTsPy is required for backend='ants'. Install with: pip install antspyx"
            )

        if self._ndim != 3:
            raise NotImplementedError("ANTs backend only supports 3D transforms")

        # Convert to ANTs images
        moving_ants = ants.from_numpy(moving.astype(np.float32))

        if reference is not None:
            reference_ants = ants.from_numpy(reference.astype(np.float32))
        else:
            reference_ants = moving_ants

        # Convert transform to ANTs
        transform_ants = self.to_ants()

        # Apply
        result_ants = transform_ants.apply_to_image(moving_ants, reference=reference_ants)

        return result_ants.numpy()

    # ==================== ANTs Conversion ====================
    def to_ants(self):
        """
        Convert to ANTs transform.

        Returns
        -------
        ants.ANTsTransform
            ANTs transform object.

        Notes
        -----
        Requires ANTsPy to be installed.
        Works for both 2D and 3D transforms.
        """
        try:
            import ants
        except ImportError:
            raise ImportError(
                "ANTsPy is required for to_ants(). Install with: pip install antspyx"
            )
        if self._ndim not in (2, 3):
            raise ValueError(f"Unsupported ndim: {self._ndim}")
        T_ants_style = self._matrix[:, :-1].ravel()
        T_ants_style[-self._ndim :] = self._matrix[: self._ndim, -1]
        T_ants = ants.new_ants_transform(
            transform_type="AffineTransform",
            dimension=self._ndim,
        )
        T_ants.set_parameters(T_ants_style)
        return T_ants

    @classmethod
    def from_ants(cls, T_ants) -> Transform:
        """
        Create Transform from ANTs transform.

        Parameters
        ----------
        T_ants : ants.ANTsTransform
            ANTs transform object.

        Returns
        -------
        Transform
            New Transform instance.

        Notes
        -----
        Works for both 2D and 3D ANTs transforms.
        Based on conversion from:
        https://sourceforge.net/p/advants/discussion/840261/thread/9fbbaab7/
        """

        params = T_ants.parameters
        fixed_params = T_ants.fixed_parameters
        if len(params) == 6:
            ndim = 2
        elif len(params) == 12:
            ndim = 3
        else:
            raise ValueError(
                f"Unexpected ANTs parameter count: {len(params)}. Expected 6 (2D) or 12 (3D)."
            )

        T_numpy = params.reshape((ndim, ndim + 1), order="F")
        T_numpy[:, :ndim] = T_numpy[:, :ndim].transpose()
        T_numpy = np.vstack((T_numpy, np.array([0] * ndim + [1])))
        T_numpy[:ndim, -1] += (np.eye(ndim) - T_numpy[:ndim, :ndim]) @ fixed_params

        return cls(T_numpy)

    # ==================== Serialization ====================

    def to_list(self) -> list[list[float]]:
        """Convert matrix to nested list (for JSON/YAML serialization)."""
        return self._matrix.tolist()

    @classmethod
    def from_list(
        cls,
        matrix_list: list[list[float]],
        transform_type: TransformType = "affine",
    ) -> Transform:
        """Create Transform from nested list."""
        return cls(np.array(matrix_list), transform_type=transform_type)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "matrix": self.to_list(),
            "transform_type": self._type,
            "ndim": self._ndim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transform:
        """Create Transform from dictionary."""
        return cls(
            np.array(data["matrix"]),
            transform_type=data.get("transform_type", "affine"),
        )

    # ==================== String Representations ====================

    def __repr__(self) -> str:
        return (
            f"Transform(ndim={self._ndim}, type='{self._type}', "
            f"translation={self.translation.round(3).tolist()})"
        )

    def __str__(self) -> str:
        matrix_str = np.array2string(self._matrix, precision=4, suppress_small=True)
        return f"Transform({self._type}, {self._ndim}D)\n{matrix_str}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return (
            self._ndim == other._ndim
            and self._type == other._type
            and np.allclose(self._matrix, other._matrix)
        )

    def __hash__(self) -> int:
        # For use in sets/dicts - based on rounded matrix values
        return hash((self._ndim, self._type, tuple(self._matrix.round(6).ravel())))


def apply_stabilization_transform(
    zyx_data: np.ndarray,
    list_of_shifts: list[np.ndarray],
    t_idx: int,
    output_shape: tuple[int, int, int] = None,
):
    """
    Apply stabilization transformations to 3D or 4D volumetric data.

    This function applies a time-indexed stabilization transformation to a single 3D (Z, Y, X) volume
    or a 4D (C, Z, Y, X) volume using a precomputed list of transformations.

    Parameters:
    - zyx_data (np.ndarray): Input 3D (Z, Y, X) or 4D (C, Z, Y, X) volumetric data.
    - list_of_shifts (list[np.ndarray]): List of transformation matrices (one per time index).
    - t_idx (int): Time index corresponding to the transformation to apply.
    - output_shape (tuple[int, int, int], optional): Desired shape of the output stabilized volume.
                                                     If None, the shape of `zyx_data` is used.
                                                     Defaults to None.

    Returns:
    - np.ndarray: The stabilized 3D (Z, Y, X) or 4D (C, Z, Y, X) volume.

    Notes:
    - If `zyx_data` is 4D, the function recursively applies stabilization to each channel (C).
    - Uses ANTsPy for applying the transformation to the input data.
    - Handles `NaN` values in the input by replacing them with 0 before applying the transformation.
    - Echoes the transformation matrix for debugging purposes when verbose logging is enabled.
    """

    if output_shape is None:
        output_shape = zyx_data.shape[-3:]
    from biahub.core.transform import convert_transform_to_ants

    # Get the transformation matrix for the current time index
    tx_shifts = convert_transform_to_ants(list_of_shifts[t_idx])

    if zyx_data.ndim == 4:
        stabilized_czyx = np.zeros((zyx_data.shape[0],) + output_shape, dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            stabilized_czyx[c] = apply_stabilization_transform(
                zyx_data[c], list_of_shifts, t_idx, output_shape
            )
        return stabilized_czyx
    else:
        click.echo(f'shifting matrix with t_idx:{t_idx} \n{list_of_shifts[t_idx]}')
        target_zyx_ants = ants.from_numpy(np.zeros((output_shape), dtype=np.float32))

        zyx_data = np.nan_to_num(zyx_data, nan=0)
        zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
        stabilized_zyx = tx_shifts.apply_to_image(
            zyx_data_ants, reference=target_zyx_ants
        ).numpy()

    return stabilized_zyx


def apply_affine_transform(
    zyx_data: np.ndarray,
    matrix: np.ndarray,
    output_shape_zyx: Tuple,
    method="ants",
    interpolation: str = "linear",
    crop_output_slicing: bool = None,
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    zyx_data : np.ndarray
        3D input array to be transformed
    matrix : np.ndarray
        3D Homogenous transformation matrix
    output_shape_zyx : Tuple
        output target zyx shape
    method : str, optional
        method to use for transformation, by default 'ants'
    interpolation: str, optional
        interpolation mode for ants, by default "linear"
    crop_output : bool, optional
        crop the output to the largest interior rectangle, by default False

    Returns
    -------
    np.ndarray
        registered zyx data
    """

    Z, Y, X = output_shape_zyx
    if crop_output_slicing is not None:
        Z_slice, Y_slice, X_slice = crop_output_slicing
        Z = Z_slice.stop - Z_slice.start
        Y = Y_slice.stop - Y_slice.start
        X = X_slice.stop - X_slice.start

    # TODO: based on the signature of this function, it should not be called on 4D array
    if zyx_data.ndim == 4:
        registered_czyx = np.zeros((zyx_data.shape[0], Z, Y, X), dtype=np.float32)
        for c in range(zyx_data.shape[0]):
            registered_czyx[c] = apply_affine_transform(
                zyx_data[c],
                matrix,
                output_shape_zyx,
                method=method,
                interpolation=interpolation,
                crop_output_slicing=crop_output_slicing,
            )
        return registered_czyx
    else:
        # Convert nans to 0
        zyx_data = np.nan_to_num(zyx_data, nan=0)

        # NOTE: default set to ANTS apply_affine method until we decide we get a benefit from using cupy
        # The ants method on CPU is 10x faster than scipy on CPU. Cupy method has not been bencharked vs ANTs

        if method == "ants":
            # The output has to be a ANTImage Object
            empty_target_array = np.zeros((output_shape_zyx), dtype=np.float32)
            target_zyx_ants = ants.from_numpy(empty_target_array)

            T_ants = convert_transform_to_ants(matrix)

            zyx_data_ants = ants.from_numpy(zyx_data.astype(np.float32))
            registered_zyx = T_ants.apply_to_image(
                zyx_data_ants, reference=target_zyx_ants, interpolation=interpolation
            ).numpy()

        elif method == "scipy":
            registered_zyx = scipy.ndimage.affine_transform(zyx_data, matrix, output_shape_zyx)

        else:
            raise ValueError(f"Unknown method {method}")

        # Crop the output to the largest interior rectangle
        if crop_output_slicing is not None:
            registered_zyx = registered_zyx[Z_slice, Y_slice, X_slice]

    return registered_zyx
