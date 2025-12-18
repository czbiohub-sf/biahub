"""
Geometric transform class for 2D/3D volumes.

This module provides an immutable Transform class that wraps homogeneous
transformation matrices and provides methods for application, inversion,
composition, and conversion between different representations.

Coordinate convention: ZYX ordering for 3D, YX for 2D.
"""

from __future__ import annotations
from typing import Literal

import numpy as np

from numpy.typing import NDArray
from scipy.ndimage import affine_transform

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
        """
        try:
            import ants
        except ImportError:
            raise ImportError(
                "ANTsPy is required for to_ants(). Install with: pip install antspyx"
            )

        if self._ndim != 3:
            raise NotImplementedError("ANTs conversion only supports 3D transforms")

        # ANTs parameter ordering (from your original convert_transform_to_ants):
        T_ants_style = self._matrix[:, :-1].ravel()
        T_ants_style[-3:] = self._matrix[:3, -1]
        T_ants = ants.new_ants_transform(
            transform_type="AffineTransform",
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
        Based on the conversion from:
        https://sourceforge.net/p/advants/discussion/840261/thread/9fbbaab7/
        """

        T_numpy = T_ants.parameters.reshape((3, 4), order="F")
        T_numpy[:, :3] = T_numpy[:, :3].transpose()
        T_numpy = np.vstack((T_numpy, np.array([0, 0, 0, 1])))

        # Reference:
        # https://sourceforge.net/p/advants/discussion/840261/thread/9fbbaab7/
        # https://github.com/netstim/leaddbs/blob/a2bb3e663cf7fceb2067ac887866124be54aca7d/helpers/ea_antsmat2mat.m
        # T = original translation offset from A
        # T = T + (I - A) @ centering

        T_numpy[:3, -1] += (np.eye(3) - T_numpy[:3, :3]) @ T_ants.fixed_parameters

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
