import ants
import numpy as np

from biahub.core.transform import Transform


def test_numpy_to_ants_transform_zyx():
    T_numpy = np.eye(4)
    T_ants = Transform(T_numpy).to_ants()
    assert isinstance(T_ants, ants.core.ants_transform.ANTsTransform)


def test_ants_to_numpy_transform_zyx():
    T_ants = ants.new_ants_transform(transform_type="AffineTransform")
    T_ants.set_parameters(np.eye(12))
    T_numpy = Transform.from_ants(T_ants).matrix
    assert isinstance(T_numpy, np.ndarray)
    assert T_numpy.shape == (4, 4)
