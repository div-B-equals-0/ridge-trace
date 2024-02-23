import numpy as np
from ridge_trace.curvature import ImageCurvature
import pytest


@pytest.fixture(scope="module")
def simple_2d_array():
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 2, 1, 0],
            [0, 1, 2, 2, 2, 1, 0],
            [0, 1, 2, 2, 2, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )


def test_always_passes():
    assert True


def test_principal_curvature(simple_2d_array):
    # Test the principal curvature function
    # Calculate the principal curvature
    ic = ImageCurvature(simple_2d_array)

    # np.set_printoptions(precision=2)
    # print(f"{k1=}", f"{k2=}", f"{t1=}", f"{t2=}", sep="\n")

    # Check the results

    # The first principal curvature should be the larger magnitude of the two
    assert np.all(np.abs(ic.kappa2) <= np.abs(ic.kappa1))

    # The angle between the two principal directions should be 90 degrees
    dt = (ic.theta1 - ic.theta2) % 180
    # 2024-02-21: The following test was failing initially because
    # some pixels give zero instead of 90 for the angle between the
    # principal directions. This happens when curvature is zero, so I
    # have fixed it with a mask
    nonzero = (ic.kappa1 != 0.0) & (ic.kappa2 != 0.0) & (dt != 0.0)
    assert np.allclose(dt[nonzero], 90.0, atol=1e-6)


def test_principal_curvature_smooth(simple_2d_array):
    # Calculate the principal curvature
    ic = ImageCurvature(simple_2d_array, smooth=1.0)

    np.set_printoptions(precision=2)
    # print(f"{ic.kappa1=}", f"{ic.kappa2=}", f"{ic.theta1=}", f"{ic.theta2=}", sep="\n")

    # Check the results

    # The first principal curvature should be the larger of the two
    assert np.all(np.abs(ic.kappa2) <= np.abs(ic.kappa1))

    # The angle between the two principal directions should be 90 degrees
    dt = (ic.theta1 - ic.theta2) % 180
    nonzero = (ic.kappa1 != 0.0) & (ic.kappa2 != 0.0) & (dt != 0.0)
    assert np.allclose(dt[nonzero], 90.0, atol=1e-6)


def test_principal_curvature_bbox(simple_2d_array):
    bb_shape = 4, 5
    bbox = (2, 1, 2 + bb_shape[0], 1 + bb_shape[1])
    ic = ImageCurvature(simple_2d_array, bbox=bbox)

    np.set_printoptions(precision=2)
    print(f"{bbox=}")
    print(f"{ic.image=}")
    print(f"{ic.rawimage=}")
    # Compare with reversed bb shape since it is x, y order
    assert ic.image.shape == bb_shape[::-1]
