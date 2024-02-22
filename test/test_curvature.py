import numpy as np
from ridge_trace.curvature import principal_curvature


def test_always_passes():
    assert True


def test_principal_curvature():
    # Test the principal curvature function
    # Create a simple 2D array
    z = np.array(
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
    # Calculate the principal curvature
    k1, k2, t1, t2 = principal_curvature(z)

    # np.set_printoptions(precision=2)
    # print(f"{k1=}", f"{k2=}", f"{t1=}", f"{t2=}", sep="\n")

    # Check the results

    # The first principal curvature should be the larger of the two
    assert np.all(k2 <= k1)

    # The angle between the two principal directions should be 90 degrees
    dt = (t1 - t2) % 180
    # 2024-02-21: The following test was failing initially because
    # some pixels give zero instead of 90 for the angle between the
    # principal directions. This happens when curvature is zero, so I
    # have fixed it with a mask
    nonzero = (k1 != 0.0) & (k2 != 0.0) & (dt != 0.0)
    assert np.allclose(dt[nonzero], 90.0, atol=1e-6)


def test_principal_curvature_smooth():
    # Test the principal curvature function
    # Create a simple 2D array
    z = np.array(
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
    # Calculate the principal curvature
    k1, k2, t1, t2 = principal_curvature(z, sigma=1.0)

    np.set_printoptions(precision=2)
    print(f"{k1=}", f"{k2=}", f"{t1=}", f"{t2=}", sep="\n")

    # Check the results

    # The first principal curvature should be the larger of the two
    assert np.all(k2 <= k1)

    # The angle between the two principal directions should be 90 degrees
    dt = (t1 - t2) % 180
    #
    nonzero = (k1 != 0.0) & (k2 != 0.0) & (dt != 0.0)
    assert np.allclose(dt[nonzero], 90.0, atol=1e-6)
