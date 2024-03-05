import numpy as np
from ridge_trace.trace import (
    interpolate_at_points,
    Point,
    PointArray,
    StraightLine,
    peak_indices,
)


def test_peak_indices(simple_2d_array):
    indices = peak_indices(simple_2d_array)
    assert simple_2d_array[indices] == np.nanmax(simple_2d_array)


def test_Point():
    """Test that Point can be created with x and y attributes."""
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2


def test_PointArray():
    """Test that PointArray can be created from a list of Points."""
    points = [Point(1, 2), Point(3, 4), Point(5, 6)]
    p = PointArray(points)
    assert np.allclose(p.x, [1, 3, 5])
    assert np.allclose(p.y, [2, 4, 6])


def test_interpolate_at_points(simple_2d_array):
    """Test that interpolate_at_points returns the expected result."""
    points = PointArray([Point(x, x) for x in [1, 2, 3, 4, 5]])
    result = interpolate_at_points(simple_2d_array, points)
    expected = np.array([1, 2, 2, 2, 1])
    np.testing.assert_allclose(result, expected)


class TestStraightLine:
    def test_slope(self):
        """Test a straight line at an arbitrary angle has the correct slope."""
        x0, y0 = 5.5, 1
        theta = 157
        slope = np.tan(np.deg2rad(theta))
        line = StraightLine(x0, y0, theta=theta, step=0.1, nsteps=10)
        assert np.allclose(slope * (line.points.x - x0), line.points.y - y0)

    def test_zeros_image_values(self):
        """A line through an image with all zeros should return all zeros."""
        line = StraightLine(1, 1, theta=0.0, step=0.1, nsteps=10)
        image = np.zeros((10, 10))
        result = line.image_values(image)
        assert np.all(result == 0)

    def test_ones_image_values(self):
        """A line through an image with all ones should return all ones."""
        line = StraightLine(1, 1, theta=0.0, step=0.1, nsteps=10)
        image = np.ones((10, 10))
        result = line.image_values(image)
        assert np.all(result == 1)

    def test_image_boundary(self):
        """Test that we get NaNs beyond the boundary of the image."""
        line = StraightLine(1, 1, theta=0.0, step=1, nsteps=10)
        image = np.ones((10, 10))
        values = line.image_values(image)
        # mask corresponding to finite values along the line
        is_finite_value = np.isfinite(values)
        # mask corresponding to line points that are within the image
        is_inside_image = (
            (line.points.x >= 0)
            & (line.points.x < 10)
            & (line.points.y >= 0)
            & (line.points.y < 10)
        )
        expected = np.array(
            [True, True, True, True, True, True, True, True, True, False, False]
        )
        # these should all be the same
        assert np.all(is_finite_value == expected)
        assert np.all(is_inside_image == expected)
        # and all finite values should be 1
        assert np.all(values[is_finite_value] == 1)
