import numpy as np
from ridge_trace.trace import interpolate_at_points, Point, PointArray, StraightLine


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
        """Test a straight line at an angle has the correct slope."""
        x0, y0 = 5.5, 1
        theta = 157
        slope = np.tan(np.deg2rad(theta))
        line = StraightLine(x0, y0, theta=theta, step=0.1, nsteps=10)
        assert np.allclose(slope * (line.points.x - x0), line.points.y - y0)

    def test_zeros_image_values(self):
        line = StraightLine(1, 1, theta=0.0, step=0.1, nsteps=10)
        image = np.zeros((10, 10))
        result = line.image_values(image)
        assert np.all(result == 0)

    def test_ones_image_values(self):
        line = StraightLine(1, 1, theta=0.0, step=0.1, nsteps=10)
        image = np.ones((10, 10))
        result = line.image_values(image)
        assert np.all(result == 1)

    def test_image_boundary(self):
        """Test that we get NaNs beyond the boundary of the image."""
        line = StraightLine(1, 1, theta=0.0, step=1, nsteps=10)
        image = np.ones((10, 10))
        result = line.image_values(image)
        finite_elements = np.isfinite(result)
        assert np.all(result[finite_elements] == 1)
        # Last two elements are NaN
        assert len(result[~finite_elements]) == 2
