"""Interpolate at arbitrary points in an image."""

import numpy as np
from dataclasses import dataclass
from scipy.ndimage import map_coordinates
from . import curvature


@dataclass
class Point:
    x: float
    y: float

    def take_step(self, theta: float, distance: float):
        return Point(
            self.x + distance * np.cos(np.deg2rad(theta)),
            self.y + distance * np.sin(np.deg2rad(theta)),
        )

    def image_value(self, image):
        return interpolate_at_points(image, self)


class PointArray:
    def __init__(self, points: list[Point]):
        self.x = np.array([p.x for p in points])
        self.y = np.array([p.y for p in points])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return Point(self.x[index], self.y[index])


class StraightLine:
    def __init__(self, x0, y0, theta, step=1.0, nsteps=100):
        self.x0 = x0
        self.y0 = y0
        self.theta = theta
        self.step = step
        self.nsteps = nsteps
        point_list = [Point(x0, y0)]
        for i in range(nsteps):
            point_list.append(point_list[-1].take_step(theta, step))
        self.points = PointArray(point_list)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

    def __iter__(self):
        return iter(self.points)

    def __repr__(self):
        return f"StraightLine(x0={self.x0}, y0={self.y0}, theta={self.theta})"

    def __str__(self):
        return (
            f"Straight line from ({self.x0}, {self.y0}) at angle {self.theta} degrees"
        )

    def image_values(self, image):
        return interpolate_at_points(image, self.points)


def interpolate_at_points(image, points: PointArray | Point):
    """Interpolate at arbitrary points in an image.

    Args:
        image: 2D numpy array.
        points: PointArray of length n where n is the number of points.

    Returns:
        1D numpy array of shape (n,) where n is the number of points.
    """
    # Note that order of coordinates is row, column in
    # map_coordinates, which is y, x in our case
    return map_coordinates(image, [points.y, points.x], order=1, cval=np.nan)


def peak_indices(image):
    """Find indices of the peak pixel in 2d image"""
    return np.unravel_index(np.nanargmax(image), image.shape)


class RidgeLine:
    """A curved line that traces a ridge in an image."""

    def __init__(
        self,
        x0: float,
        y0: float,
        ic: curvature.ImageCurvature,
        step0: float = 1.0,
        maxsteps: int = 100,
    ):
        """Create a ridge line.

        Args:
            x0: x-coordinate of the starting point.
            y0: y-coordinate of the starting point.
            ic: ImageCurvature instance with image, curvatures, and gradients
            th2_image: 2d image of the principal curvature angle.
            step0: Initial step size.
            maxsteps: Maximum number of steps to take.

        Note that the ridge in z_image should be a local maximum,
        so pass negative of the array to trace a minimum (for instance, curvature).
        """
        self.x0 = x0
        self.y0 = y0
        self.z_image = z_image
        self.th2_image = th2_image
        self.step0 = step0
        self.maxsteps = maxsteps
        point_list = [Point(x0, y0)]
        for i in range(maxsteps):
            current_point = point_list[-1]
            trial_theta = current_point.image_value(th2_image)
            trial_point = current_point.take_step(trial_theta, step0)

            point_list.append(point_list[-1].take_step(theta, step))
        self.points = PointArray(point_list)

    def take_explore_step(self, theta, distance):
        return Point(
            self.x + distance * np.cos(np.deg2rad(theta)),
            self.y + distance * np.sin(np.deg2rad(theta)),
        )

    def look_sideways(self, image, theta, distance):
        """Look sideways from the current point to find the direction of the ridge."""
        # Look at the image values at the current point and at a point
        # a small distance away in the direction perpendicular to the ridge.
        # The difference between the two values will be positive if the
        # ridge is to the left of the current point and negative if it is to the right.
        current_value = interpolate_at_points(image, PointArray([self]))
        sideways_value = interpolate_at_points(
            image, PointArray([self.take_explore_step(theta + 90, distance)])
        )
        return sideways_value - current_value

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

    def __iter__(self):
        return iter(self.points)

    def __repr__(self):
        return f"RidgeLine(x0={self.x0}, y0={self.y0}, theta={self.theta})"

    def __str__(self):
        return f"Ridge line from ({self.x0}, {self.y0}) at angle {self.theta} degrees"

    def image_values(self, image):
        return interpolate_at_points(image, self.points)

    def peak_index(self, image):
        return peak_indices(self.image_values(image))
