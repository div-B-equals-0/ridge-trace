"""Interpolate at arbitrary points in an image."""

import numpy as np
from dataclasses import dataclass
from scipy.ndimage import map_coordinates


@dataclass
class Point:
    x: float
    y: float

    def take_step(self, theta: float, step: float):
        return Point(
            self.x + step * np.cos(np.rad2deg(theta)),
            self.y + step * np.sin(np.rad2deg(theta)),
        )


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
        points = [Point(x0, y0)]
        for i in range(nsteps):
            self.points.append(self.points[-1].take_step(theta, step))
        self.points = PointArray(points)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

    def __iter__(self):
        return iter(self.points)

    def image_values(self, image):
        return interpolate_at_points(image, self.points)


def interpolate_at_points(image, points: PointArray):
    """Interpolate at arbitrary points in an image.

    Args:
        image: 2D numpy array.
        points: PointArray of length n where n is the number of points.

    Returns:
        1D numpy array of shape (n,) where n is the number of points.
    """
    return map_coordinates(image, [points.x, points.y], order=1)
