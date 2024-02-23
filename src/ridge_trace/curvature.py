"""Calculate principal curvatures and principal directions from an image"""

import numpy as np
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel


class ImageCurvature:
    """Calculate principal curvatures and principal directions from an image."""

    def __init__(self, image, smooth=None, sharp=None, max=None, min=None, bbox=None):
        """Calculate principal curvatures and principal directions from an image

        Parameters
        ----------
        image : array_like
            The image to calculate the curvatures from
        smooth : float, optional
            The scale of the low-pass filter to apply to the image.
            This is the sigma (in pixels) of a Gaussian kernel that will be convolved
            with the image, which will remove any smaller-scale noise
        sharp : float, optional
            The scale of the high-pass filter to apply to the image.
            This is the sigma (in pixels) of a Gaussian kernel that will be convolved
            with the image before subtracting from the original image, which will
            remove any smooth larger-scale features
        max : float, optional
            The maximum value to threshold the image at. This is useful for removing
            bright stars or other features that may dominate the curvature calculation.
            Pixels with values greater than `max` will be set to NaN, so that they will
            be interpolated over by the smoothing.
        min : float, optional
            The minimum value to threshold the image at. This is useful for removing
            negative instrumental artefacts or noisy regions that may interfere with
            the curvature calculation. Pixels with values greater than `min` will be
            set to NaN, so that they will be interpolated over by the smoothing.
        bbox : x1 y1 x2 y2, optional
            The bounding box of the region to calculate the curvatures in. If not
            provided, the entire image will be used. An additional margin is added
            around the bounding box for the spatial filtering and derivative
            determinations to ensure that the smoothing does not introduce edge effects.
        """
        self.rawimage = image
        self.image = image.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            margin = 0  # Minimum for second derivatives
            if smooth is not None:
                # Extra margin for low-pass filter
                margin += int(3 * smooth)
            if sharp is not None:
                # Extra margin for high-pass filter
                margin += int(sharp)
            assert (
                (x1 - margin >= 0)
                and (y1 - margin >= 0)
                and (x2 + margin <= self.rawimage.shape[1])
                and (y2 + margin <= self.rawimage.shape[0])
            ), "Bounding box is too close to the edge of the image"
            # Full window is used for filtering and derivative calculations
            self.full_window = (
                slice(y1 - margin, y2 + margin),
                slice(x1 - margin, x2 + margin),
            )
            self.image = self.image[self.full_window]
            # The window is what we return
            self.window = slice(y1, y2), slice(x1, x2)
            # trim is the window into full_window
            self.trim = (
                slice(margin, margin + (y2 - y1)),
                slice(margin, margin + (x2 - x1)),
            )
        else:
            self.window = slice(None), slice(None)
            self.trim = slice(None), slice(None)

        # This all seems far too complicated. Let's at least check
        # that the two ways of slicing the image are consistent
        assert np.all(
            self.image[self.trim] == self.rawimage[self.window]
        ), "Image window and full window do not match"

        self.smooth = smooth
        self.sharp = sharp
        self.max = max
        self.min = min
        self._apply_spatial_filters()
        self._find_curvatures()

    def _apply_spatial_filters(self):
        """Apply thresholding and hi-pass and lo-pass filters to the image"""
        # Threshold the image, replacing with NaN
        if self.max is not None:
            self.image[self.image > self.max] = np.nan
        if self.min is not None:
            self.image[self.image < self.min] = np.nan
        # Hi-pass filter to remove smooth features on scales > `sharp`
        if self.sharp is not None:
            kernel = Gaussian2DKernel(self.sharp)
            blurred = convolve_fft(self.image, kernel)
            self.image = self.image - blurred
        # Lo-pass filter to remove noise on scales < `smooth`
        if self.smooth is not None:
            kernel = Gaussian2DKernel(self.smooth)
            self.image = convolve_fft(self.image, kernel)
        # Remove the margin from the image
        self.image = self.image[self.trim]

    def _find_curvatures(self):
        """Calculate principal curvatures and principal directions"""
        # Calculate the gradient
        self.grad_x, self.grad_y = np.gradient(self.image)
        # Calculate the second derivatives
        self.grad_xx, self.grad_xy = np.gradient(self.grad_x)
        self.grad_yx, self.grad_yy = np.gradient(self.grad_y)

        # Calculate the Gaussian and mean curvatures
        self.kgauss = self.grad_xx * self.grad_yy - self.grad_xy * self.grad_yx
        self.kmean = (self.grad_xx + self.grad_yy) / 2
        # Calculate the principal curvatures
        self.kappa1 = self.kmean + np.sign(self.kmean) * np.sqrt(
            self.kmean**2 - self.kgauss
        )
        self.kappa2 = self.kmean - np.sign(self.kmean) * np.sqrt(
            self.kmean**2 - self.kgauss
        )

        # Calculate the principal directions (in degrees cc from x-axis)
        self.theta1 = np.rad2deg(np.arctan2(self.kappa1 - self.grad_xx, -self.grad_xy))
        self.theta2 = np.rad2deg(np.arctan2(self.kappa2 - self.grad_xx, -self.grad_xy))
