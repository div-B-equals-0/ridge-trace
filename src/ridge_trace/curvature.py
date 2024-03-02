"""Calculate principal curvatures and principal directions from an image"""

import numpy as np
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel


class ImageCurvature:
    """Calculate principal curvatures and principal directions from an image."""

    def __init__(
        self,
        image,
        smooth=None,
        sharp=None,
        max_cutoff=None,
        min_cutoff=None,
        bbox=None,
        preserve_nan=True,
        scales=(1.0, 1.0),
    ):
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
        max_cutoff : float, optional
            The maximum value to threshold the image at. This is useful for removing
            bright stars or other features that may dominate the curvature calculation.
            Pixels with values greater than `max_cutoff` will be set to NaN, so that they will
            be interpolated over by the smoothing (if `preserve_nan=False`).
        min_cutoff : float, optional
            The minimum value to threshold the image at. This is useful for removing
            negative instrumental artefacts or noisy regions that may interfere with
            the curvature calculation. Pixels with values greater than `min_cutoff` will be
            set to NaN, so that they will be interpolated over by the smoothing
            (if `preserve_nan=False`).
        bbox : x1 y1 x2 y2, optional
            The bounding box of the region to calculate the curvatures in. If not
            provided, the entire image will be used. An additional margin is added
            around the bounding box for the spatial filtering and derivative
            determinations to ensure that the smoothing does not introduce edge effects.
        preserve_nan : bool, optional
            If True, any NaNs in the input image, together with any introduced by thresholding
            will be preserved after filtering.
        scales : tuple of floats, optional
            The scales of the image given as (value, length) with length in pixels.
            Gradients will be scaled by value/length and curvatures by value/length^2.
        """
        self.rawimage = image
        self.image = image.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            margin = 0  # Minimum for second derivatives
            if smooth is not None:
                # Extra margin for low-pass filter
                margin = max(margin, int(3 * smooth))
            if sharp is not None:
                # Extra margin for high-pass filter
                margin = max(margin, 2 * int(sharp))
                # Note that this can take us outside the image if we are too close to the edge
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
        self.max_cutoff = max_cutoff
        self.min_cutoff = min_cutoff
        self.value_scale, self.length_scale = scales
        self._apply_spatial_filters(preserve_nan)
        self._find_curvatures()

    def _apply_spatial_filters(self, preserve_nan):
        """Apply thresholding and hi-pass and lo-pass filters to the image"""
        nan_mask = np.isnan(self.image)
        # Threshold the image, replacing with NaN
        if self.max_cutoff is not None:
            mask_hi = self.image > self.max_cutoff
            self.image[mask_hi] = np.nan
            nan_mask = nan_mask | mask_hi
        if self.min_cutoff is not None:
            mask_lo = self.image < self.min_cutoff
            self.image[mask_lo] = np.nan
            nan_mask = nan_mask | mask_lo
        # Hi-pass filter to remove smooth features on scales > `sharp`
        if self.sharp is not None:
            kernel = Gaussian2DKernel(self.sharp)
            # We use the wrap boundary condition to minimise edge effects
            blurred = convolve_fft(self.image, kernel, boundary="wrap")
            self.image = self.image - blurred
        # Lo-pass filter to remove noise on scales < `smooth`
        if self.smooth is not None:
            kernel = Gaussian2DKernel(self.smooth)
            self.image = convolve_fft(self.image, kernel)
        # Re-instate NaNs if required
        if preserve_nan:
            self.image[nan_mask] = np.nan
        # Remove the margin from the image
        self.image = self.image[self.trim]

    def _find_curvatures(self):
        """Calculate principal curvatures and principal directions"""
        # Calculate the gradient
        self.grad_y, self.grad_x = np.gradient(self.image)
        # Optionally scale the gradient (for instance by the RMS
        # variation of the image)
        self.grad_x /= self.value_scale / self.length_scale
        self.grad_y /= self.value_scale / self.length_scale
        # Calculate the second derivatives
        self.grad_xy, self.grad_xx = np.gradient(self.grad_x)
        self.grad_yy, self.grad_yx = np.gradient(self.grad_y)
        self.grad_xx *= self.length_scale
        self.grad_yy *= self.length_scale
        self.grad_xy *= self.length_scale
        self.grad_yx *= self.length_scale
        # The following fails! Probably from edge effects
        # assert np.all(self.grad_yx == self.grad_xy), "Gradient is not symmetric"

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
        self.theta1 = np.rad2deg(np.arctan2(self.kappa1 - self.grad_xx, self.grad_xy))
        self.theta2 = np.rad2deg(np.arctan2(self.kappa2 - self.grad_xx, self.grad_xy))
