"""Calculate principal curvatures and principal directions from an image"""

import numpy as np
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel


def principal_curvature(image, sigma=None):
    """Calculate principal curvatures and principal directions from an image

    Args:
        image: 2D array
            Image to calculate curvature from
        sigma: float
            Standard deviation of the Gaussian kernel used for smoothing

    Returns:
        kappa1: 2D array
            First principal curvature (higher value)
        kappa2: 2D array
            Second principal curvature (lower value)
        dir1: 2D array
            Principal direction 1
        dir2: 2D array
            Principal direction 2
    """

    # Optionally smooth the image
    if sigma is not None:
        kernel = Gaussian2DKernel(sigma)
        image = convolve_fft(image, kernel)

    # Calculate the gradient
    grad_x, grad_y = np.gradient(image)
    # Calculate the second derivatives
    grad_xx, grad_xy = np.gradient(grad_x)
    grad_yx, grad_yy = np.gradient(grad_y)

    # Calculate the Gaussian and mean curvatures
    kgauss = grad_xx * grad_yy - grad_xy * grad_yx
    kmean = (grad_xx + grad_yy) / 2
    # Calculate the principal curvatures
    kappa1 = kmean + np.sqrt(kmean**2 - kgauss)
    kappa2 = kmean - np.sqrt(kmean**2 - kgauss)

    # Calculate the principal directions
    dir1 = np.arctan2(kappa1 - grad_xx, -grad_xy)
    dir2 = np.arctan2(kappa2 - grad_xx, -grad_xy)

    return kappa1, kappa2, np.rad2deg(dir1), np.rad2deg(dir2)
