"""Cubic transfer of optical coherence between pupil lag grids."""

import numpy as np
from scipy import fft, ndimage
from scipy.sparse import csr_matrix


def cubic_stencil(coordinates, side):
    """Evaluate periodic cubic spline coefficients at the given coordinates."""
    coordinates = coordinates.ravel() % side
    start = np.floor(coordinates).astype(int)
    fraction = coordinates - start
    weights = (
        np.stack(
            (
                (1 - fraction) ** 3,
                3 * fraction**3 - 6 * fraction**2 + 4,
                -3 * fraction**3 + 3 * fraction**2 + 3 * fraction + 1,
                fraction**3,
            ),
            axis=1,
        )
        / 6
    )
    columns = (start[:, None] + np.arange(-1, 3)) % side
    return csr_matrix(
        (weights.ravel(), columns.ravel(), np.arange(0, 4 * len(start) + 1, 4)),
        shape=(len(start), side),
    )


class CubicLagTransfer:
    """Move coherence relative to the static pupil onto signed native lags.

    Coarse lag arrays have zero lag at their centre. The native coordinates
    are signed offsets in coarse grid units. Native and coarse factors carry
    the static optics and any stationary phase terms supplied by the caller.
    Both zero-lag values are fixed by flux conservation.
    """

    def __init__(self, coarse_static, native_coordinates):
        self.coarse_static = coarse_static
        self.valid = abs(coarse_static) > 1e-8 * abs(coarse_static).max()
        self.nearest = tuple(
            ndimage.distance_transform_edt(
                ~self.valid, return_distances=False, return_indices=True
            )
        )
        self.coarse_center = len(coarse_static) // 2
        self.native_center = len(native_coordinates) // 2
        self.interpolation = cubic_stencil(native_coordinates, len(coarse_static))

    def forward(self, coarse, native_factor, coarse_factor=None):
        """Restore the native optical transfer function at fixed total flux.

        See §II.C for the B-spline coefficient prefilter and evaluation.
        [Thévenaz et al. (2000)](https://bigwww.epfl.ch/publications/thevenaz0002.pdf)
        """
        ratio = np.divide(
            coarse, self.coarse_static, out=np.zeros_like(coarse), where=self.valid
        )
        ratio = ratio[self.nearest]
        ratio[self.coarse_center, self.coarse_center] = 1
        if coarse_factor is not None:
            ratio *= coarse_factor
        coefficients = ndimage.spline_filter(
            fft.ifftshift(ratio), output=np.complex128, mode="grid-wrap"
        )
        matrix = self.interpolation
        rows = matrix @ coefficients
        native = native_factor * (matrix @ np.ascontiguousarray(rows.T)).T
        center = self.native_center
        native[center, center] = native_factor[center, center]
        return native

    def adjoint(self, native, native_factor, coarse_factor=None):
        """Pull a native lag kernel back under the real bilinear metric pairing."""
        weighted = native * native_factor
        weighted[self.native_center, self.native_center] = 0
        matrix = self.interpolation.T
        rows = matrix @ weighted
        sampled = (matrix @ np.ascontiguousarray(rows.T)).T
        # The periodic spline prefilter is symmetric, so its transpose is itself.
        pulled = fft.fftshift(
            ndimage.spline_filter(sampled, output=np.complex128, mode="grid-wrap")
        )
        if coarse_factor is not None:
            pulled *= coarse_factor
        pulled[self.coarse_center, self.coarse_center] = 0
        coarse = np.zeros_like(self.coarse_static)
        np.add.at(coarse, self.nearest, pulled)
        np.divide(coarse, self.coarse_static, out=coarse, where=self.valid)
        return coarse
