"""Pupil-pair response and all-order Gaussian remainder quadrature."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import qmc

from ._sobol_remainder import (
    decoded_quartet_functions,
    pair_search_tables,
    quartet_accumulator,
)

SOBOL_POWER = 20
SOBOL_REPLICATES = 8
SOBOL_SEED = 20261105
SOBOL_BLOCK_SIZE = 65536
SOBOL_INDEX_CACHE_BYTES = 128 * 1024**2


def quadratic_response(weights, *, basis=None):
    """Form first- and second-order metric kernels in pupil or modal coordinates."""
    row_sum, col_sum = weights.sum(axis=2), weights.sum(axis=1)
    first = (1j * (row_sum - col_sum)).real
    second = weights.real + weights.real.swapaxes(1, 2)
    diagonal = np.arange(weights.shape[1])
    second[:, diagonal, diagonal] -= (row_sum + col_sum).real
    if basis is not None:
        first = first @ basis.T
        second = basis @ second @ basis.T
    return first, second


def quadratic_covariance(covariance, first, second):
    """Contract the first two Gaussian orders at one or several time lags."""
    transformed = second @ covariance[..., None, :, :]
    return np.einsum("mi,...ij,mj->...m", first, covariance, first) + 0.5 * np.einsum(
        "...mij,...mij->...m", transformed, covariance[..., None, :, :] @ second
    )


def quadratic_variance(covariance, weights, *, basis=None):
    """Contract the first two Gaussian orders at zero lag."""
    return quadratic_covariance(covariance, *quadratic_response(weights, basis=basis))


def restored_covariance(coherence, amplitude):
    """Recover the piston-free covariance from the full Gaussian coherence."""
    ratio = np.abs(coherence) / amplitude
    if not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError(
            "Cannot recover a finite phase structure function from the mean"
        )
    structure = -2 * np.log(ratio)
    covariance = -0.5 * (
        structure
        - structure.mean(0)[None]
        - structure.mean(1)[:, None]
        + structure.mean()
    )
    values = np.linalg.eigvalsh(covariance)
    if values.min() < -1e-9 * max(1, values.max()):
        raise ValueError(
            f"Restored pupil covariance is not positive semidefinite: "
            f"min={values.min():.6g}, max={values.max():.6g}, "
            f"negative trace fraction={-values[values < 0].sum() / values.sum():.6g}"
        )
    return covariance


class RemainderQuadrature:
    """Reuse an equal-time importance proposal at every covariance lag.

    Pair-reversal symmetry leaves real weights multiplying cosh(K)-1-K²/2
    and imaginary weights multiplying sinh(K)-K. Multiplying each absolute
    kernel by exp(D/2) removes mean-coherence suppression from the proposal.
    Cauchy's inequality bounds the quartet exponential by exp((Dab+Dcd)/2),
    so this proposal covers distant pairs that dominate short-frame variance.
    All metrics and covariance variants share the same pupil quartets.
    Independent scrambles estimate numerical error. The first quarter of
    each net provides a matched resolution check.
    """

    def __init__(self, covariances, weights):
        a, b = np.triu_indices(weights.shape[2], 1)
        pair_weights = weights[:, :, a, b]
        structure = (
            np.diagonal(covariances, axis1=1, axis2=2)[:, a]
            + np.diagonal(covariances, axis1=1, axis2=2)[:, b]
            - 2 * covariances[:, a, b]
        )
        absolute = np.abs(pair_weights) * np.exp(0.5 * structure)[:, None]
        probability = np.mean(absolute / absolute.sum(axis=2)[:, :, None], axis=(0, 1))
        probability /= probability.sum()
        matrix = np.zeros(covariances.shape[1:])
        matrix[a, b] = probability
        marginal = matrix.sum(axis=1)
        conditional = np.divide(
            matrix,
            marginal[:, None],
            out=np.zeros_like(matrix),
            where=marginal[:, None] > 0,
        )
        conditional_cdf = conditional.cumsum(axis=1)
        conditional_cdf[:, -1] = 1
        marginal_cdf = marginal.cumsum()
        marginal_cdf[-1] = 1
        search = pair_search_tables(marginal_cdf, conditional_cdf)
        self.proposal = matrix
        self.search = search
        self._decoded = None

    @property
    def metal_indices(self):
        """Return a complete set of decoded draws eligible for the Metal kernel."""
        if self._decoded is not None:
            indices = self._decoded[1]
            if all(
                item is not None and item.dtype == np.uint16 and len(item) >= 128
                for item in indices
            ):
                return tuple(indices)
        return None

    def evaluate(
        self,
        covariances,
        weights,
        power,
        replicates,
        seed,
        *,
        workers=1,
        reuse_draws=False,
    ):
        """Use matched scrambled quartets at a single lag, returning fine and coarse estimates.

        See the Sobol notes for the linear matrix scramble and digital
        shift used for each replicate. [SciPy Sobol documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)
        """
        accumulate = quartet_accumulator()
        matrix, search = self.proposal, self.search
        count = 1 << power
        coarse_count = count // 4
        dtype = np.uint16 if len(matrix) <= 65536 else np.uint32
        cache = None
        if (
            reuse_draws
            and replicates * count * 4 * np.dtype(dtype).itemsize
            <= SOBOL_INDEX_CACHE_BYTES
        ):
            key = power, replicates, seed, SOBOL_BLOCK_SIZE
            if self._decoded is None or self._decoded[0] != key:
                self._decoded = (key, [None] * replicates)
            cache = self._decoded[1]
            decode, sum_indices = decoded_quartet_functions()

        def integrate(replicate):
            indices = None if cache is None else cache[replicate]
            if indices is None:
                net = qmc.Sobol(
                    4,
                    scramble=True,
                    seed=np.random.default_rng(
                        np.random.SeedSequence([seed, replicate])
                    ),
                )
                if cache is not None:
                    indices = np.empty((count, 4), dtype=dtype)
            estimates = np.zeros(weights.shape[:2])
            coarse = np.zeros_like(estimates)
            # Consecutive blocks form the same complete power-of-two net. Split
            # at the coarse endpoint without holding a full scramble in memory.
            for lower, upper in ((0, coarse_count), (coarse_count, count)):
                for start in range(lower, upper, SOBOL_BLOCK_SIZE):
                    stop = min(start + SOBOL_BLOCK_SIZE, upper)
                    if cache is None:
                        subtotal = accumulate(
                            covariances,
                            weights,
                            matrix,
                            *search,
                            net.random(stop - start),
                        )
                    else:
                        block = indices[start:stop]
                        if cache[replicate] is None:
                            decode(*search, net.random(stop - start), block)
                        subtotal = sum_indices(covariances, weights, matrix, block)
                    estimates += subtotal
                    if lower == 0:
                        coarse += subtotal
            if cache is not None:
                cache[replicate] = indices
            return estimates / count, coarse / coarse_count

        if workers == 1:
            results = list(map(integrate, range(replicates)))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(integrate, range(replicates)))
        estimates, coarse = zip(*results, strict=True)
        return np.array(estimates), np.array(coarse)


def remainder_draws(covariances, weights, power, replicates, seed, *, workers=1):
    """Integrate orders three and above at zero lag."""
    return RemainderQuadrature(covariances, weights).evaluate(
        covariances, weights, power, replicates, seed, workers=workers
    )


def integrate_all_orders(responses, power, replicates, seed, *, workers=1):
    """Integrate instantaneous variance with shared draws for all responses."""
    _, covariances, weights, base = (
        np.asarray(values) for values in zip(*responses, strict=True)
    )
    remainders, coarse = remainder_draws(
        covariances, weights, power, replicates, seed, workers=workers
    )
    variance = base + remainders.mean(axis=0)
    coarse_variance = base + coarse.mean(axis=0)
    if np.any(variance <= 0) or np.any(coarse_variance <= 0):
        raise ValueError("Nonpositive instantaneous metric variance")
    return variance, base, remainders, coarse_variance
