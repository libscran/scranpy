import numpy

from . import lib_scranpy as lib


def choose_pseudo_count(
    size_factors: numpy.ndarray,
    quantile: float = 0.05, 
    max_bias: float = 1,
    min_value: float = 1
) -> float:
    """Choose a suitable pseudo-count to control the bias introduced by log-transformation of normalized counts.

    Args:
        size_factors:
            Floating-point array of size factors for all cells.

        quantile:
            Quantile to use for defining extreme size factors.

        max_bias:
            Maximum allowed bias in the log-fold changes between cells.

        min_value: 
            Minimum value for the pseudo-count.

    Returns:
        Choice of pseudo-count, for use in :py:func:`~scranpy.normalize_counts.normalize_counts`.

    References:
        The ``choose_pseudo_count`` function in the `scran_norm <https://libscran.github.io/scran_norm>`_ C++ library, which describes the rationale behind the choice of pseudo-count.
    """
    local_sf = numpy.array(size_factors, dtype=numpy.float64, copy=None)
    return lib.choose_pseudo_count(local_sf, quantile, max_bias, min_value)
