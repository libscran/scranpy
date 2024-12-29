from typing import Any, Sequence, Optional, Literal
from dataclasses import dataclass

import numpy
import biocutils
import knncolle

from . import lib_scranpy as lib


@dataclass
class CorrectMnnResults:
    """Results of :py:func:`~correct_mnn`."""

    corrected: numpy.ndarray
    """Floating-point matrix of the same dimensions as the ``x`` used
    in :py:func:`~correct_mnn`, containing the corrected values."""

    merge_order: list[str]
    """Merge order for the levels of the blocking factor. The first level in
    this vector is used as the reference batch; all other batches are
    iteratively merged and added to the reference."""

    num_pairs: numpy.ndarray
    """Integer vector of length equal to the number of batches minus 1.
    This contains the number of MNN pairs at each merge step."""


def correct_mnn(
    x: numpy.ndarray,
    block: Sequence,
    num_neighbors: int = 15,
    num_mads: int = 3,
    robust_iterations: int = 2,
    robust_trim: float = 0.25,
    mass_cap: Optional[int] = None,
    order: Optional[Sequence] = None,
    reference_policy: Literal["max-rss", "max-size", "max-variance", "input"] = "max-rss",
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters(),
    num_threads: int = 1
) -> CorrectMnnResults:
    """Apply mutual nearest neighbor (MNN) correction to remove batch effects
    from a low-dimensional matrix.

    Args:
        x:
            Matrix of coordinates where rows are dimensions and columns are
            cells, typically generated by :py:func:`~scranpy.run_pca.run_pca`.

        block:
            Factor specifying the block of origin (e.g., batch, sample) for
            each cell. Length should equal the number of columns in ``x``.

        num_neighbors:
            Number of neighbors to use when identifying MNN pairs.

        num_mads:
            Number of median absolute deviations to use for removing outliers
            in the center-of-mass calculations.

        robust_iterations:
            Number of iterations for robust calculation of the center of mass.

        robust_trim:
            Trimming proportion for robust calculation of the center of mass.
            This should be a value in [0, 1).

        mass_cap:
            Cap on the number of observations to use for center-of-mass
            calculations on the reference dataset. A value of 100,000 may be
            appropriate for speeding up correction of very large datasets.
            If None, no cap is used.

        order:
            Sequence containing the unique levels of ``block`` in the desired
            merge order If None, a suitable merge order is automatically
            determined.

        reference_policy:
            Policy to use to choose the first reference batch.  This can be
            based on the largest batch (``max-size``), the most variable batch
            (``max-variance``), the batch with the largest residual sum of
            squares (``max-rss``), or the first specified input (``input``).
            Only used for automatic merges, i.e., when ``order = None``.

        nn_parameters:
            The nearest-neighbor algorithm to use.

        num_threads:
            Number of threads to use.

    Returns:
        The results of the MNN correction, including a matrix of the corrected
        coordinates and some additional diagnostics.
    """
    blocklev, blockind = biocutils.factorize(block, fail_missing=True, dtype=numpy.uint32)

    if not order is None:
        order = biocutils.match(order, blocklev, dtype=numpy.uint32)
        if sorted(list(order)) != list(range(len(blocklev))):
            raise ValueError("'order' should contain unique values in 'block'"); 

    if mass_cap is None:
        mass_cap = -1

    builder, _ = knncolle.define_builder(nn_parameters)
    corrected, merge_order, num_pairs = lib.correct_mnn(
        x, 
        blockind, 
        num_neighbors,
        num_mads,
        robust_iterations,
        robust_trim,
        num_threads,
        mass_cap,
        order, 
        reference_policy,
        builder.ptr
    )

    return CorrectMnnResults(
        corrected,
        biocutils.subset_sequence(blocklev, merge_order),
        num_pairs
    )