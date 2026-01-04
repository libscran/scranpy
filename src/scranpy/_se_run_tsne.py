from typing import Optional, Sequence, Union

import delayedarray
import singlecellexperiment

from . import _utils_se as seutils
from ._run_tsne import *


def run_tsne_se( 
    x: singlecellexperiment.SingleCellExperiment,
    perplexity: float = 30,
    num_threads: int = 1,
    more_tsne_args: dict = {},
    reddim_type: Union[int, str] = "PCA", 
    output_name: str = "TSNE"
) -> singlecellexperiment.SingleCellExperiment:
    """
    Generate a t-SNE visualization from an existing embedding.
    This calls :py:func:`~scranpy.run_tsne.run_tsne` on a reduced dimension entry in a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Args:
        x:
            A :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` object or one of its subclasses.
            Rows correspond to genomic features and columns correspond to cells.

        perplexity:
            Perplexity for the t-SNE algorithm, passed to :py:func:`~scranpy.run_tsne.run_tsne`.

        num_threads:
            Number of threads for the neighbor search and optimization, passed to :py:func:`~scranpy.run_tsne.run_tsne`.

        more_tsne_args:
            Additional arguments to pass to :py:func:`~scranpy.run_tsne.run_tsne`.

        reddim_type: 
            Name or index of the existing reduced dimension embedding in ``x`` from which to generate t-SNE coordinates.
            Alternatively a tuple, where the first element contains the name of an alternative experiment of ``x``,
            and the second element contains the name/index of an embedding in that alternative experiment.

        output_name:
            Name of the reduced dimension entry in which to store the t-SNE coordinates in the output object.

    Returns:
        A copy of ``x`` with a new reduced dimension entry containing the t-SNE coordinates. 

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("pca")
        >>> # Using fewer iterations for a faster-running example.
        >>> sce = scranpy.run_tsne_se(sce, more_tsne_args={ "max_iterations": 50 })
        >>> sce.get_reduced_dimension("TSNE")[:5,:]
    """

    out  = run_tsne(
        seutils.get_transposed_reddim(x, reddim_type),
        perplexity=perplexity,
        num_threads=num_threads,
        **more_tsne_args
    )

    return _add_tsne_results(x, output_name, out)


def _add_tsne_results(
    x: singlecellexperiment.SingleCellExperiment,
    output_name: str,
    res: numpy.ndarray
) -> singlecellexperiment.SingleCellExperiment:
    return x.set_reduced_dimension(output_name, res)
