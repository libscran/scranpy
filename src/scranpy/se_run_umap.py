from typing import Optional, Sequence, Union

import delayedarray
import singlecellexperiment

from . import _utils_se as seutils
from .run_umap import *


def run_umap_se( 
    x: singlecellexperiment.SingleCellExperiment,
    num_dim: int = 2,
    min_dist: float = 0.1,
    num_neighbors: int = 15,
    num_threads: int = 1,
    more_umap_args: dict = {},
    reddim_type: Union[int, str] = "PCA", 
    output_name: str = "UMAP"
) -> singlecellexperiment.SingleCellExperiment:
    """
    Generate a UMAP visualization from an existing embedding.
    This calls :py:func:`~scranpy.run_umap.run_umap` on a reduced dimension entry in a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Args:
        x:
            A :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` object or one of its subclasses.
            Rows correspond to genomic features and columns correspond to cells.

        num_dim:
            Number of dimensions in the UMAP, passed to :py:func:`~scranpy.run_umap.run_umap`. 

        min_dist:
            Minimum distance between cells in the UMAP, passed to :py:func:`~scranpy.run_umap.run_umap`. 

        num_neighbors:
            Number of neighbors, passed to :py:func:`~scranpy.run_umap.run_umap`.

        num_threads:
            Number of threads for the neighbor search and optimization, passed to :py:func:`~scranpy.run_umap.run_umap`.

        more_umap_args:
            Additional arguments to pass to :py:func:`~scranpy.run_umap.run_umap`.

        reddim_type: 
            Name or index of the existing reduced dimension embedding in ``x`` from which to generate UMAP coordinates.
            Alternatively a tuple, where the first element contains the name of an alternative experiment of ``x``,
            and the second element contains the name/index of an embedding in that alternative experiment.

        output_name:
            Name of the reduced dimension entry in which to store the UMAP coordinates in the output object.

    Returns:
        A copy of ``x`` with a new reduced dimension entry containing the UMAP coordinates. 
    """

    out = run_umap(
        seutils.get_transposed_reddim(x, reddim_type),
        num_dim=num_dim,
        min_dist=min_dist,
        num_neighbors=num_neighbors,
        num_threads=num_threads,
        **more_umap_args
    )

    return _add_umap_results(x, output_name, out)


def _add_umap_results(
    x: singlecellexperiment.SingleCellExperiment,
    output_name: str,
    res: numpy.ndarray
) -> singlecellexperiment.SingleCellExperiment:
    return x.set_reduced_dimension(output_name, res)
