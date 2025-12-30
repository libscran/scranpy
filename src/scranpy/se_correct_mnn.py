from typing import Sequence, Union

import singlecellexperiment
import knncolle

from . import _utils_se as seutils
from .correct_mnn import *


def correct_mnn_se(
    x: singlecellexperiment.SingleCellExperiment,
    block: Sequence,
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters(),
    num_threads: int = 1,
    more_mnn_args: dict = {},
    reddim_type: Union[str, int, tuple] = "PCA", 
    output_name: str = "MNN"
) -> singlecellexperiment.SingleCellExperiment:
    """
    Correct batch effects from an existing embedding with mutual nearest neighbors (MNNs).
    This calls :py:func:`~scranpy.correct_mnn.correct_mnn` on reduced dimensions from a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Args:
        x:
            A :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` object or one of its subclasses.
            Rows correspond to genomic features and columns correspond to cells.

        block: 
            Block assignment of each cell, see :py:func:`~scranpy.correct_mnn.correct_mnn` for more details.

        nn_parameters:
            Algorithm for nearest neighbor search, see :py:func:`~scranpy.correct_mnn.correct_mnn` for more details.

        num_threads:
            Number of threads, see :py:func:`~scranpy.correct_mnn.correct_mnn` for more details.

        more_mnn_args:
            Additional arguments to pass to :py:func:`~scranpy.correct_mnn.correct_mnn`.

        reddim_type:
            Name or index of an existing embedding in ``x``, on which to perform the batch correction.
            Alternatively a tuple, where the first element contains the name of an alternative experiment of ``x``,
            and the second element contains the name/index of an embedding in that alternative experiment.

        output_name:
            Name of the reduced dimension entry which to store the corrected coordinates.

    Returns:
        A copy of ``x`` , with the corrected embedding stored as a reduced dimension entry.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("pca")
        >>> # Treating the tissue of origin as the batch.
        >>> sce = scranpy.correct_mnn_se(sce, block=sce.get_column_data()["tissue"])
        >>> sce.get_reduced_dimension_names()
    """

    out = correct_mnn(
        seutils.get_transposed_reddim(x, reddim_type),
        block=block,
        num_threads=num_threads,
        nn_parameters=nn_parameters,
        **more_mnn_args
    )

    return seutils.add_transposed_reddim(x, output_name, out["corrected"])

