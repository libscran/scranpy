from typing import Optional, Sequence, Union

import delayedarray
import summarizedexperiment
import singlecellexperiment

from . import _utils_se as seutils
from ._run_pca import *


def run_pca_se(
    x: summarizedexperiment.SummarizedExperiment,
    features: Optional[Sequence],
    number: int = 25,
    block: Optional[Sequence] = None,
    num_threads: int = 1,
    more_pca_args: dict = {},
    assay_type: Union[int, str] = "logcounts",
    output_name: str = "PCA",
    meta_name: Optional[str] = "PCA"
) -> singlecellexperiment.SingleCellExperiment:
    """
    Compact and denoise the dataset by performing PCA on the (log-)normalized expression matrix.
    This calls :py:func:`~scranpy.run_pca` on an assay of a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

    Args:
        x:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment` object or one of its subclasses.
            Rows correspond to genomic features and columns correspond to cells.

        features:
            Features of interest to use in the PCA.
            This can be a sequence of row indices, row names, or booleans indicating the rows of ``x`` to use.
            For RNA data, this is typically the ``hvg`` vector added to the row data of ``x`` by :py:func:`~scranpy.choose_rna_hvgs_se`.
            If ``None``, all available features are used.

        number:
            Number of PCs to retain, passed to :py:func:`~scranpy.run_pca`. 

        block:
            Block assignment of each cell, passed to :py:func:`~scranpy.run_pca`. 

        num_threads:
            Number of threads, passed to :py:func:`~scranpy.run_pca`. 

        more_pca_args:
            Additional arguments passed to :py:func:`~scranpy.run_pca`. 

        assay_type:
            Name or index of the assay of ``x`` to be used for PCA.
            This is typically the log-normalized expression matrix created by :py:func:`~scranpy.normalize_rna_counts_se` or equivalent.

        output_name:
            Name of the reduced dimension entry in which to store the PC scores in the output object.

        meta_name:
            Name of the metadata entry in which to store other PCA statistics in the output object.

    Returns:
        A copy of ``x`` with the newly computed principal component scores in the reduced dimensions.
        Additional outputs (e.g., rotation matrix, variance explained) are stored in the metadata.

    Examples:
        >>> import scranpy
        >>> sce = scranpy.get_test_rna_data_se("hvg")
        >>> sce = scranpy.run_pca_se(sce, features=sce.get_row_data()["hvg"])
        >>> sce.get_reduced_dimension("PCA").shape
        >>> pcameta = sce.get_metadata()["PCA"]
        >>> pcameta["variance_explained"] / pcameta["total_variance"]
    """

    y = x.get_assay(assay_type)
    if features is not None:
        y = delayedarray.DelayedArray(y)[features,:] # ensure that no copy is made.

    out = run_pca(y, number=number, block=block, num_threads=num_threads, **more_pca_args)

    if not isinstance(x, singlecellexperiment.SingleCellExperiment):
        x = singlecellexperiment.SingleCellExperiment(
            assays=x.get_assays(),
            column_data=x.get_column_data(),
            column_names=x.get_column_names(),
            row_data=x.get_row_data(),
            row_names=x.get_row_names(),
            metadata=x.get_metadata()
        )

    x = seutils.add_transposed_reddim(x, output_name, out["components"])

    if meta_name is not None:
        del out["components"] # already stored in the reduced dimensions.
        new_meta = x.get_metadata().set_value(meta_name, out)
        x = x.set_metadata(new_meta)

    return x
