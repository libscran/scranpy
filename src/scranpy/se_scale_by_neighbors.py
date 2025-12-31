from typing import Sequence, Union, Optional

import knncolle
import biocutils
import singlecellexperiment

from .scale_by_neighbors import *
from . import _utils_se as seutils


def scale_by_neighbors_se(
    x: singlecellexperiment.SingleCellExperiment,
    altexp_reddims: Union[dict, biocutils.NamedList],
    main_reddims: Union[int, str, Sequence] = "PCA", 
    num_neighbors: int = 20,
    block: Optional[Sequence] = None,
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters(),
    num_threads: int = 1,
    more_scale_args: dict = {},
    output_name: str = "combined",
    meta_name: Optional[str] = "combined"
) -> singlecellexperiment.SingleCellExperiment:
    """
    Scale embeddings for different modalities to equalize their intra-population variance, and then combine them into a single embedding for downstream analysis.
    This calls :py:func:`~scranpy.scale_by_neighbors.scale_by_neighbors` on the main/alternative experiments of a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Args:
        x:
            A :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` object or one of its subclasses.
            Rows correspond to genomic features and columns correspond to cells.

        altexp_reddims:
            Dictionary specifying the reduced dimensions of the alternative experiments of ``x`` to be combined.
            Specifically, each key is the name of an alternative experiment and the value is a list of strings/integers specifying the names/indices of the reduced dimensions of that experiment.

            Alternatively, each value may be a single string or integer that will be converted into a list of length 1.
            Each value may also be ``None`` in which case the corresponding alternative experiment is ignored.

            Alternatively, a :py:class:`~biocutils.NamedList.NamedList` where each element is named after one or more alternative experiments,
            and each value is as described above for a dictionary.

        main_reddims:
            List of strings/integers specifying the names/indices of the reduced dimensions from the main experiment of ``x``.

            Alternatively, a single string or integer that will be converted into a list of length 1.

            Alternatively ``None``, which will be treated as a 

        num_neighbors:
            Number of neighbors to compute the scaling, see :py:func:`~scranpy.scale_by_neighbors.scale_by_neighbors`. 

        block:
            Block assignment for each cell, passed to :py:func:`~scranpy.scale_by_neighbors.scale_by_neighbors`. 

        nn_parameters:
            Algorithm for the nearest neighbor search, passed to :py:func:`~scranpy.scale_by_neighbors.scale_by_neighbors`. 

        num_threads:
            Number of threads, passed to :py:func:`~scranpy.scale_by_neighbors.scale_by_neighbors`. 

        more_scale_args:
            Additional arguments to pass to :py:func:`~scranpy.scale_by_neighbors.scale_by_neighbors`. 

        output_name:
            Name of the reduced dimension entry in which to store the combined embeddings in the output object.

        meta_name:
            Name of the metadata entry in which to store additional metrics in the output object.
            If ``None``, additional metrics are not stored.

    Returns:
        A copy of ``x`` with the combined embeddings stored in its row data.
        The scaling factors for all embeddings are stored in the metadata.

    Examples:
        >>> import scranpy 
        >>> sce = scranpy.get_test_adt_data_se("pca")
        >>> sce = scranpy.scale_by_neighbors_se(sce, altexp_reddims={ "ADT": "PCA" })
        >>> sce.get_reduced_dimension_names()
        >>> sce.get_metadata()["combined"]
    """

    all_embeddings = []
    main_reddims = _sanitize_reddims(main_reddims, x.get_reduced_dimension_names())
    for r in main_reddims:
        all_embeddings.append(seutils.get_transposed_reddim(x, r))

    altexp_reddims = seutils.to_NamedList(altexp_reddims)
    if altexp_reddims.get_names() is None:
        raise ValueError("'altexp_reddims' should be a dictionary or a NamedList") 

    ae_names = []
    ae_sanitized = set()
    for ae in altexp_reddims.get_names():
        if ae in ae_sanitized:
            continue
        ae_se = x.get_alternative_experiment(ae)
        cur_reddim = _sanitize_reddims(altexp_reddims[ae], ae_se.get_reduced_dimension_names())
        for r in cur_reddim:
            all_embeddings.append(seutils.get_transposed_reddim(ae_se, r))
        ae_names.append((ae, cur_reddim))
        ae_sanitized.add(ae)

    out = scale_by_neighbors(
        all_embeddings,
        num_neighbors=num_neighbors,
        block=block,
        num_threads=num_threads,
        nn_parameters=nn_parameters,
        **more_scale_args
    )

    x = seutils.add_transposed_reddim(x, output_name, out["combined"])

    if meta_name is not None:
        # Formatting it in the same manner as the arguments.
        counter = 0
        main_scaling = biocutils.FloatList()
        for r in main_reddims:
            main_scaling[r] = out["scaling"][counter]
            counter += 1

        altexp_scaling = biocutils.NamedList()
        for (ae, ae_reddims) in ae_names:
            current_scaling = biocutils.FloatList()
            for rd in ae_reddims:
                current_scaling[rd] = out["scaling"][counter]
                counter += 1
            altexp_scaling[ae] = current_scaling

        import copy
        meta = copy.copy(x.get_metadata())
        meta[meta_name] = { "main_scaling": main_scaling, "altexp_scaling": altexp_scaling }
        x = x.set_metadata(meta)

    return x


def _sanitize_reddims(reddims: Sequence, all_reddims: Sequence) -> Sequence:
    if reddims is None:
        return []
    elif isinstance(reddims, str):
        return [reddims]
    elif isinstance(reddims, int):
        return [all_reddims[reddims]]

    present = set()
    collected = []
    for r in reddims:
        if isinstance(r, int):
            r = all_reddims[r]
        if r in present:
            continue
        present.add(r)
        collected.append(r)
    return collected
