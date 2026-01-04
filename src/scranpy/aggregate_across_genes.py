from typing import Any, Sequence, Union, Optional

import numpy
import mattress
import biocframe
import biocutils

from . import _utils_general as gutils
from . import lib_scranpy as lib


def aggregate_across_genes(
    x: Any,
    sets: Union[dict, Sequence, biocutils.NamedList],
    row_names: Optional[Sequence] = None,
    average: bool = False,
    num_threads: int = 1
) -> biocutils.NamedList:
    """Aggregate expression values across genes, potentially with weights.
    This is typically used to summarize expression values for gene sets into a single per-cell score.

    Args:
        x:
            Matrix-like object where rows correspond to genes or genomic features and columns correspond to cells. 
            Values are expected to be log-expression values.

        sets:
            Sequence of gene sets.
            Each gene set may be represented by:

            - A sequence of integers, specifying the row indices of the genes in that set.
            - A sequence of strings, specifying the row names of the genes in that set.
              If any strings are present, ``row_names`` should also be supplied.
            - A tuple of length 2, containing a sequence of strings/integers (row names/indices) and a numeric array (weights).
            - A :py:class:`~biocframe.BiocFrame.BiocFrame` where each row corresponds to a gene.
              The first column contains the row names/indices and the second column contains the weights.

            Alternatively, a dictionary may be supplied where each key is the name of a gene set and each value is a sequence/tuple as described above.
            The keys will be used to name the output ``NamedList``.

            Alternatively, a :py:class:`~biocutils.NamedList.NamedList` where each entry is a gene set represented by a sequence/tuple as described above.
            If names are available, they will be used to name the output ``NamedList``.

        row_names:
            Sequence of strings of length equal to the number of rows of ``x``, containing the name of each gene.
            Duplicate names are allowed but only the first occurrence will be used.
            If ``None``, rows are assumed to be unnamed.

        average:
            Whether to compute the average rather than the sum.

        num_threads: 
            Number of threads to be used for aggregation.

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` of length equal to that of ``sets``.
        Each entry is a numeric vector of length equal to the number of columns in ``x``,
        containing the (weighted) sum/mean of expression values for the corresponding set across all cells.

    References:
        The ``aggregate_across_genes`` function in the `scran_aggregate <https://libscran.github.io/scran_aggregate>`_ C++ library, which implements the aggregation.

    Examples:
        >>> import numpy
        >>> mat = numpy.random.rand(100, 20)
        >>> import scranpy
        >>> sets = {
        >>>     "foo": [ 1, 3, 5, 7, 9 ],
        >>>     "bar": range(10, 40, 2)
        >>> } 
        >>> aggr = scranpy.aggregate_across_genes(mat, sets)
        >>> print(aggr.get_names())
        >>> print(aggr[0])
    """

    sets = gutils.to_NamedList(sets)

    new_sets = [] 
    mapping = {}
    NR = x.shape[0]

    for s in sets:
        if isinstance(s, tuple) or isinstance(s, biocframe.BiocFrame):
            new_sets.append((
                _check_for_strings(s[0], mapping, row_names, NR),
                numpy.array(s[1], copy=None, order="A", dtype=numpy.float64),
            ))
        else:
            new_sets.append(_check_for_strings(s, mapping, row_names, NR))

    mat = mattress.initialize(x)
    output = lib.aggregate_across_genes(
        mat.ptr,
        new_sets,
        average,
        num_threads
    )

    return biocutils.NamedList(output, sets.get_names())


def _check_for_strings(y: Sequence, mapping: dict, row_names: Optional[Sequence], nrow: int) -> numpy.ndarray:
    has_str = False
    for x in y:
        if isinstance(x, str):
            has_str = True
            break

    if not has_str:
        return numpy.array(y, copy=None, order="A", dtype=numpy.uint32)

    if "realized" not in mapping:
        mapping["realized"] = gutils.create_row_names_mapping(row_names, nrow)
    found = mapping["realized"]

    output = numpy.ndarray(len(y), dtype=numpy.uint32)
    for i, x in enumerate(y):
        if isinstance(x, str):
            output[i] = found[x]
        else:
            output[i] = x

    return output
