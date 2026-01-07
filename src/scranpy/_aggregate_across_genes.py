from typing import Any, Sequence, Union, Optional

import numpy
import mattress
import biocframe
import biocutils

from . import _utils_general as gutils
from . import _lib_scranpy as lib


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
              Strings not present in ``row_names`` are ignored.
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
            new_sets.append(_sanitize_gene_set(s[0], mapping, row_names, NR, weights=s[1]))
        else:
            new_sets.append(_sanitize_gene_set(s, mapping, row_names, NR, weights=None))

    mat = mattress.initialize(x)
    output = lib.aggregate_across_genes(
        mat.ptr,
        new_sets,
        average,
        num_threads
    )

    return biocutils.NamedList(output, sets.get_names())


def _sanitize_gene_set(y: Sequence, mapping: dict, row_names: Optional[Sequence], nrow: int, weights: Optional[numpy.ndarray]):
    def _create_output(i, w):
        if w is None:
            return i
        else:
            return i, numpy.array(w, copy=None, order="A", dtype=numpy.float64)

    if isinstance(y, range):
        return _create_output(numpy.array(y, dtype=numpy.uint32), weights)
    if isinstance(y, slice):
        y = y.indices(nrow)
        return _create_output(numpy.array(range(*y), dtype=numpy.uint32), weights)

    if isinstance(y, numpy.ndarray):
        if numpy.issubdtype(y.dtype, numpy.bool):
            if len(y) != nrow:
                raise ValueError("length of a boolean gene set is not equal to the number of rows")
            if weights is not None:
                raise ValueError("weights are not supported for a boolean gene set")
            return numpy.where(y)[0].astype(numpy.uint32)

        if numpy.issubdtype(y.dtype, numpy.integer):
            return _create_output(y.astype(numpy.uint32, copy=False), weights)
        elif numpy.issubdtype(y.dtype, numpy.str_):
            if "realized" not in mapping:
                mapping["realized"] = gutils.create_row_names_mapping(row_names, nrow)
            found = mapping["realized"]

            if weights is None:
                collected = []
                for ss in y:
                    if ss in found:
                        collected.append(found[ss])
                return numpy.array(collected, dtype=numpy.uint32)
            else:
                collected_idx = []
                collected_wts = []
                for i, x in enumerate(y):
                    if x not in found:
                        continue
                    collected_idx.append(found[x])
                    collected_wts.append(weights[i])
                return numpy.array(collected_idx, dtype=numpy.uint32), numpy.array(collected_wts, dtype=numpy.float64)
        else:
            raise TypeError("'dtype' of the gene set should either be bool, integer or string")

    has_bool = False
    has_str = False
    has_int = False
    for ss in y:
        if isinstance(ss, bool) or isinstance(ss, numpy.bool):
            has_bool = True
        elif isinstance(ss, str) or isinstance(ss, numpy.str_):
            has_str = True
        elif isinstance(ss, int) or isinstance(ss, numpy.integer):
            has_int = True
        else:
            raise TypeError("unknown type " + str(type(ss)) + " in a gene set")

    if has_bool:
        if has_str or has_int:
            raise TypeError("gene set defined by booleans should only contain booleans")
        if weights is not None:
            raise ValueError("weights are not supported for a boolean gene set")
        if len(y) != nrow:
            raise ValueError("length of a boolean gene set is not equal to number of rows")
        return numpy.where(y)[0].astype(numpy.uint32)

    if not has_str:
        return _create_output(numpy.array(y, dtype=numpy.uint32), weights)

    if "realized" not in mapping:
        mapping["realized"] = gutils.create_row_names_mapping(row_names, nrow)
    found = mapping["realized"]

    if weights is not None:
        collected_idx = []
        collected_wts = []
        for i, x in enumerate(y):
            if isinstance(ss, str) or isinstance(ss, numpy.str_):
                if x not in found:
                    continue
                x = found[x]
            collected_idx.append(x)
            collected_wts.append(weights[i])
        return numpy.array(collected_idx, dtype=numpy.uint32), numpy.array(collected_wts, dtype=numpy.float64)
    else:
        collected = []
        for i, x in enumerate(y):
            if isinstance(ss, str) or isinstance(ss, numpy.str_):
                if x not in found:
                    continue
                x = found[x]
            collected.append(x)
        return numpy.array(collected, dtype=numpy.uint32)
