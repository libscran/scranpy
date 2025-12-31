from typing import Any, Sequence, Union, Optional

import numpy
import mattress
import biocutils

from . import lib_scranpy as lib


def aggregate_across_genes(
    x: Any,
    sets: Sequence,
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
            Sequence of sequences containing strings or integers.
            Each sequence corresponds to a gene set and contains the row indices or names of the genes in that set.
            If any strings are present, ``row_names`` should be supplied.

            Alternatively, each entry may be a tuple of length 2, containing a sequence of strings/integers (row names/indices) and a numeric array (weights).
            If this is a :py:class:`~biocutils.NamedList.NamedList`, the names will be preserved in the output.

        row_names:
            Sequence of strings of length equal to the number of rows of ``x``, containing the name of each gene.
            If ``None``, rows are assumed to be unnamed.

        average:
            Whether to compute the average rather than the sum.

        num_threads: 
            Number of threads to be used for aggregation.

    Returns:
        List of length equal to that of ``sets``.
        Each entry is a numeric vector of length equal to the number of columns in ``x``,
        containing the (weighted) sum/mean of expression values for the corresponding set across all cells.

    References:
        The ``aggregate_across_genes`` function in the `scran_aggregate <https://libscran.github.io/scran_aggregate>`_ C++ library, which implements the aggregation.
    """ 
    new_sets = [] 
    mapping = {}
    for s in sets:
        if isinstance(s, tuple):
            new_sets.append((
                _check_for_strings(s[0], mapping, row_names),
                numpy.array(s[1], copy=None, order="A", dtype=numpy.float64),
            ))
        else:
            new_sets.append(_check_for_strings(s, mapping, row_names))

    mat = mattress.initialize(x)
    output = lib.aggregate_across_genes(
        mat.ptr,
        new_sets,
        average,
        num_threads
    )

    names = None
    if isinstance(sets, biocutils.NamedList):
        names = sets.get_names()
    return biocutils.NamedList(output, names)


def _check_for_strings(y: Sequence, mapping: dict, row_names: Optional[Sequence]):
    has_str = False
    for x in y:
        if isinstance(x, str):
            has_str = True
            break

    if not has_str:
        return numpy.array(y, copy=None, order="A", dtype=numpy.uint32)

    if "realized" not in mapping:
        if row_names is None:
            raise ValueError("no 'row_names' supplied for mapping gene sets with names")
        found = {}
        for i, s in enumerate(row_names):
            if s not in found:
                found[s] = i
        mapping["realized"] = found
    else: 
        found = mapping["realized"]

    output = numpy.ndarray(len(y), dtype=numpy.uint32)
    for i, x in enumerate(y):
        if isinstance(x, str):
            output[i] = found[x]
        else:
            output[i] = x

    return output
