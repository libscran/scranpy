from typing import Union

import biocutils
import biocframe
import numpy

from . import lib_scranpy as lib
from . import _utils_se as seutils


def combine_factors(factors: Union[list, dict, biocutils.NamedList], keep_unused: bool = False) -> biocutils.NamedList:
    """Combine multiple categorical factors based on the unique combinations of levels from each factor.

    Args:
        factors:
            List containing factors of interest.
            Each entry corresponds to a factor and should be a sequence of the same length.
            Corresponding elements across all factors represent the combination of levels for a single observation.

            Alternatively, a dictionary where each entry corresponds to a factor.
            Each key is the name of the factor and value is a sequence containing the factor.

            Alternatively, a :py:class:`~biocutils.NamedList.NamedList` where each entry is a factor.
            This may or may not be named.

        keep_unused:
            Whether to report unused combinations of levels.
            If any entry of ``factors`` is a :py:class:`~biocutils.Factor.Factor` object, any unused levels will also be preserved.

    Returns:
        :py:class:`~biocutils.NamedList.NamedList` containing:

        - ``levels``: a :py:func:`~biocframe.BiocFrame.BiocFrame` containing the sorted and unique combinations of levels as a tuple.
          Each column corresponds to a factor in ``factors`` while each row represents a unique combination.
          Corresponding elements of each column define a single combination, i.e., the ``i``-th combination is defined by taking the ``i``-th element of each column.
        - ``index``: an integer NumPy array specifying the index into ``levels`` for each observation.

        For observation ``i`` and factor ``j``, ``levels[j][index[i]]`` will recover ``factors[j][i]``.

    References:
        The ``combine_factors`` function in the `scran_aggregate <https://libscran.github.io/scran_aggregate>`_ library. 

    Examples:
        >>> import scranpy
        >>> import random
        >>> x = random.choices(["A", "B", "C"], k=20)
        >>> y = random.choices([True, False], k = 20)
        >>> combined = scranpy.combine_factors({ "foo": x, "bar":  y })
        >>> print(combined["levels"])
    """

    factors = seutils.to_NamedList(factors)
    facnames = factors.get_names()
    if facnames is None:
        facnames = [str(i) for i in range(len(factors))]

    f0 = []
    levels0 = []
    for current in factors:
        if isinstance(current, biocutils.Factor):
            f0.append(current.get_codes().astype(numpy.uint32, copy=None))
            levels0.append(current.get_levels())
        else:
            lev, ind = biocutils.factorize(current, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
            f0.append(ind)
            levels0.append(lev)

    ind, combos = lib.combine_factors(
        (*f0,),
        keep_unused,
        numpy.array([len(l) for l in levels0], dtype=numpy.uint32)
    )

    new_combinations = {} 
    for f, current in enumerate(combos):
        new_combinations[facnames[f]] = biocutils.subset_sequence(levels0[f], current)

    return biocutils.NamedList.from_dict({
        "levels": biocframe.BiocFrame(new_combinations, column_names=facnames),
        "index": ind
    })
