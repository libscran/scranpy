from typing import Sequence, Union

import numpy

from . import _lib_scranpy as lib


def _count_overlaps(x: Sequence, sets: Sequence) -> numpy.ndarray:
    overlap = numpy.ndarray((len(sets),), dtype=numpy.uint32)
    xset = set(x)
    for s, curset in enumerate(sets):
        counter = 0
        for g in curset:
            counter += (g in xset)
        overlap[s] = counter
    return overlap


def test_enrichment(
    x: Sequence,
    sets: Union[dict, Sequence],
    universe: Union[int, Sequence],
    log: bool = False,
    num_threads: int = 1
) -> numpy.ndarray:
    """Perform a hypergeometric test for enrichment of interesting genes (e.g., markers) in one or more pre-defined gene sets.

    Args:
        x: 
            Sequence of identifiers for the interesting genes.

        sets:
            Sequence of gene sets, where each entry corresponds to a gene set and contains a sequence of identifiers for genes in that set.

            Alternatively, a dictionary where each key is the name of a gene set and each value is a sequence of identifiers for that gene set.

        universe:
            Sequence of identifiers for the universe of genes in the dataset.
            It is assumed that ``x`` is a subset of ``universe``.
            Identifiers in ``sets`` that are not in ``universe`` will be ignored.

            Alternatively, an integer specifying the number of genes in the universe.

        log:
            Whether to report the log-transformed p-values.

        num_threads: 
            Number of threads to use.

    Returns:
        Double-precision NumPy array of (log-transformed) p-values to test for significant enrichment of ``x`` in each entry of ``sets``.

    References:
        https://libscran.github.io/phyper, for the underlying implementation.

    Examples:
        >>> LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        >>> import scranpy
        >>> scranpy.test_enrichment(
        >>>     x=["A", "B", "C", "D", "E"],
        >>>     sets={
        >>>         "set1": LETTERS[:10],
        >>>         "set2": ["B", "D", "F", "H", "J"],
        >>>         "set3": LETTERS[10:20]
        >>>     },
        >>>     universe=LETTERS
        >>> )
    """

    if isinstance(sets, dict):
        sets = list(sets.values())
    overlap = _count_overlaps(x, sets)

    if isinstance(universe, int):
        set_sizes = numpy.ndarray((len(sets),), dtype=numpy.uint32)
        for s, curset in enumerate(sets):
            set_sizes[s] = len(curset)
    else:
        set_sizes = _count_overlaps(universe, sets)
        universe = len(universe)

    return lib.test_enrichment(
        overlap,
        len(x),
        set_sizes,
        universe,
        log,
        num_threads
    )
