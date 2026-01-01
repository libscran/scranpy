from typing import Sequence, Tuple, Union, Optional
from collections.abc import Mapping

import numpy
import biocutils

from . import _utils_general as gutils


def _sanitize_subsets(x: Union[Sequence, Mapping], extent: int, row_names: Optional[Sequence]) -> Tuple:
    if isinstance(x, biocutils.NamedList):
        if x.get_names() is None:
            raise ValueError("subsets should be named")
        keys = x.get_names().as_list()
        vals = list(x.as_list())
    elif isinstance(x, Mapping):
        keys = x.keys()
        vals = list(x.values())
    elif len(x) == 0 or x is None:
        keys = []
        vals = []
    else:
        raise ValueError("unknown type " + str(type(x)) + " for the subsets")

    cached_mapping = {}
    for i, s in enumerate(vals):
        vals[i] = _to_logical(s, extent, cached_mapping, row_names)
    return keys, vals


def _to_logical(selection: Sequence, length: int, cached_mapping: dict, row_names: Optional[Sequence]) -> numpy.ndarray:
    if isinstance(selection, range) or isinstance(selection, slice):
        output = numpy.zeros((length,), dtype=numpy.bool)
        output[selection] = True
        return output

    if isinstance(selection, numpy.ndarray):
        if numpy.issubdtype(selection.dtype, numpy.bool):
            if len(selection) != length:
                raise ValueError("length of 'selection' is not equal to 'length'")
            return selection
        elif numpy.issubdtype(selection.dtype, numpy.integer):
            output = numpy.zeros((length,), dtype=numpy.bool)
            output[selection] = True
            return output
        else:
            raise TypeError("'selection.dtype' should either be bool or integer")

    output = numpy.zeros((length,), dtype=numpy.bool)
    if len(selection) == 0:
        return output

    all_types = set()
    for ss in selection:
        all_types.add(type(ss))

    if bool in all_types:
        if len(all_types) > 1:
            raise TypeError("'selection' with booleans should only contain booleans")
        if len(selection) != length:
            raise ValueError("length of 'selection' is not equal to 'length'")
        output[:] = selection
        return output

    found = None
    if str in all_types:
        if "realized" not in cached_mapping:
            cached_mapping["realized"] = gutils.create_row_names_mapping(row_names, length)
        found = cached_mapping["realized"]

    for ss in selection:
        if isinstance(ss, int):
            output[ss] = True
        elif isinstance(ss, str):
            output[found[ss]] = True
        else:
            raise TypeError("unknown type " + str(type(ss)) + " in 'selections'")

    return output
