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

        output = numpy.zeros((length,), dtype=numpy.bool)
        if numpy.issubdtype(selection.dtype, numpy.integer):
            output[selection] = True
            return output
        elif numpy.issubdtype(selection.dtype, numpy.str_):
            if "realized" not in cached_mapping:
                cached_mapping["realized"] = gutils.create_row_names_mapping(row_names, length)
            found = cached_mapping["realized"]
            for ss in selection:
                if ss in found:
                    output[found[ss]] = True
        else:
            raise TypeError("'selection.dtype' should either be bool, integer or string")

    output = numpy.zeros((length,), dtype=numpy.bool)
    if len(selection) == 0:
        return output

    has_bool = False
    has_str = False
    has_int = False
    for ss in selection:
        if isinstance(ss, bool) or isinstance(ss, numpy.bool):
            has_bool = True
        elif isinstance(ss, str) or isinstance(ss, numpy.str_):
            has_str = True
        elif isinstance(ss, int) or isinstance(ss, numpy.integer):
            has_int = True
        else:
            raise TypeError("unknown type " + str(type(ss)) + " in 'selections'")

    if has_bool:
        if has_str or has_int:
            raise TypeError("'selection' with booleans should only contain booleans")
        if len(selection) != length:
            raise ValueError("length of 'selection' is not equal to 'length'")
        output[:] = selection
        return output

    if not has_str:
        output[selection] = True
        return output

    if "realized" not in cached_mapping:
        cached_mapping["realized"] = gutils.create_row_names_mapping(row_names, length)
    found = cached_mapping["realized"]

    for ss in selection:
        if isinstance(ss, str) or isinstance(ss, numpy.str_):
            if ss not in found:
                continue
            ss = found[ss]
        output[ss] = True

    return output


def _populate_subset_thresholds(thresholds: Union[dict, biocutils.NamedList], subset_field: str, has_block: bool) -> biocutils.NamedList: 
    thresholds = gutils.to_NamedList(thresholds)
    if subset_field not in thresholds.get_names(): 
        if has_block:
            CON = biocutils.NamedList
        else:
            CON = biocutils.FloatList
        thresholds = thresholds.set_value(subset_field, CON([], []))
    return thresholds


def _check_block_names(threshold_names: biocutils.Names, expected_names: biocutils.Names, message: str):
    if threshold_names != expected_names:
        raise TypeError("names on 'thresholds[\"" + message + "\"]' do not match those in 'block_ids'")
