from typing import Sequence, Union, Optional

import biocutils


def to_NamedList(x: Union[dict, Sequence, biocutils.NamedList]) -> biocutils.NamedList:
    if isinstance(x, biocutils.NamedList):
        return x
    if isinstance(x, dict):
        return biocutils.NamedList.from_dict(x)
    return biocutils.NamedList.from_list(x)


def create_row_names_mapping(row_names: Optional[Sequence]) -> dict:
    if row_names is None:
        raise ValueError("no 'row_names' supplied for mapping names to row indices")
    found = {}
    for i, s in enumerate(row_names):
        if s not in found:
            found[s] = i
    return found
