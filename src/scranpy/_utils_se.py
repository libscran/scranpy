from typing import Sequence, Union
import biocutils


def sanitize_altexp_assays(altexps: Union[dict, Sequence], all_altexps: Sequence, default_assay_type: str) -> dict:
    altexps = to_NamedList(altexps)

    if altexps.get_names() is not None:
        mapping = {}
        for nm in altexps.get_names():
            if nm in mapping:
                continue
            mapping[nm] = altexps[nm]
        return mapping
    
    mapping = {}
    for ae in altexps:
        if isinstance(ae, int):
            ae = all_altexps[ae]
        mapping[ae] = default_assay_type

    return mapping


def to_NamedList(x: Union[dict, Sequence]) -> biocutils.NamedList:
    if isinstance(x, biocutils.NamedList):
        return x
    if isinstance(x, dict):
        return bioc.NamedList.from_dict(x)
    return biocutils.NamedList.from_list(x)
