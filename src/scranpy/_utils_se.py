from typing import Sequence, Union
import biocutils


def sanitize_altexp_assays(altexps: Union[dict, Sequence], all_altexps: Sequence, default_assay_type: str) -> dict:
    if not isinstance(altexps, biocutils.NamedList):
        if isinstance(altexps, dict):
            altexps = biocutils.NamedList.from_dict(altexps)
        else:
            altexps = biocutils.NamedList.from_list(altexps)

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

