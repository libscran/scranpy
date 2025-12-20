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

    
if biocutils.package_utils.is_package_installed("singlecellexperiment"):
    import singlecellexperiment
    import numpy
    import delayedarray


    def get_transposed_reddim(x: singlecellexperiment.SingleCellExperiment, name: Union[int, str, tuple]) -> numpy.ndarray:
        if not isinstance(name, tuple):
            mat = x.get_reduced_dimension(name)
        else:
            mat = x.get_alternative_experiment(name[0]).get_reduced_dimension(name[1])

        mat = numpy.transpose(mat)
        if isinstance(mat, numpy.ndarray):
            return mat

        # Possibly a no-op if .add_transposed_reddim was set with delayed=TRUE.
        if isinstance(mat, delayedarray.DelayedArray):
            if isinstance(mat, numpy.ndarray):
                return mat

        return delayedarray.to_dense_array(mat)
