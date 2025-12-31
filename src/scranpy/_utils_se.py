from typing import Sequence, Union
import biocutils


def sanitize_altexp_assays(altexps: Union[str, int, dict, Sequence, biocutils.NamedList], all_altexps: Sequence, default_assay_type: str) -> dict:
    if isinstance(altexps, str):
        return { altexps: default_assay_type }
    elif isinstance(altexps, int):
        return { all_altexps[altexps]: default_assay_type }
    elif isinstance(altexps, dict):
        return altexps

    mapping = {}
    if isinstance(altexps, biocutils.NamedList) and altexps.get_names() is not None:
        for nm in altexps.get_names():
            if nm in mapping:
                continue
            mapping[nm] = altexps[nm]
    else:
        for ae in altexps:
            if isinstance(ae, int):
                ae = all_altexps[ae]
            mapping[ae] = default_assay_type

    return mapping


def to_NamedList(x: Union[dict, Sequence, biocutils.NamedList]) -> biocutils.NamedList:
    if isinstance(x, biocutils.NamedList):
        return x
    if isinstance(x, dict):
        return biocutils.NamedList.from_dict(x)
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
        return numpy.transpose(mat) # this should be the same as the 'mat' supplied to add_transposed_reddim.


    def add_transposed_reddim(
        x: singlecellexperiment.SingleCellExperiment,
        name: Union[int, str, tuple],
        mat: numpy.ndarray
    ) -> singlecellexperiment.SingleCellExperiment:
        mat = numpy.transpose(mat) # this should be a view if 'mat' is contiguous, so no copy is made.
        return x.set_reduced_dimension(name, mat)
