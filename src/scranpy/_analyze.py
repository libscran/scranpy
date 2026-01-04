from ._se_analyze import analyze_se

import biocutils


def analyze(*args, **kwargs) -> biocutils.NamedList:
    """Deprecated, use :py:func:`~scranpy.se_analyze.analyze_se` instead."""
    import warnings
    warnings.warn(DeprecationWarning("use 'scranpy.analyze_se()' instead"))
    return analyze_se(*args, **kwargs)
