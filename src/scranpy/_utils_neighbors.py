import numpy


def _check_neighbor_results(index, distance):
    if len(index.shape) != 2:
        raise ValueError("'index' should be a two-dimensional array")

    mn = index.min()
    if not numpy.isfinite(mn) or mn < 0:
        raise ValueError("'index' should contain finite positive integers")

    mx = index.max()
    if not numpy.isfinite(mx) or mx >= index.shape[0]:
        raise ValueError("'index' should contain finite integers no greater than the number of columns")

    if distance is not None and index.shape != distance.shape:
        raise ValueError("'index' and 'distance' should have the same shape")
