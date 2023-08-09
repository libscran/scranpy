import numpy as np
from mattress import tatamize
from scranpy.normalization import LogNormalizeCountsArgs, log_norm_counts

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_log_norm_counts(mock_data):
    x = mock_data.x
    y = tatamize(x)
    result = log_norm_counts(y)

    # Comparison to a reference.
    sf = x.sum(0)
    csf = sf / sf.mean()
    ref = np.log2(x[0, :] / csf + 1)
    assert np.allclose(result.row(0), ref)

    # Works without centering.
    result_uncentered = log_norm_counts(y, LogNormalizeCountsArgs(center=False))
    assert np.allclose(result_uncentered.row(0), np.log2(x[0, :] / sf + 1))

    result_blocked = log_norm_counts(y, LogNormalizeCountsArgs(block=mock_data.block))
    first_blocked = result_blocked.row(0)
    assert np.allclose(first_blocked, ref) is False

    # Same results after parallelization.
    result_parallel = log_norm_counts(y, LogNormalizeCountsArgs(num_threads=3))
    assert (result.row(0) == result_parallel.row(0)).all()
    last = result.nrow() - 1
    assert (result.row(last) == result_parallel.row(last)).all()
