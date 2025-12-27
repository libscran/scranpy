import scranpy
import numpy
import delayedarray
import summarizedexperiment


def create_test_se():
    mat = numpy.reshape(numpy.random.poisson(lam=5, size=1000), (100, 10))
    return summarizedexperiment.SummarizedExperiment({ "counts": mat })


def test_normalize_rna_counts_se():
    se = create_test_se()
    out = scranpy.normalize_rna_counts_se(se)
    assert numpy.issubdtype(out.get_column_data()["size_factor"].dtype, numpy.dtype("double"))
    assert numpy.isclose(out.get_column_data()["size_factor"].mean(), 1)
    assert isinstance(out.get_assay("logcounts"), delayedarray.DelayedArray)

    # Check size factors are correctly computed.
    libs = se.get_assay(0).sum(axis=0)
    out2 = scranpy.normalize_rna_counts_se(se, size_factors=libs)
    assert numpy.allclose(out.get_column_data()["size_factor"], out2.get_column_data()["size_factor"])

    sf = libs/libs.mean()
    out3 = scranpy.normalize_rna_counts_se(se, size_factors=sf, center=False)
    assert numpy.allclose(out.get_column_data()["size_factor"], out3.get_column_data()["size_factor"])

    # Works with blocking.
    block = ["A", "B"] * 5
    out_block = scranpy.normalize_rna_counts_se(se, block=block)
    assert not numpy.allclose(out.get_column_data()["size_factor"], out_block.get_column_data()["size_factor"])

    # Can disable the saving of the factors.
    out4 = scranpy.normalize_rna_counts_se(se, factor_name=None)
    assert not out4.get_column_data().has_column("size_factor")
    assert isinstance(out4.get_assay("logcounts"), delayedarray.DelayedArray)
