import scranpy
import numpy
import biocframe
import summarizedexperiment


def create_test_se():
    mean = numpy.random.rand(100) * 5
    mat = numpy.ndarray((100, 10), dtype=numpy.dtype("double"))
    for c in range(mat.shape[1]):
        mat[:,c] = numpy.random.poisson(lam=mean, size=mean.shape[0])
    se = summarizedexperiment.SummarizedExperiment({ "counts": mat })
    return scranpy.normalize_rna_counts_se(se)


def test_choose_rna_hvgs_se():
    se = create_test_se()
    out = scranpy.choose_rna_hvgs_se(se, top=20)
    assert out.get_row_data().has_column("mean")
    assert out.get_row_data()["hvg"].sum() > 0
    assert not out.get_row_data()["hvg"].all()

    # Works with a prefix.
    out = scranpy.choose_rna_hvgs_se(se, top=2000, output_prefix="VAR_")
    assert numpy.issubdtype(out.get_row_data()["VAR_mean"].dtype, numpy.dtype("double"))
    assert numpy.issubdtype(out.get_row_data()["VAR_variance"].dtype, numpy.dtype("double"))
    assert out.get_row_data()["VAR_hvg"].all()

    # Works with blocking.
    block = ["A", "B"] * 5
    out = scranpy.choose_rna_hvgs_se(se, top=20, block=block)
    assert not out.get_row_data().has_column("per_block")

    out = scranpy.choose_rna_hvgs_se(se, top=20, block=block, include_per_block=True)
    assert isinstance(out.get_row_data()["per_block"], biocframe.BiocFrame)
    assert numpy.issubdtype(out.get_row_data()["per_block"]["A"]["mean"].dtype, numpy.dtype("double"))
    assert numpy.issubdtype(out.get_row_data()["per_block"]["B"]["variance"].dtype, numpy.dtype("double"))
