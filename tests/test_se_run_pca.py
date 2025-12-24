import scranpy
import numpy
import summarizedexperiment 
import singlecellexperiment


def test_run_pca_basic():
    mat = numpy.random.rand(500, 200)
    sce = singlecellexperiment.SingleCellExperiment({ "logcounts": mat })

    out = scranpy.run_pca_se(sce, features=range(50))
    assert "PCA" in out.get_reduced_dimension_names()
    assert out.get_metadata()["PCA"].rotation.shape[0] == 50
    assert out.get_reduced_dimension("PCA").shape[1] == 25

    null = scranpy.run_pca_se(sce[:50,:], features=None)
    assert (out.get_reduced_dimension("PCA") == null.get_reduced_dimension("PCA")).all()
    assert (out.get_metadata()["PCA"].rotation == null.get_metadata()["PCA"].rotation).all()

    nometa = scranpy.run_pca_se(sce, features=range(10), meta_name=None)
    assert not "PCA" in nometa.get_metadata()


def test_run_pca_se():
    mat = numpy.random.rand(500, 200)
    se = summarizedexperiment.SummarizedExperiment({ "logcounts": mat })
    out = scranpy.run_pca_se(se, features=range(5, 100))
    assert isinstance(out, singlecellexperiment.SingleCellExperiment)
    assert "PCA" in out.get_reduced_dimension_names()
