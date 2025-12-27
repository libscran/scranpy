import numpy
import scranpy
import singlecellexperiment


def test_run_tsne_se():
    mat = numpy.random.rand(0, 200)
    sce = singlecellexperiment.SingleCellExperiment({ "counts": mat })
    sce.set_reduced_dimension("PCA", numpy.random.rand(200, 5), in_place=True)
    out = scranpy.run_tsne_se(sce)
    assert "TSNE" in out.get_reduced_dimension_names()
