import numpy
import scranpy
import singlecellexperiment
import delayedarray


def test_correct_mnn_se():
    sce = singlecellexperiment.SingleCellExperiment({ "counts": numpy.random.rand(0, 200) })
    sce = sce.set_reduced_dimension("PCA", numpy.random.rand(200, 5))

    block = ["A", "B"] * 100
    out = scranpy.correct_mnn_se(sce, block=block)
    assert isinstance(out.get_reduced_dimension("MNN"), numpy.ndarray)

    out = scranpy.correct_mnn_se(sce, block=block, delayed_transpose=True)
    assert isinstance(out.get_reduced_dimension("MNN"), delayedarray.DelayedArray)
