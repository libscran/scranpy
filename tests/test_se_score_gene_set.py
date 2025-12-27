import numpy
import scranpy
import summarizedexperiment


def test_score_gene_set_se():
    mat = numpy.random.rand(200, 10)
    sce = summarizedexperiment.SummarizedExperiment({ "logcounts": mat })

    out = scranpy.score_gene_set_se(sce, set=range(10))
    assert len(out.scores) == sce.shape[1]
    assert len(out.weights) == 10
