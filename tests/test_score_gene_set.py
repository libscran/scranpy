import scranpy
import numpy


def test_score_gene_set():
    numpy.random.rand(1000)
    x = numpy.random.randn(1000, 100)

    res = scranpy.score_gene_set(x, range(10, 60))
    assert len(res["scores"]) == x.shape[1]
    assert len(res["weights"]) == 50

    # Now with blocking. 
    block = (numpy.random.rand(100) * 3).astype(numpy.int32)
    bres = scranpy.score_gene_set(x, range(100, 200), block=block)
    assert len(bres["scores"]) == x.shape[1]
    assert len(bres["weights"]) == 100

    # Now with names.
    row_names = ["GENE_" + str(i) for i in range(x.shape[0])]
    nres = scranpy.score_gene_set(x, row_names[10:60], row_names=row_names)
    assert (res["scores"] == nres["scores"]).all()
    assert (res["weights"] == nres["weights"]).all()
