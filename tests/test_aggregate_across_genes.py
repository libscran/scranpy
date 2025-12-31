import scranpy
import numpy
import biocutils
import pytest


def test_aggregate_across_genes_unweighted():
    x = numpy.random.rand(1000, 100)

    sets = [
        (numpy.random.rand(20) * x.shape[0]).astype(numpy.int32),
        (numpy.random.rand(10) * x.shape[0]).astype(numpy.int32),
        (numpy.random.rand(500) * x.shape[0]).astype(numpy.int32)
    ]

    agg = scranpy.aggregate_across_genes(x, sets)
    for i, ss in enumerate(sets):
        assert numpy.allclose(agg[i], x[ss,:].sum(axis=0))

    agg = scranpy.aggregate_across_genes(x, sets, average=True)
    for i, ss in enumerate(sets):
        assert numpy.allclose(agg[i], x[ss,:].mean(axis=0))

    # Works with set names.
    names = ["foo", "bar", "whee"]
    agg = scranpy.aggregate_across_genes(x, biocutils.NamedList(sets, names))
    assert agg.get_names().as_list() == names


def test_aggregate_across_genes_weighted():
    x = numpy.random.rand(1000, 100)

    sets = [
        (
            (numpy.random.rand(20) * x.shape[0]).astype(numpy.int32),
            numpy.random.randn(20)
        ),
        (
            (numpy.random.rand(10) * x.shape[0]).astype(numpy.int32),
            numpy.random.randn(10)
        ),
        (
            (numpy.random.rand(500) * x.shape[0]).astype(numpy.int32),
            numpy.random.randn(500)
        )
    ]

    agg = scranpy.aggregate_across_genes(x, sets)
    for i, ss in enumerate(sets):
        assert numpy.allclose(agg[i], (x[ss[0],:].T * ss[1]).sum(axis=1))

    agg = scranpy.aggregate_across_genes(x, sets, average=True)
    for i, ss in enumerate(sets):
        assert numpy.allclose(agg[i], (x[ss[0],:].T * ss[1]).sum(axis=1) / ss[1].sum())

    with pytest.raises(Exception, match = "equal length"):
        scranpy.aggregate_across_genes(x, [([0], [1,2,3])])


def test_aggregate_across_genes_names():
    x = numpy.random.rand(1000, 100)
    sets = [
        (numpy.random.rand(20) * x.shape[0]).astype(numpy.int32),
        (numpy.random.rand(10) * x.shape[0]).astype(numpy.int32),
        (numpy.random.rand(500) * x.shape[0]).astype(numpy.int32)
    ]
    agg = scranpy.aggregate_across_genes(x, sets)

    # Works with gene names.
    names = ["GENE_" + str(i) for i in range(x.shape[0])]
    named_sets = [biocutils.subset(names, i) for i in sets]
    named_agg = scranpy.aggregate_across_genes(x, named_sets, row_names=names)

    assert len(agg) == len(named_agg)
    for i in range(len(sets)):
        assert (agg[i] == named_agg[i]).all()

    with pytest.raises(Exception, match="no 'row_names' supplied"):
        scranpy.aggregate_across_genes(x, named_sets)
