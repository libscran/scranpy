import scranpy
import numpy
import biocutils
import biocframe
import pytest

from scranpy._aggregate_across_genes import _sanitize_gene_set


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

    dfsets = [biocframe.BiocFrame({"index": s[0], "weight": s[1] }) for s in sets]
    dfagg = scranpy.aggregate_across_genes(x, dfsets)
    for i in range(len(sets)):
        assert (agg[i] == dfagg[i]).all()

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

    # Works with weights.
    wsets = [(s, numpy.random.rand(len(s))) for s in sets]
    wagg = scranpy.aggregate_across_genes(x, wsets)

    named_wsets = [(biocutils.subset(names, i), w) for i, w in wsets]
    named_wagg = scranpy.aggregate_across_genes(x, named_wsets, row_names=names)

    assert len(wagg) == len(named_wagg)
    for i in range(len(wsets)):
        assert (wagg[i] == named_wagg[i]).all()


def test_sanitize_gene_set_slice():
    payload = _sanitize_gene_set(range(10, 20), {}, None, nrow=20, weights=None)
    assert (payload == numpy.array(range(10, 20))).all()

    payload = _sanitize_gene_set(slice(10, 20), {}, None, nrow=20, weights=None)
    assert (payload == numpy.array(range(10, 20))).all()

    w = numpy.random.rand(10)
    payload_i, payload_w = _sanitize_gene_set(slice(10, 20), {}, None, nrow=20, weights=w)
    assert (payload_i == numpy.array(range(10, 20))).all()
    assert (payload_w == w).all()


def test_sanitize_gene_set_numpy_int():
    y = numpy.random.randint(20, size=10)
    payload = _sanitize_gene_set(y, {}, None, nrow=20, weights=None)
    assert (y == payload).all()

    w = numpy.random.rand(20)
    payload_i, payload_w = _sanitize_gene_set(y, {}, None, nrow=20, weights=w)
    assert (y == payload_i).all()
    assert (w == payload_w).all()


def test_sanitize_gene_set_numpy_bool():
    y = numpy.array([True, False] * 5)
    with pytest.raises(match="length of a boolean gene set"):
        _sanitize_gene_set(y, {}, None, nrow=20, weights=None)
    with pytest.raises(match="weights are not supported"):
        _sanitize_gene_set(y, {}, None, nrow=10, weights=range(10))

    payload = _sanitize_gene_set(y, {}, None, nrow=10, weights=None)
    assert (payload == numpy.array(range(0, 10, 2))).all()


def test_sanitize_gene_set_numpy_string():
    mapping = {}
    rnames = numpy.array(["GENE_" + str(i) for i in range(20)])
    payload = _sanitize_gene_set(rnames[:20:2], mapping=mapping, row_names=rnames, nrow=len(rnames), weights=None)
    assert (payload == range(0, 20, 2)).all()
    assert "realized" in mapping

    w = numpy.random.rand(5)
    payload_i, payload_w = _sanitize_gene_set(rnames[:20:4], mapping=mapping, row_names=rnames, nrow=len(rnames), weights=w)
    assert (payload_i == range(0, 20, 4)).all()
    assert (payload_w == w).all()

    missing = numpy.array(["BAR", "GENE_1", "GENE_10", "GENE_19", "FOO"])
    payload = _sanitize_gene_set(missing, mapping=mapping, row_names=rnames, nrow=len(rnames), weights=None)
    assert (payload == [1, 10, 19]).all()

    w = numpy.random.rand(5)
    payload_i, payload_w = _sanitize_gene_set(missing, mapping=mapping, row_names=rnames, nrow=len(rnames), weights=w)
    assert (payload_i == [1, 10, 19]).all()
    assert (payload_w == w[1:4]).all()


def test_sanitize_gene_set_list_bool():
    y = [True, False] * 5
    with pytest.raises(match="length of a boolean gene set"):
        _sanitize_gene_set(y, {}, None, nrow=20, weights=None)
    with pytest.raises(match="weights are not supported"):
        _sanitize_gene_set(y, {}, None, nrow=10, weights=range(10))

    payload = _sanitize_gene_set(y, {}, None, nrow=10, weights=None)
    assert (payload == numpy.array(range(0, 10, 2))).all()

    ny = [numpy.bool(x) for x in y]
    payload = _sanitize_gene_set(ny, {}, None, nrow=10, weights=None)
    assert (payload == numpy.array(range(0, 10, 2))).all()

    with pytest.raises(match="only contain booleans"):
        _sanitize_gene_set([True, 1], {}, None, nrow=10, weights=range(10))


def test_sanitize_gene_set_list_int():
    y = [1, 10, 19]
    payload = _sanitize_gene_set(y, {}, None, nrow=20, weights=None)
    assert (y == payload).all()

    w = numpy.random.rand(3)
    payload_i, payload_w = _sanitize_gene_set(y, {}, None, nrow=20, weights=w)
    assert (y == payload_i).all()
    assert (w == payload_w).all()


def test_sanitize_gene_set_list_string():
    mapping = {}
    rnames = ["GENE_" + str(i) for i in range(20)]
    payload = _sanitize_gene_set(rnames[:20:2], mapping=mapping, row_names=rnames, nrow=len(rnames), weights=None)
    assert (payload == range(0, 20, 2)).all()
    assert "realized" in mapping

    w = numpy.random.rand(5)
    payload_i, payload_w = _sanitize_gene_set(rnames[:20:4], mapping=mapping, row_names=rnames, nrow=len(rnames), weights=w)
    assert (payload_i == range(0, 20, 4)).all()
    assert (payload_w == w).all()

    missing = ["BAR", "GENE_1", "GENE_10", "GENE_19", "FOO"]
    payload = _sanitize_gene_set(missing, mapping=mapping, row_names=rnames, nrow=len(rnames), weights=None)
    assert (payload == [1, 10, 19]).all()

    w = numpy.random.rand(5)
    payload_i, payload_w = _sanitize_gene_set(missing, mapping=mapping, row_names=rnames, nrow=len(rnames), weights=w)
    assert (payload_i == [1, 10, 19]).all()
    assert (payload_w == w[1:4]).all()
