import numpy
import pytest
import scranpy._utils_qc as qcutils


def test_to_logical_slice():
    out = qcutils._to_logical(slice(5, 8), 10, cached_mapping={}, row_names=None)
    assert (out == numpy.array([False] * 5 + [True] * 3 + [False] * 2)).all()

    out = qcutils._to_logical(range(0, 5), 10, cached_mapping={}, row_names=None)
    assert (out == numpy.array([True] * 5 + [False] * 5)).all()


def test_to_logical_numpy():
    y = numpy.array([False, True, False, True])
    out = qcutils._to_logical(y, 4, cached_mapping={}, row_names=None)
    assert (out == y).all()
    with pytest.raises(Exception, match="length of 'selection'"):
        out = qcutils._to_logical(y, 5, cached_mapping={}, row_names=None)

    y = numpy.array([2, 4, 8, 0, 6])
    out = qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)
    assert (out == numpy.array([True, False] * 5)).all()

    out = qcutils._to_logical([], 10, cached_mapping={}, row_names=None)
    assert len(out) == 10
    assert not out.any()

    # Works with NumPy strings.
    y = numpy.array(["GENE_1", "GENE_3", "GENE_5", "GENE_7", "GENE_9"])
    with pytest.raises(Exception, match="mapping names"):
        qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)

    mapping = {}
    names = ["GENE_" + str(i) for i in range(10)]
    out = qcutils._to_logical(y, 10, cached_mapping=mapping, row_names=names)
    assert (out == numpy.array([False, True] * 5)).all()
    assert "realized" in mapping

    y2 = numpy.array(["FOO"] + list(y) + ["BAR"]) # just ignores missing names.
    out2 = qcutils._to_logical(y2, 10, cached_mapping=mapping, row_names=names)
    assert (out == out2).all()


def test_to_logical_bool_list():
    y = [True, False] * 5
    out = qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)
    assert (out == numpy.array([True, False] * 5)).all()

    y = [True, 1, 2]
    with pytest.raises(Exception, match="should only contain"):
        qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)
    y = [True]
    with pytest.raises(Exception, match="length of 'selection'"):
        qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)

    y2 = [numpy.bool(True), numpy.bool(False)] * 5 # still works with a list of NumPy scalars.
    out2 = qcutils._to_logical(y2, 10, cached_mapping={}, row_names=None)
    assert (out == out2).all()


def test_to_logical_int_list():
    y = [2, 4, 8, 0, 6]
    out = qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)
    assert (out == numpy.array([True, False] * 5)).all()

    y2 = [numpy.int32(x) for x in y] # still works with a list of NumPy scalars.
    out2 = qcutils._to_logical(y2, 10, cached_mapping={}, row_names=None)
    assert (out == out2).all()


def test_to_logical_str_list():
    y = ["GENE_1", "GENE_3", "GENE_5", "GENE_7", "GENE_9"]
    with pytest.raises(Exception, match="mapping names"):
        qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)

    mapping = {}
    names = ["GENE_" + str(i) for i in range(10)]
    out = qcutils._to_logical(y, 10, cached_mapping=mapping, row_names=names)
    assert (out == numpy.array([False, True] * 5)).all()
    assert "realized" in mapping

    y2 = ["FOO"] + y + ["BAR"] # just ignores missing names.
    out2 = qcutils._to_logical(y, 10, cached_mapping=mapping, row_names=names)
    assert (out == out2).all()

    y2 = [numpy.str_(x) for x in y] # still works with a list of NumPy scalars.
    out2 = qcutils._to_logical(y2, 10, cached_mapping=mapping, row_names=names)
    assert (out == out2).all()
