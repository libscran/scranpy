import numpy
import pytest
import scranpy._utils_qc as qcutils


def test_to_logical():
    out = qcutils._to_logical(slice(5, 8), 10, cached_mapping={}, row_names=None)
    assert (out == numpy.array([False] * 5 + [True] * 3 + [False] * 2)).all()

    out = qcutils._to_logical(range(0, 5), 10, cached_mapping={}, row_names=None)
    assert (out == numpy.array([True] * 5 + [False] * 5)).all()

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

    y = [True, False] * 5
    out = qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)
    assert (out == numpy.array([True, False] * 5)).all()

    y = [True, 1, 2]
    with pytest.raises(Exception, match="should only contain"):
        qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)
    y = [True]
    with pytest.raises(Exception, match="length of 'selection'"):
        qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)

    y = [2, 4, 8, 0, 6]
    out = qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)
    assert (out == numpy.array([True, False] * 5)).all()

    y = ["GENE_1", "GENE_3", "GENE_5", "GENE_7", "GENE_9"]
    with pytest.raises(Exception, match="mapping names"):
        qcutils._to_logical(y, 10, cached_mapping={}, row_names=None)
    mapping = {}
    names = ["GENE_" + str(i) for i in range(10)]
    out = qcutils._to_logical(y, 10, cached_mapping=mapping, row_names=names)
    assert (out == numpy.array([False, True] * 5)).all()
    assert "realized" in mapping
