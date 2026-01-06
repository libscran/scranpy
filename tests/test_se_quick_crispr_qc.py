import numpy
import scranpy
import summarizedexperiment


def create_test_se():
    mat = numpy.reshape(numpy.random.poisson(size=1000, lam=2), (100, 10))
    return summarizedexperiment.SummarizedExperiment({ "counts": mat })


def test_quick_crispr_qc_se():
    se = create_test_se() 

    out = scranpy.quick_crispr_qc_se(se)
    assert numpy.issubdtype(out.get_column_data()["sum"].dtype, numpy.float64)
    assert numpy.issubdtype(out.get_column_data()["detected"].dtype, numpy.uint32)
    assert numpy.issubdtype(out.get_column_data()["max_value"].dtype, numpy.float64)
    assert numpy.issubdtype(out.get_column_data()["max_index"].dtype, numpy.uint32)
    assert numpy.issubdtype(out.get_column_data()["keep"].dtype, numpy.bool)
    assert "thresholds" in out.get_metadata()["qc"]

    out2 = scranpy.quick_crispr_qc_se(se, output_prefix="WHEE_")
    assert (out.get_column_data()["sum"] == out2.get_column_data()["WHEE_sum"]).all()
    assert (out.get_column_data()["detected"] == out2.get_column_data()["WHEE_detected"]).all()
    assert (out.get_column_data()["max_value"] == out2.get_column_data()["WHEE_max_value"]).all()
    assert (out.get_column_data()["max_index"] == out2.get_column_data()["WHEE_max_index"]).all()
    assert (out.get_column_data()["keep"] == out2.get_column_data()["WHEE_keep"]).all()

    out3 = scranpy.quick_crispr_qc_se(se, meta_name=None)
    assert not "qc" in out3.get_metadata()


def test_quick_crispr_qc_predefined():
    mat = numpy.reshape(numpy.random.poisson(size=1000, lam=1), (100, 10))
    se = summarizedexperiment.SummarizedExperiment({ "counts": mat })

    out = scranpy.quick_crispr_qc_se(se, thresholds={ "max_value": 1 })
    assert out.get_metadata()["qc"]["thresholds"]["max_value"] == 1
