import numpy
import scranpy
import biocframe
import summarizedexperiment
import singlecellexperiment 


def test_quick_rna_qc_basic():
    mat = numpy.reshape(numpy.random.poisson(size=1000, lam=1), (100, 10))
    se = summarizedexperiment.SummarizedExperiment({ "counts": mat })

    out = scranpy.quick_rna_qc_se(se, subsets=[])
    assert numpy.issubdtype(out.get_column_data()["sum"].dtype, numpy.float64)
    assert numpy.issubdtype(out.get_column_data()["detected"].dtype, numpy.uint32)
    assert numpy.issubdtype(out.get_column_data()["keep"].dtype, numpy.bool)
    assert "thresholds" in out.get_metadata()["qc"]

    out = scranpy.quick_rna_qc_se(se, subsets={ "mt": range(5) })
    assert numpy.issubdtype(out.get_column_data()["subset_proportion_mt"].dtype, numpy.float64)

    out2 = scranpy.quick_rna_qc_se(se, subsets={ "mt": range(5) }, output_prefix="WHEE_")
    assert (out.get_column_data()["sum"] == out2.get_column_data()["WHEE_sum"]).all()
    assert (out.get_column_data()["detected"] == out2.get_column_data()["WHEE_detected"]).all()
    assert (out.get_column_data()["subset_proportion_mt"] == out2.get_column_data()["WHEE_subset_proportion_mt"]).all()
    assert (out.get_column_data()["keep"] == out2.get_column_data()["WHEE_keep"]).all()

    out3 = scranpy.quick_rna_qc_se(se, subsets={ "mt": range(5), "bar": range(5, 10) }, flatten=False)
    assert (out.get_column_data()["subset_proportion_mt"] == out3.get_column_data()["subset_proportion"]["mt"]).all()
    assert numpy.issubdtype(out3.get_column_data()["subset_proportion"]["bar"].dtype, numpy.float64)

    out4 = scranpy.quick_rna_qc_se(se, subsets=[], meta_name=None)
    assert not "qc" in out4.get_metadata()


def test_quick_rna_qc_altexp():
    mat = numpy.reshape(numpy.random.poisson(size=1000, lam=1), (100, 10))
    sce = singlecellexperiment.SingleCellExperiment({ "counts": mat })

    ercc_mat = numpy.reshape(numpy.random.poisson(size=100, lam=1), (10, 10))
    ercc_se = summarizedexperiment.SummarizedExperiment({ "counts": ercc_mat })
    sce = sce.set_alternative_experiment("ERCC", ercc_se)

    out = scranpy.quick_rna_qc_se(sce, subsets=[], altexp_proportions=["ERCC"])
    expected = 1 / (1 + out.get_column_data()["sum"] / out.get_alternative_experiment("ERCC").get_column_data()["sum"])
    assert numpy.allclose(out.get_column_data()["subset_proportion_ERCC"], expected)

    out = scranpy.quick_rna_qc_se(sce, subsets=[], altexp_proportions=["ERCC"], output_prefix="WHEE_")
    assert numpy.allclose(out.get_column_data()["WHEE_subset_proportion_ERCC"], expected)


def test_quick_rna_qc_predefined():
    mat = numpy.reshape(numpy.random.poisson(size=1000, lam=1), (100, 10))
    se = summarizedexperiment.SummarizedExperiment({ "counts": mat })

    out = scranpy.quick_rna_qc_se(se, subsets=[], thresholds={ "sum": 1, "detected": 1 })
    assert out.get_metadata()["qc"]["thresholds"]["sum"] == 1
    assert out.get_metadata()["qc"]["thresholds"]["detected"] == 1
    assert len(out.get_metadata()["qc"]["thresholds"]["subset_proportion"]) == 0
