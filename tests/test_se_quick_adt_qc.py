import numpy
import scranpy
import biocframe
import summarizedexperiment


def test_quick_adt_qc_se():
    y = (numpy.random.rand(100, 1000) * 5).astype(numpy.uint32)
    se = summarizedexperiment.SummarizedExperiment(
        { "counts": y },
        column_data = biocframe.BiocFrame({ "foo": numpy.random.rand(1000) })
    )

    out = scranpy.quick_adt_qc_se(se, subsets={ "igg": [1,2,3] })
    assert "sum" in out.get_column_data().get_column_names()
    assert "detected" in out.get_column_data().get_column_names()
    assert "subset_sum_igg" in out.get_column_data().get_column_names()
    assert out.get_metadata()["qc"]["thresholds"] is not None
    assert "foo" in out.get_column_data().get_column_names() # existing columns are still preserved.

    # Works if we slap in a prefix.   
    out = scranpy.quick_adt_qc_se(se, subsets={ "igg": [1,2,3] }, output_prefix="FOO_")
    assert "FOO_sum" in out.get_column_data().get_column_names()

    # Works if we skip the metadata.
    out = scranpy.quick_adt_qc_se(se, subsets=[], meta_name=None)
    assert "qc" not in out.get_metadata()
