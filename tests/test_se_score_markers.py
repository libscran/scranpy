import numpy
import scranpy
import biocutils
import summarizedexperiment


def create_test_se():
    mat = numpy.random.rand(50, 20)
    se = summarizedexperiment.SummarizedExperiment({ "logcounts": mat })
#    se.set_row_names(["gene" + str(i) for i in range(mat.shape[0])], in_place=True)
    se.get_column_data().set_column("group", ["A", "B", "C", "D"] * 5, in_place=True)
    return se


def test_score_markers_se_basic():
    se = create_test_se()
    out = scranpy.score_markers_se(se, se.get_column_data()["group"]) 
    assert out.get_names().as_list() == ["A", "B", "C", "D"]

    for g in out.get_names():
        df = out[g]
        assert df.shape[0] == se.shape[0]
        assert se.get_row_names() == df.get_row_names()

        assert numpy.issubdtype(df["cohens_d_mean"].dtype, numpy.dtype("double"))
        assert numpy.issubdtype(df["auc_median"].dtype, numpy.dtype("double"))
        assert numpy.issubdtype(df["delta_mean_min_rank"].dtype, numpy.dtype("uint32"))
        assert numpy.issubdtype(df["mean"].dtype, numpy.dtype("double"))
        assert numpy.issubdtype(df["detected"].dtype, numpy.dtype("double"))

        default_order = df["cohens_d_mean"] # i.e., the default order-by choice.
        for i in range(1, df.shape[0]):
            assert default_order[i] <= default_order[i-1]


def test_score_markers_se_extra_columns():
    se = create_test_se()
    symbols = ["SYMBOL-" + str(i) for i in range(se.shape[0])]
    se.get_row_data().set_column("symbol", symbols, in_place=True)

    out = scranpy.score_markers_se(se, se.get_column_data()["group"], extra_columns="symbol")
    for g in out.get_names():
        df = out[g]
#        m = biocutils.match(df.get_row_names(), se.get_row_names()) # TODO: fix this once we get some row names on the score_markers output.
#        assert df.get_column("symbol") == biocutils.subset(symbols, m)


def test_score_markers_se_quantiles():
    se = create_test_se()
    out = scranpy.score_markers_se(se, se.get_column_data()["group"], more_marker_args={ "compute_summary_quantiles": [0, 0.5, 1] })
    for g in out.get_names():
        df = out[g]
        assert numpy.allclose(df.get_column("auc_quantile_0.5"), df.get_column("auc_median"))
        assert numpy.allclose(df.get_column("cohens_d_quantile_0.0"), df.get_column("cohens_d_min"))
        assert numpy.allclose(df.get_column("delta_detected_quantile_1.0"), df.get_column("delta_detected_max"))


def test_score_markers_se_none():
    se = create_test_se()
    out = scranpy.score_markers_se(
        se,
        se.get_column_data()["group"],
        more_marker_args={ 
            #"compute_group_mean": False, # TODO: uncomment this when score_markers actually returns DFs.
            "compute_group_detected": False,
            "compute_cohens_d": False
        }
    )

    for g in out.get_names():
        df = out[g]
        assert df.shape[0] == se.shape[0]
        assert se.get_row_names() == df.get_row_names()

        #assert not df.has_column("mean")
        assert not df.has_column("detected")
        assert not df.has_column("cohens_d_mean")
        assert df.has_column("auc_mean")

        default_order = df["auc_mean"] # i.e., the next default order-by choice.
        for i in range(1, df.shape[0]):
            assert default_order[i] <= default_order[i-1]


def test_score_markers_se_min_rank():
    se = create_test_se()
    out = scranpy.score_markers_se(se, se.get_column_data()["group"], order_by="cohens_d_min_rank")
    for g in out.get_names():
        df = out[g]
        default_order = df["cohens_d_min_rank"]
        for i in range(1, df.shape[0]):
            assert default_order[i] >= default_order[i-1]


def test_preview_markers():
    se = create_test_se()
    out = scranpy.score_markers_se(se, se.get_column_data()["group"])

    preview = scranpy.preview_markers(out[0])
    assert preview.get_column_names().as_list() == ["mean", "detected", "lfc"]
    assert preview.shape[0] == 10

    preview = scranpy.preview_markers(out[0], order_by=True)
    assert preview.get_column_names().as_list() == ["mean", "detected", "lfc", "cohens_d_mean"]
    assert preview.shape[0] == 10

    preview = scranpy.preview_markers(out[0], columns=None, include_order_by=False)
    assert len(preview.get_column_names()) == 0
    assert preview.shape[0] == 10

    preview = scranpy.preview_markers(out[0], rows=None)
    assert out[0].shape[0] == preview.shape[0]
    assert out[0].get_row_names() == preview.get_row_names()

    preview = scranpy.preview_markers(out[0], order_by="auc_median")
    assert preview.shape[0] == 10

    preview = scranpy.preview_markers(out[0], order_by="auc_median", rows=None)
    assert preview.shape[0] == out[0].shape[0]
    #assert preview.get_row_names() == biocutils.subset(out[0].get_names(), numpy.argsort(out[0]["auc_median"]))

    preview = scranpy.preview_markers(out[0], order_by="auc_min_rank", rows=None)
    #assert preview.get_row_names() == biocutils.subset(out[0].get_names(), numpy.argsort(out[0]["auc_min_rank"]))
