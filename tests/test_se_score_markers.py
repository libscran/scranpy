import numpy
import scranpy
import biocutils
import summarizedexperiment


def create_test_se():
    mat = numpy.random.rand(50, 20)
    se = summarizedexperiment.SummarizedExperiment({ "logcounts": mat })
    se.set_row_names(["gene" + str(i) for i in range(mat.shape[0])], in_place=True)
    se.get_column_data().set_column("group", ["A", "B", "C", "D"] * 5, in_place=True)
    return se


def test_score_markers_se_basic():
    se = create_test_se()
    out = scranpy.score_markers_se(se, se.get_column_data()["group"]) 
    assert out.get_names().as_list() == ["A", "B", "C", "D"]

    for g in out.get_names():
        df = out[g]
        assert df.shape[0] == se.shape[0]
        assert sorted(se.get_row_names()) == sorted(df.get_row_names())

        assert numpy.issubdtype(df["cohens_d_mean"].dtype, numpy.dtype("double"))
        assert numpy.issubdtype(df["auc_median"].dtype, numpy.dtype("double"))
        assert numpy.issubdtype(df["delta_mean_min_rank"].dtype, numpy.dtype("uint32"))
        assert numpy.issubdtype(df["mean"].dtype, numpy.dtype("double"))
        assert numpy.issubdtype(df["detected"].dtype, numpy.dtype("double"))

        default_order = df["cohens_d_mean"] # i.e., the default order-by choice.
        for i in range(1, df.shape[0]):
            assert default_order[i] <= default_order[i-1]

    # Same results with an integer group.
    groups_as_int = biocutils.match(se.get_column_data()["group"], ["A", "B", "C", "D"])
    out_int = scranpy.score_markers_se(se, groups_as_int)
    assert out_int.get_names().as_list() == ["0", "1", "2", "3"]
    for g in range(4):
        df = out[g]
        df_int = out_int[g]
        assert (df.get_column("mean") == df_int.get_column("mean")).all()
        assert (df.get_column("cohens_d_median") == df_int.get_column("cohens_d_median")).all()
        assert (df.get_column("auc_mean") == df_int.get_column("auc_mean")).all()


def test_score_markers_se_extra_columns():
    se = create_test_se()
    symbols = ["SYMBOL-" + str(i) for i in range(se.shape[0])]
    se.get_row_data().set_column("symbol", symbols, in_place=True)

    out = scranpy.score_markers_se(se, se.get_column_data()["group"], extra_columns="symbol")
    for g in out.get_names():
        df = out[g]
        m = biocutils.match(df.get_row_names(), se.get_row_names())
        assert df.get_column("symbol") == biocutils.subset(symbols, m)

    # Same results without row names.
    unnamed = se.set_row_names(None)
    unout = scranpy.score_markers_se(unnamed, unnamed.get_column_data()["group"], extra_columns="symbol")
    assert out.get_names() == unout.get_names()
    for g in out.get_names():
        assert out[g]["symbol"] == unout[g]["symbol"]
        assert (out[g]["mean"] == unout[g]["mean"]).all()
        assert (out[g]["delta_mean_median"] == unout[g]["delta_mean_median"]).all()


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
            "compute_group_mean": False,
            "compute_group_detected": False,
            "compute_cohens_d": False
        }
    )

    for g in out.get_names():
        df = out[g]
        assert df.shape[0] == se.shape[0]
        assert sorted(se.get_row_names()) == sorted(df.get_row_names())

        assert not df.has_column("mean")
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
    assert preview.get_row_names() is not None

    order_preview = scranpy.preview_markers(out[0], order_by=True)
    assert order_preview.get_column_names().as_list() == ["mean", "detected", "lfc", "cohens_d_mean"]
    assert order_preview.shape[0] == 10
    assert order_preview.get_row_names() is not None

    none_preview = scranpy.preview_markers(out[0], columns=None, order_by=True, include_order_by=False)
    assert len(none_preview.get_column_names()) == 0
    assert none_preview.shape[0] == 10
    assert none_preview.get_row_names() == order_preview.get_row_names()

    eff_preview = scranpy.preview_markers(out[0], order_by=True, include_order_by="effect")
    assert eff_preview.get_column_names().as_list() == ["mean", "detected", "lfc", "effect"]
    assert eff_preview.shape[0] == 10
    assert eff_preview.get_row_names() == order_preview.get_row_names()

    preview = scranpy.preview_markers(out[0], rows=None)
    assert out[0].shape[0] == preview.shape[0]
    assert out[0].get_row_names() == preview.get_row_names()

    preview = scranpy.preview_markers(out[0], order_by="auc_median")
    assert preview.shape[0] == 10

    preview = scranpy.preview_markers(out[0], order_by="auc_median", rows=None)
    assert preview.shape[0] == out[0].shape[0]
    assert preview.get_row_names() == biocutils.subset(out[0].get_row_names(), numpy.argsort(-out[0]["auc_median"]))

    preview = scranpy.preview_markers(out[0], order_by="auc_min_rank", rows=None)
    assert preview.get_row_names() == biocutils.subset(out[0].get_row_names(), numpy.argsort(out[0]["auc_min_rank"]))
