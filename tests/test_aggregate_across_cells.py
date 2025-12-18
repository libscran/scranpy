import scranpy
import numpy
import biocutils
import pytest


def test_aggregate_across_cells_single():
    numpy.random.seed(69)
    x = (numpy.random.rand(1000, 100) * 4).astype(numpy.int32)
    levels = ["A", "B", "C", "D", "E"]
    clusters = [levels[i] for i in (numpy.random.rand(x.shape[1]) * len(levels)).astype(numpy.int32)]
    agg = scranpy.aggregate_across_cells(x, (clusters,))

    assert len(agg.combinations) == 1
    assert agg.combinations[0] == levels
    assert [levels[i] for i in agg.index] == clusters
    assert agg.sum.shape == (1000, 5)
    assert agg.detected.shape == (1000, 5)

    for u, lev in enumerate(levels):
        chosen = [lev == c for c in clusters]
        assert numpy.array(chosen).sum() == agg.counts[u]
        submat = x[:,chosen]
        sum_expected = submat.sum(axis=1)
        assert numpy.allclose(sum_expected, agg.sum[:,u])
        detected_expected = (submat > 0).sum(axis=1)
        assert numpy.allclose(detected_expected, agg.detected[:,u])

    # Seeing if we can coerce it to a SE.
    se = agg.to_summarizedexperiment()
    assert sorted(se.get_assay_names()) == ["detected", "sum"]
    assert se.get_column_data().get_column("0") == levels
    assert (se.get_column_data().get_column("counts") == agg.counts).all()


def test_aggregate_across_cells_multiple():
    numpy.random.seed(692)
    x = (numpy.random.rand(1000, 500) * 4).astype(numpy.int32)
    levels = ["a", "b", "c", "d"]
    clusters = [levels[i] for i in (numpy.random.rand(x.shape[1]) * len(levels)).astype(numpy.int32)]
    samples = (numpy.random.rand(x.shape[1]) * 3).astype(numpy.int32)
    agg = scranpy.aggregate_across_cells(x, (clusters, samples))

    assert len(agg.combinations) == 2
    assert agg.combinations[0] == ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3
    assert agg.combinations[1] == [0, 1, 2] * 4
    assert [agg.combinations[0][i] for i in agg.index] == clusters
    assert [agg.combinations[1][i] for i in agg.index] == list(samples)
    assert agg.sum.shape == (1000, 12)
    assert agg.detected.shape == (1000, 12)

    for u in range(len(agg.combinations[0])):
        curclust = agg.combinations[0][u]
        cursamp = agg.combinations[1][u]
        chosen = [clusters[i] == curclust and samples[i] == cursamp for i in range(x.shape[1])]
        assert numpy.array(chosen).sum() == agg.counts[u]
        submat = x[:,chosen]
        sum_expected = submat.sum(axis=1)
        assert numpy.allclose(sum_expected, agg.sum[:,u])
        detected_expected = (submat > 0).sum(axis=1)
        assert numpy.allclose(detected_expected, agg.detected[:,u])

    # Seeing if we can coerce it to a SE.
    se = agg.to_summarizedexperiment()
    assert sorted(se.get_assay_names()) == ["detected", "sum"]
    assert se.get_column_data().get_column("0") == agg.combinations[0]
    assert se.get_column_data().get_column("1") == agg.combinations[1]
    assert (se.get_column_data().get_column("counts") == agg.counts).all()


def test_aggregate_across_cells_named():
    numpy.random.seed(69)
    x = (numpy.random.rand(1000, 100) * 4).astype(numpy.int32)
    levels = ["A", "B", "C", "D", "E"]
    clusters = [levels[i] for i in (numpy.random.rand(x.shape[1]) * len(levels)).astype(numpy.int32)]
    agg = scranpy.aggregate_across_cells(x, biocutils.NamedList([clusters], ["foo"]))

    assert agg.combinations.get_names().as_list() == ["foo"]
    assert agg.combinations[0] == levels

    se = agg.to_summarizedexperiment()
    assert se.get_column_data().get_column("foo") == agg.combinations[0]
    assert (se.get_column_data().get_column("counts") == agg.counts).all()

    # Fails if we try to coerce with a 'counts' name.
    agg = scranpy.aggregate_across_cells(x, biocutils.NamedList([clusters], ["counts"]))
    with pytest.raises(ValueError, match="conflicting"):
        agg.to_summarizedexperiment()
    se = agg.to_summarizedexperiment(include_counts=False)
    assert se.get_column_data().get_column("counts") == agg.combinations[0]


def create_test_se():
    import summarizedexperiment
    mat = numpy.random.rand(10, 100)
    se = summarizedexperiment.SummarizedExperiment({ "counts": mat })

    import random
    import biocframe
    random.seed(50)
    se.set_column_data(
        biocframe.BiocFrame({
            "stuff": random.choices(["A", "B", "C", "D", "E"], k=mat.shape[1]),
            "whee": random.choices([True, False], k=mat.shape[1])
        }),
        in_place=True
    )

    se.set_row_data(
        biocframe.BiocFrame({
            "foo": numpy.random.rand(mat.shape[0])
        }),
        in_place=True
    )
    return se


# TODO: move to biocutils.
def table(x, sort: bool = True) -> biocutils.IntegerList:
    output = {}
    for v in x:
        if v in output:
            output[v] += 1
        else:
            output[v] = 1

    if sort:
        collected = sorted(output.keys())
        tmp = {}
        for y in collected:
            tmp[y] = output[y]
        output = tmp

    return biocutils.IntegerList.from_dict(output)


def test_aggregate_across_cells_se_basic():
    se = create_test_se()
    out = scranpy.aggregate_across_cells_se(se, se.get_column_data()[:,"stuff"])
    assert out.shape[1] == len(set(se.get_column_data()["stuff"]))
    assert out.get_row_data().get_column_names() == se.get_row_data().get_column_names()
    assert out.get_column_data()["factor_stuff"] == out.get_column_data()["stuff"]
    assert sorted(out.get_column_data()["factor_stuff"]) == sorted(list(set(se.get_column_data()["stuff"])))

    refcounts = table(se.get_column_data()["stuff"])[out.get_column_data()["stuff"]]
    assert (out.get_column_data()["counts"] == refcounts.as_list()).all()
    assert (out.get_metadata()["aggregated"]["index"] == biocutils.match(se.get_column_data()["stuff"], out.get_column_data()["stuff"])).all()

    # Works if we provide a list.
    listy = scranpy.aggregate_across_cells_se(se, [se.get_column_data()["stuff"]], output_prefix=None)
    assert listy.get_column_data()["0"] == out.get_column_data()["stuff"]
    assert (listy.get_assay("sum") == out.get_assay("sum")).all()

    # Works with multiple factors.
    mult = scranpy.aggregate_across_cells_se(se, se.get_column_data()[["stuff", "whee"]])
    assert mult.get_column_data()["factor_stuff"] == mult.get_column_data()["stuff"]
    assert mult.get_column_data()["factor_whee"] == mult.get_column_data()["whee"]

    left_combined = list(zip(se.get_column_data()["stuff"], se.get_column_data()["whee"]))
    right_combined = list(zip(mult.get_column_data()["stuff"], mult.get_column_data()["whee"]))
    counter = biocutils.match(left_combined, right_combined)
    assert (table(counter).as_list() == mult.get_column_data()["counts"]).all()
    assert (mult.get_metadata()["aggregated"]["index"] == counter).all()

    # Works if we disable the outputs.
    empty = scranpy.aggregate_across_cells_se(se, [se.get_column_data()["stuff"]], meta_name=None, counts_name=None, include_coldata=False)
    assert not empty.get_column_data().has_column("counts")
    assert "aggregated" not in empty.get_metadata()
    assert not empty.get_column_data().has_column("stuff")


def test_aggregate_across_cells_se_altexp():
    se = create_test_se()
    import singlecellexperiment
    sce = singlecellexperiment.SingleCellExperiment(se.get_assays(), column_data=se.get_column_data(), row_data=se.get_row_data())

    import biocframe
    is_even = range(0, sce.shape[0], 2)
    sce.set_alternative_experiment("GuP", se[is_even,:], in_place=True)
    replacement = ["FOO_" + y for y in sce.get_column_data()["stuff"]]
    sce = sce.set_column_data(sce.get_column_data().set_column("random", replacement))

    out = scranpy.aggregate_across_cells_se(sce, sce.get_column_data()[["stuff", "whee"]])
    assert not isinstance(out, singlecellexperiment.SingleCellExperiment)

    out = scranpy.aggregate_across_cells_se(sce, sce.get_column_data()[["stuff", "whee"]], altexps=["GuP"])
    assert isinstance(out, singlecellexperiment.SingleCellExperiment)
    assert (out.get_assay(0)[is_even,:] == out.get_alternative_experiment(0).get_assay(0)).all()
    assert (out.get_row_data()[is_even,:]["foo"] == out.get_alternative_experiment(0).get_row_data()["foo"]).all()
    assert out.get_column_data()["stuff"] == out.get_alternative_experiment(0).get_column_data()["stuff"]
    assert out.get_column_data()["random"] == ["FOO_" + y for y in out.get_column_data()["stuff"]]
    assert not out.get_alternative_experiment(0).get_column_data().has_column("factor_stuff")
    assert not out.get_alternative_experiment(0).get_column_data().has_column("factor_whee")
    assert not out.get_alternative_experiment(0).get_column_data().has_column("counts")
    assert not out.get_alternative_experiment(0).get_column_data().has_column("random")
    assert "aggregated" not in out.get_alternative_experiment(0).get_metadata()

    # Copying restores the extra outputs.
    out = scranpy.aggregate_across_cells_se(sce, sce.get_column_data()[["stuff", "whee"]], altexps=["GuP"], copy_altexps=True)
    assert isinstance(out, singlecellexperiment.SingleCellExperiment)
    assert out.get_alternative_experiment(0).get_column_data()["factor_stuff"] == out.get_column_data()["stuff"]
    assert out.get_alternative_experiment(0).get_column_data()["factor_whee"] == out.get_column_data()["whee"]
    assert (out.get_alternative_experiment(0).get_column_data()["counts"] == out.get_column_data()["counts"]).all()
    assert (out.get_alternative_experiment(0).get_metadata()["aggregated"]["index"] == out.get_metadata()["aggregated"]["index"]).all()


def test_aggregate_column_data():
    import biocframe
    df = biocframe.BiocFrame({
        "stuff": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4 + ["E"] * 4,
        "whee": [True, False] * 10
    })

    adf = scranpy.aggregate_column_data(df, biocutils.factorize(df["stuff"])[1], 5)
    assert adf["stuff"] == ["A", "B", "C", "D", "E"]
    assert adf["whee"] == [None] * 5

    adf = scranpy.aggregate_column_data(df, biocutils.factorize(df["whee"])[1], 2)
    assert adf["stuff"] == [None] * 2
    assert adf["whee"] == [True, False]

    # Still works as a factor.
    fdf = df.set_column("stuff", biocutils.Factor.from_sequence(df["stuff"]))
    adf = scranpy.aggregate_column_data(fdf, biocutils.factorize(df["stuff"])[1], 5)
    assert isinstance(adf["stuff"], biocutils.Factor)
    assert adf["stuff"] == biocutils.Factor.from_sequence(["A", "B", "C", "D", "E"])

    # Preserves list type.
    sdf = df.set_column("stuff", biocutils.StringList(df["stuff"]))
    adf = scranpy.aggregate_column_data(sdf, biocutils.factorize(df["stuff"])[1], 5)
    assert isinstance(adf["stuff"], biocutils.StringList)
    assert adf["stuff"] == biocutils.StringList(["A", "B", "C", "D", "E"])

    # Handles non-atomic, non-factor columns.
    ddf = df.set_column("foo", biocframe.BiocFrame({ "bar": numpy.random.rand(df.shape[0]) }))
    adf = scranpy.aggregate_column_data(ddf, biocutils.factorize(df["whee"])[1], 2)
    assert not adf.has_column("foo")
    adf = scranpy.aggregate_column_data(ddf, biocutils.factorize(df["whee"])[1], 2, only_simple=False)
    assert adf["foo"] == [None] * 2
