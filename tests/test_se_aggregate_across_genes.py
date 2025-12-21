import random
import numpy
import biocutils
import biocframe
import summarizedexperiment
import scranpy


def create_test_se():
    mat = numpy.random.rand(100, 10)
    se = summarizedexperiment.SummarizedExperiment({ "logcounts": mat })
    se.set_column_data(
        biocframe.BiocFrame({
            "stuff": random.choices(["A", "B", "C", "D", "E"], k=se.shape[1]),
            "whee": random.choices([True, False], k=se.shape[1])
        }),
        in_place = True
    )
    return se


def create_test_sets(n):
    return biocutils.NamedList.from_dict({
        "foo": random.sample(range(n), 10),
        "bar": random.sample(range(n), 20),
        "stuff": random.sample(range(n), 30)
    })


def test_aggregate_across_genes_se_basic():
    se = create_test_se()
    sets = create_test_sets(se.shape[0])

    out = scranpy.aggregate_across_genes_se(se, sets)
    assert out.shape == (3, se.shape[1])
    assert out.get_assay_names() == ["logcounts"]
    assert out.get_row_names().as_list() == ["foo", "bar", "stuff"]
    assert out.get_column_data().get_column_names().as_list() == ["stuff", "whee"]

    # Checking the names.
    out2 = scranpy.aggregate_across_genes_se(se, sets, assay_type=0)
    assert out2.get_assay_names() == ["aggregated"]
    out2 = scranpy.aggregate_across_genes_se(se, sets, assay_type=0, output_name="FOO")
    assert out2.get_assay_names() == ["FOO"]

    # Works with empty inputs.
    zeroed = scranpy.aggregate_across_genes_se(se, [])
    assert zeroed.shape[0] == 0
    assert zeroed.get_row_names() is None

    # Works with DataFrame inputs.
    dfsets = create_test_sets(se.shape[0])
    for i, val in enumerate(dfsets):
        dfsets[i] = biocframe.BiocFrame({ "index": val, "weights": numpy.random.rand(len(val)) })
    dfout = scranpy.aggregate_across_genes_se(se, dfsets)
    assert dfout.shape == (3, se.shape[1])


