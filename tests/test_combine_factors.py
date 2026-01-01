import scranpy
import numpy
import biocutils
import biocframe


def test_combine_factors_single():
    numpy.random.rand(10001)
    levels = ["A", "B", "C", "D", "E"]
    x = [levels[i] for i in (numpy.random.rand(100) * len(levels)).astype(numpy.int32)]

    out = scranpy.combine_factors((x,))
    outcomb = out["levels"]
    outind = out["index"]

    assert outcomb.shape[1] == 1
    assert outcomb[0] == levels
    assert [levels[i] for i in outind] == x

    # Drops unused elements.
    x = biocutils.Factor.from_sequence(["A", "D", "F", "B"], levels=["A", "B", "C", "D", "E", "F", "G"])
    out = scranpy.combine_factors((x,))
    outcomb = out["levels"]
    outind = out["index"]

    assert outcomb.shape[1] == 1
    expected = ["A", "B", "D", "F"]
    assert outcomb[0].as_list() == expected
    assert [expected[i] for i in outind] == list(x)

    # Unless we request to keep them.
    out = scranpy.combine_factors((x,), keep_unused=True)
    outcomb = out["levels"]
    outind = out["index"]

    assert outcomb.shape[1] == 1
    expected = ["A", "B", "C", "D", "E", "F", "G"]
    assert outcomb[0].as_list() == expected
    assert [expected[i] for i in outind] == list(x)


def test_combine_factors_multiple():
    numpy.random.rand(10002)
    upper_levels = ["A", "B", "C", "D", "E"]
    x = [upper_levels[i] for i in (numpy.random.rand(1000) * len(upper_levels)).astype(numpy.int32)]
    y = (numpy.random.rand(1000) * 3 + 10).astype(numpy.int32)
    lower_levels = ["x", "y", "z"]
    z = [lower_levels[i] for i in (numpy.random.rand(1000) * len(lower_levels)).astype(numpy.int32)]

    out = scranpy.combine_factors((x, y, z))
    outcomb = out["levels"]
    outind = out["index"]

    assert outcomb.shape[1] == 3
    assert len(outcomb[0]) == 45
    assert [outcomb[0][i] for i in outind] == x
    assert [outcomb[1][i] for i in outind] == list(y)
    assert [outcomb[2][i] for i in outind] == z

    # Also accepts a BiocFrame.
    dfout = scranpy.combine_factors(biocframe.BiocFrame({"X": x, "Y": y, "Z": z}))
    assert dfout["levels"].get_column_names().as_list() == ["X", "Y", "Z"]
    assert dfout["levels"][0] == outcomb[0]
    assert dfout["levels"][1] == outcomb[1]
    assert dfout["levels"][2] == outcomb[2]
    assert (dfout["index"] == out["index"]).all()


def test_combine_factors_multiple_unused():
    numpy.random.rand(10003)
    x = ["A","B","C","D","E"]
    y = [1,2,3,1,2]
    z = ["x", "x", "y", "y", "z"]

    # Sanity check.
    out = scranpy.combine_factors((x, y, z))
    outcomb = out["levels"]
    outind = out["index"]

    assert outcomb.shape[1] == 3
    assert len(outcomb[0]) < 45
    assert [outcomb[0][i] for i in outind] == x
    assert [outcomb[1][i] for i in outind] == list(y)
    assert [outcomb[2][i] for i in outind] == z

    # With unused.
    out = scranpy.combine_factors((x, y, z), keep_unused=True)
    outcomb = out["levels"]
    outind = out["index"]

    assert outcomb.shape[1] == 3
    assert len(outcomb[0]) == 45
    assert [outcomb[0][i] for i in outind] == x
    assert [outcomb[1][i] for i in outind] == list(y)
    assert [outcomb[2][i] for i in outind] == z
