import scranpy
import numpy
import pytest


def test_correct_mnn_simple():
    numpy.random.seed(696969)
    x = numpy.asfortranarray(numpy.random.randn(10, 1000))
    block = ["A"] * 500 + ["B"] * 300 + ["C"] * 200
    x[:,[i == "B" for i in block]] += 3
    x[:,[i == "C" for i in block]] += 5

    Ameans = x[:,[i == "A" for i in block]].mean()
    Bmeans = x[:,[i == "B" for i in block]].mean()
    Cmeans = x[:,[i == "C" for i in block]].mean()
    assert Bmeans > Ameans + 2
    assert Cmeans > Bmeans + 1

    res = scranpy.correct_mnn(x, block)
    assert res["corrected"].shape == (10, 1000)

    Ameans = res["corrected"][:,[i == "A" for i in block]].mean()
    Bmeans = res["corrected"][:,[i == "B" for i in block]].mean()
    Cmeans = res["corrected"][:,[i == "C" for i in block]].mean()
    assert abs(Bmeans - Ameans) < 0.5
    assert abs(Cmeans - Bmeans) < 0.5
