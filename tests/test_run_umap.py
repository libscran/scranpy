import numpy
import knncolle
import scranpy
import warnings
import pytest


def test_run_umap():
    x = numpy.random.rand(10, 500)

    embed = scranpy.run_umap(x)
    assert embed.shape == (2, 500)

    again = scranpy.run_umap(x)
    assert (embed == again).all() # check that it's reproducible.

    alt = scranpy.run_umap(x, num_neighbors=20)
    assert alt.shape == (2, 500)
    assert (alt != embed).any() # check that perplexity has an effect.

    idx = knncolle.build_index(knncolle.AnnoyParameters(), x.T)
    res = knncolle.find_knn(idx, num_neighbors=15)
    nnin = scranpy.run_umap(res)
    assert (nnin == embed).all()


def test_run_umap_initialize():
    x = numpy.random.rand(10, 500)
    ref = scranpy.run_umap(x)

    with pytest.raises(Exception, match="initial coordinates"):
        scranpy.run_umap(x, initialize_method="none")

    init = numpy.random.rand(2, 500)
    alt = scranpy.run_umap(x, initialize_method="none", initial_coordinates=init)
    assert ref.shape == alt.shape
    assert not (alt == ref).all()

