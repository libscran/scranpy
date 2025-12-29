import numpy
import scranpy


def test_summarize_effects():
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(100) * 4).astype(numpy.int32)
    summ = scranpy.score_markers(x, g, min_rank_limit=x.shape[0])
    full = scranpy.score_markers(x, g, all_pairwise=True)

    csumm = scranpy.summarize_effects(full["cohens_d"])
    assert (csumm[0]["min"] == summ["cohens_d"][0]["min"]).all()
    assert (csumm[1]["mean"] == summ["cohens_d"][1]["mean"]).all()
    assert (csumm[2]["median"] == summ["cohens_d"][2]["median"]).all()
    assert (csumm[3]["max"] == summ["cohens_d"][3]["max"]).all()
    assert (csumm[0]["min_rank"] == summ["cohens_d"][0]["min_rank"]).all()

    asumm = scranpy.summarize_effects(full["auc"], num_threads=2)
    assert (asumm[0]["min"] == summ["auc"][0]["min"]).all()
    assert (asumm[1]["mean"] == summ["auc"][1]["mean"]).all()
    assert (asumm[2]["median"] == summ["auc"][2]["median"]).all()
    assert (asumm[3]["max"] == summ["auc"][3]["max"]).all()
    assert (asumm[0]["min_rank"] == summ["auc"][0]["min_rank"]).all()


def test_summarize_effects_quantiles():
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(100) * 4).astype(numpy.int32)
    full = scranpy.score_markers(x, g, all_pairwise=True)

    summ = scranpy.summarize_effects(full["auc"], compute_quantiles=[0, 0.5, 1])
    for y in summ:
        assert (y["min"] == y["quantile"]["0.0"]).all()
        assert numpy.allclose(y["median"], y["quantile"]["0.5"])
        assert (y["max"] == y["quantile"]["1.0"]).all()


def test_summarize_effects_empty():
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(100) * 4).astype(numpy.int32)
    full = scranpy.score_markers(x, g, all_pairwise=True)

    summ = scranpy.summarize_effects(full["auc"], compute_min=False, compute_mean=False, compute_median=False, compute_max=False, compute_min_rank=False)
    for y in summ:
        assert y.shape[0] == x.shape[0]
        assert not y.has_column("min")
        assert not y.has_column("mean")
        assert not y.has_column("median") 
        assert not y.has_column("max") 
        assert not y.has_column("min_rank")
