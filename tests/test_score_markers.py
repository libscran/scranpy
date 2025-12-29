import scranpy
import numpy
import biocframe


def _check_summaries(summary):
    for x in summary:
        assert (x["min"] <= x["max"]).all()
        assert (x["min"] <= x["mean"] + 1e-8).all() # add some tolerance for numerical imprecision when averaging identical effects.
        assert (x["min"] <= x["median"]).all()
        assert (x["mean"] <= x["max"] + 1e-8).all()
        assert (x["median"] <= x["max"]).all()
        assert (x["min_rank"] <= 1000).all()
        assert (x["min_rank"] < 1000).sum() >= 500 # default min_rank_limit.
        assert (x["min_rank"] >= 0).all()


def test_score_markers_simple():
    numpy.random.seed(42)
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(x.shape[1]) * 4).astype(numpy.int32)
    out = scranpy.score_markers(x, g)

    assert out["num_genes"] == 1000
    assert out["group_ids"] == [0,1,2,3]
    assert numpy.allclose(out["mean"][:,0], x[:,g==0].mean(axis=1))
    assert numpy.allclose(out["detected"][:,3], (x[:,g==3] > 0).mean(axis=1))

    _check_summaries(out["cohens_d"])
    _check_summaries(out["auc"])
    _check_summaries(out["delta_mean"])
    _check_summaries(out["delta_detected"])

    for aeff in out["auc"]:
        assert (aeff["min"] >= 0).all() and (aeff["min"] <= 1).all()
        assert (aeff["mean"] >= 0).all() and (aeff["mean"] <= 1).all()
        assert (aeff["median"] >= 0).all() and (aeff["median"] <= 1).all()
        assert (aeff["max"] >= 0).all() and (aeff["max"] <= 1).all()

    # Works with multiple threads.
    pout = scranpy.score_markers(x, g, num_threads=2)
    assert (out["mean"] == pout["mean"]).all()
    assert (out["detected"] == pout["detected"]).all()
    assert (out["cohens_d"][0]["mean"] == pout["cohens_d"][0]["mean"]).all()
    assert (out["delta_detected"][1]["median"] == pout["delta_detected"][1]["median"]).all()
    assert (out["delta_mean"][2]["max"] == pout["delta_mean"][2]["max"]).all()
    assert (out["auc"][3]["min_rank"] == pout["auc"][3]["min_rank"]).all()


def test_score_markers_quantiles():
    numpy.random.seed(42)
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(x.shape[1]) * 4).astype(numpy.int32)
    out = scranpy.score_markers(x, g, compute_summary_quantiles=[0, 0.5, 1])

    for eff in ["cohens_d", "auc", "delta_mean", "delta_detected"]:
        summary = out[eff]
        for x in summary:
            assert (x["min"] == x["quantile"]["0.0"]).all()
            assert (x["max"] == x["quantile"]["1.0"]).all()
            assert numpy.allclose(x["median"], x["quantile"]["0.5"])


def test_score_markers_empty():
    numpy.random.seed(42)
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(x.shape[1]) * 4).astype(numpy.int32)
    ref = scranpy.score_markers(x, g)

    # Works without the AUC.
    aout = scranpy.score_markers(x, g, compute_auc=False)
    assert "auc" not in aout.get_names()
    assert "cohens_d" in aout.get_names()
    assert (aout["mean"] == ref["mean"]).all()
    assert (aout["detected"] == ref["detected"]).all()

    # Works without any effect sizes.
    empty = scranpy.score_markers(x, g, compute_auc=False, compute_cohens_d=False, compute_delta_detected=False, compute_delta_mean=False)
    assert "auc" not in empty.get_names()
    assert "cohens_d" not in empty.get_names()
    assert "delta_mean" not in empty.get_names()
    assert "delta_detected" not in empty.get_names()
    assert (empty["mean"] == ref["mean"]).all()
    assert (empty["detected"] == ref["detected"]).all()

    # Works without any summaries.
    empty2 = scranpy.score_markers(x, g,
        compute_group_mean=False,
        compute_group_detected=False,
        compute_summary_min=False,
        compute_summary_mean=False,
        compute_summary_max=False,
        compute_summary_min_rank=False,
        compute_summary_median=False
    )
    assert not empty2["auc"]['0'].has_column("min")
    assert not empty2["auc"]['0'].has_column("min_rank")
    assert not empty2["cohens_d"]['1'].has_column("mean")
    assert not empty2["delta_mean"]['2'].has_column("median")
    assert not empty2["delta_detected"]['3'].has_column("max")
    assert "mean" not in empty2.get_names()
    assert "detected" not in empty2.get_names()


def test_score_markers_blocked():
    numpy.random.seed(421)
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(x.shape[1]) * 4).astype(numpy.int32)
    b = (numpy.random.rand(x.shape[1]) * 3).astype(numpy.int32)
    out = scranpy.score_markers(x, g, block=b, block_weight_policy="equal")

    bkeep = (g == 2)
    assert numpy.allclose(out["mean"][:,2], (
        x[:,numpy.logical_and(bkeep, b == 0)].mean(axis=1) + 
        x[:,numpy.logical_and(bkeep, b == 1)].mean(axis=1) + 
        x[:,numpy.logical_and(bkeep, b == 2)].mean(axis=1)
    )/3)
    ckeep = (g == 3)
    assert numpy.allclose(out["mean"][:,3], (
        x[:,numpy.logical_and(ckeep, b == 0)].mean(axis=1) + 
        x[:,numpy.logical_and(ckeep, b == 1)].mean(axis=1) + 
        x[:,numpy.logical_and(ckeep, b == 2)].mean(axis=1)
    )/3)

    _check_summaries(out["cohens_d"])
    _check_summaries(out["auc"])
    _check_summaries(out["delta_mean"])
    _check_summaries(out["delta_detected"])

    for aeff in out["auc"]:
        assert (aeff["min"] >= 0).all() and (aeff["min"] <= 1).all()
        assert (aeff["mean"] >= 0).all() and (aeff["mean"] <= 1).all()
        assert (aeff["median"] >= 0).all() and (aeff["median"] <= 1).all()
        assert (aeff["max"] >= 0).all() and (aeff["max"] <= 1).all()


def test_score_markers_pairwise():
    numpy.random.seed(422)
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(x.shape[1]) * 4).astype(numpy.int32)
    full = scranpy.score_markers(x, g, all_pairwise=True)

    # Checking that we set the dimensions correctly.
    for g1 in range(4):
        assert (full["delta_mean"][g1,g1,:] == 0).all()
        assert (full["auc"][g1,g1,:] == 0).all()
        for g2 in range(g1):
            assert numpy.allclose(full["delta_mean"][g1,g2,:], -full["delta_mean"][g2,g1,:])
            assert numpy.allclose(full["auc"][g1,g2,:], 1 - full["auc"][g2,g1,:])

    assert (full["auc"] >= 0).all()
    assert (full["auc"] <= 1).all()

    # Works without AUCs.
    aout = scranpy.score_markers(x, g, all_pairwise=True, compute_auc=False)
    assert "auc" not in aout.get_names()
    assert (aout["mean"] == full["mean"]).all()
    assert (aout["detected"] == full["detected"]).all()

    # Works without anything.
    empty  = scranpy.score_markers(x, g, compute_auc=False, compute_cohens_d=False, compute_delta_detected=False, compute_delta_mean=False, all_pairwise=True)
    assert "auc" not in empty.get_names()
    assert "cohens_d" not in empty.get_names()
    assert "delta_mean" not in empty.get_names()
    assert "delta_detected" not in empty.get_names()
    assert (empty["mean"] == full["mean"]).all()
    assert (empty["detected"] == full["detected"]).all()

    # Works with blocking.
    b = (numpy.random.rand(x.shape[1]) * 3).astype(numpy.int32)
    bout = scranpy.score_markers(x, g, block=b, block_weight_policy="equal", all_pairwise=True)
    sbout = scranpy.score_markers(x, g, block=b, block_weight_policy="equal")
    assert (bout["mean"] == sbout["mean"]).all()
    assert (bout["detected"] == sbout["detected"]).all()


def test_score_markers_best():
    numpy.random.seed(422)
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(x.shape[1]) * 4).astype(numpy.int32)
    full = scranpy.score_markers(x, g, all_pairwise=True)

    def convert_to_best(observed, pairwise, n, bound):
        nlabels = pairwise.shape[0]
        output = [None] * nlabels
        assert len(observed) == nlabels

        for g1 in range(nlabels):
            curobs = observed[g1]
            assert len(curobs) == nlabels
            current = [None] * nlabels

            for g2 in range(nlabels):
                curtop = curobs[g2]
                if g1 == g2:
                    assert curtop is None
                    continue

                stats = pairwise[g2, g1, :] # remember, second dimension is the first group in the comparison.
                keep = numpy.where(stats > bound)[0]
                o = sorted(zip(-stats[keep], keep)) # don't use reverse= as we want a stable sort on the index in case of ties.
                if n < len(o):
                    o = o[:n]

                refbest = numpy.array([p[1] for p in o], dtype=numpy.dtype("uint32"))
                refstat = numpy.array([-p[0] for p in o], dtype=numpy.dtype("double")) # -1 to cancel out the previous operation.
                assert (refbest == curtop.column("index")).all()
                assert (refstat == curtop.column("effect")).all()

    best = scranpy.score_markers(x, g, all_pairwise=10)
    assert (best["mean"] == full["mean"]).all()
    assert (best["detected"] == full["detected"]).all()
    convert_to_best(best["cohens_d"], full["cohens_d"], 10, 0)
    convert_to_best(best["auc"], full["auc"], 10, 0.5)
    convert_to_best(best["delta_mean"], full["delta_mean"], 10, 0)
    convert_to_best(best["delta_detected"], full["delta_detected"], 10, 0)

    # Works without AUCs and the groupwise means.
    aout = scranpy.score_markers(x, g, all_pairwise=10, compute_auc=False, compute_group_mean=False, compute_group_detected=False)
    assert "auc" not in aout.get_names()
    assert "mean" not in aout.get_names()
    assert "detected" not in aout.get_names()
    convert_to_best(best["cohens_d"], full["cohens_d"], 10, 0)

    # Works with blocking.
    b = (numpy.random.rand(x.shape[1]) * 3).astype(numpy.int32)
    bout = scranpy.score_markers(x, g, block=b, block_weight_policy="equal", all_pairwise=10)
    sbout = scranpy.score_markers(x, g, block=b, block_weight_policy="equal")
    assert (bout["mean"] == sbout["mean"]).all()
    assert (bout["detected"] == sbout["detected"]).all()
