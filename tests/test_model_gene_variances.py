import scranpy
import numpy

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def test_model_gene_variances_basic():
    x = numpy.random.rand(1000, 100) * 10
    out = scranpy.model_gene_variances(x)["statistics"]

    assert out.shape == (1000, 4)
    assert numpy.allclose(out["mean"], x.mean(axis=1))
    assert numpy.allclose(out["variance"], x.var(axis=1, ddof=1))
    assert numpy.allclose(out["variance"] - out["fitted"], out["residual"])
    assert not out.has_column("per_block")

    fitted, resid = scranpy.fit_variance_trend(out["mean"], out["variance"])
    assert (out["fitted"] == fitted).all()
    assert (out["residual"] == resid).all()

    # Responds to trend-fitting options. 
    out2 = scranpy.model_gene_variances(x, span=0.5)["statistics"]
    assert (out["variance"] == out2["variance"]).all()
    assert (out["mean"] == out2["mean"]).all()
    assert (out["fitted"] != out2["fitted"]).any()

    fitted2, resid2 = scranpy.fit_variance_trend(out["mean"], out["variance"], span=0.5)
    assert (out2["fitted"] == fitted2).all()
    assert (out2["residual"] == resid2).all()


def test_model_gene_variances_blocked():
    numpy.random.seed(52)
    x = numpy.random.rand(1000, 100) * 10
    block = (numpy.random.rand(x.shape[1]) * 3).astype(numpy.int32)
    out = scranpy.model_gene_variances(x, block=block, block_weight_policy="equal")

    aveout = out["statistics"]
    pbout = out["per_block"]
    for b in range(3):
        sub = x[:,block == b]
        current = pbout[b]
        assert numpy.allclose(current["mean"], sub.mean(axis=1))
        assert numpy.allclose(current["variance"], sub.var(axis=1, ddof=1))
        assert numpy.allclose(current["variance"] - current["fitted"], current["residual"])

    assert out["block_ids"] == [0, 1, 2]
    assert pbout.get_names().as_list() == ["0", "1", "2"]
    assert numpy.allclose(aveout["mean"], (pbout[0]["mean"] + pbout[1]["mean"] + pbout[2]["mean"])/3)
    assert numpy.allclose(aveout["variance"], (pbout[0]["variance"] + pbout[1]["variance"] + pbout[2]["variance"])/3)
    assert numpy.allclose(aveout["fitted"], (pbout[0]["fitted"] + pbout[1]["fitted"] + pbout[2]["fitted"])/3)
    assert numpy.allclose(aveout["residual"], (pbout[0]["residual"] + pbout[1]["residual"] + pbout[2]["residual"])/3)
