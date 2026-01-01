import numpy
import scranpy


def test_fit_variance_trend():
    x = numpy.random.rand(1000)
    y = 2**numpy.random.randn(1000)
    out = scranpy.fit_variance_trend(x, y)
    assert numpy.allclose(y - out["fitted"], out["residual"])

    # Works with parallellization.
    out2 = scranpy.fit_variance_trend(x, y, num_threads=2)
    assert (out2["fitted"] == out["fitted"]).all()
    assert (out2["residual"] == out["residual"]).all()

    # Responds to the various options.
    out2["fitted"], resids2 = scranpy.fit_variance_trend(x, y, use_min_width=True, min_width=0.5)
    assert (out["fitted"] != out2["fitted"]).any()

    out2["fitted"], resids2 = scranpy.fit_variance_trend(x, y, transform=False)
    assert (out["fitted"] != out2["fitted"]).any()

    out2["fitted"], resids2 = scranpy.fit_variance_trend(x, y, span=0.5)
    assert (out["fitted"] != out2["fitted"]).any()
