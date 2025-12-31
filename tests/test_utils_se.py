import scranpy._utils_se as seutils
import biocutils


def test_sanitize_altexp_assays():
    assert seutils.sanitize_altexp_assays(0, ["foo", "bar"], "whee") == { "foo": "whee" }
    assert seutils.sanitize_altexp_assays("stuff", ["foo", "bar"], "whee") == { "stuff": "whee" }
    assert seutils.sanitize_altexp_assays(["stuff", 1], ["foo", "bar"], "whee") == { "stuff": "whee", "bar": "whee" }
    assert seutils.sanitize_altexp_assays({ "stuff": 1, "yay": "counts" }, ["foo", "bar"], "whee") == { "stuff": 1, "yay": "counts" }
    assert seutils.sanitize_altexp_assays(biocutils.NamedList([0, "foo", 1]), ["foo", "bar"], "whee") == { "foo": "whee", "foo": "whee", "bar": "whee" }
    assert seutils.sanitize_altexp_assays(biocutils.NamedList([0, "foo", 1], ["x", "y", "z"]), ["foo", "bar"], "whee") == { "x": 0, "y": "foo", "z": 1 }
    assert seutils.sanitize_altexp_assays(biocutils.NamedList([0, "foo", 1], ["x", "x", "x"]), ["foo", "bar"], "whee") == { "x": 0 } # first occurrence only.
