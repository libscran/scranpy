import numpy
import scranpy
import biocutils
import biocframe
import summarizedexperiment
import singlecellexperiment 


def create_test_se():
    mat = numpy.random.rand(0, 200)
    sce = singlecellexperiment.SingleCellExperiment({ "counts": mat })
    sce = sce.set_reduced_dimension("PCA1", numpy.random.rand(200, 5))
    sce = sce.set_reduced_dimension("PCA2", numpy.random.rand(200, 6))

    sce2 = singlecellexperiment.SingleCellExperiment({ "counts": mat })
    sce2 = sce2.set_reduced_dimension("PCA1", numpy.random.rand(200, 7))
    sce2 = sce2.set_reduced_dimension("PCA2", numpy.random.rand(200, 4))

    sce3 = singlecellexperiment.SingleCellExperiment({ "counts": mat })
    sce3 = sce3.set_reduced_dimension("PCA1", numpy.random.rand(200, 8))
    sce3 = sce3.set_reduced_dimension("PCA2", numpy.random.rand(200, 3))

    sce = sce.set_alternative_experiment("ADT", sce2)
    sce = sce.set_alternative_experiment("other", sce2)
    return sce


def test_scale_by_neighbors_se_basic():
    sce = create_test_se()
    out = scranpy.scale_by_neighbors_se(
        sce,
        altexp_reddims={
            "ADT": ["PCA2", "PCA1"],
            "other": ["PCA1", "PCA2"]
        },
        main_reddims=["PCA1", "PCA2"]
    )

    ref = scranpy.scale_by_neighbors(
        [
            sce.get_reduced_dimension("PCA1").transpose(),
            sce.get_reduced_dimension("PCA2").transpose(),
            sce.get_alternative_experiment("ADT").get_reduced_dimension("PCA2").transpose(),
            sce.get_alternative_experiment("ADT").get_reduced_dimension("PCA1").transpose(),
            sce.get_alternative_experiment("other").get_reduced_dimension("PCA1").transpose(),
            sce.get_alternative_experiment("other").get_reduced_dimension("PCA2").transpose()
        ]
    )
    assert (out.get_reduced_dimension("combined") == ref["combined"].transpose()).all()

    meta = out.get_metadata()["combined"]
    assert len(meta["main_scaling"]) == 2
    assert len(meta["altexp_scaling"]) == 2
    assert len(meta["altexp_scaling"]["ADT"]) == 2
    assert len(meta["altexp_scaling"]["other"]) == 2

    expected_scaling = [
        meta["main_scaling"]["PCA1"],
        meta["main_scaling"]["PCA2"],
        meta["altexp_scaling"]["ADT"]["PCA2"],
        meta["altexp_scaling"]["ADT"]["PCA1"],
        meta["altexp_scaling"]["other"]["PCA1"],
        meta["altexp_scaling"]["other"]["PCA2"]
    ]

    assert list(ref["scaling"]) == expected_scaling


def test_scale_by_neighbors_se_edge_cases():
    sce = create_test_se()

    # Main experiment only.
    out = scranpy.scale_by_neighbors_se(sce, altexp_reddims={}, main_reddims=["PCA2", "PCA1"])
    ref = scranpy.scale_by_neighbors([
        sce.get_reduced_dimension("PCA2").transpose(),
        sce.get_reduced_dimension("PCA1").transpose()
    ])
    assert (out.get_reduced_dimension("combined") == ref["combined"].transpose()).all()
    assert out.get_metadata()["combined"]["main_scaling"].get_names().as_list() == ["PCA2", "PCA1"]
    assert out.get_metadata()["combined"]["main_scaling"].as_list() == list(ref["scaling"])
    assert len(out.get_metadata()["combined"]["altexp_scaling"]) == 0

    # Alternative experiments only.
    out = scranpy.scale_by_neighbors_se(sce, altexp_reddims={ "ADT": "PCA2", "other": "PCA2" }, main_reddims=None)
    ref = scranpy.scale_by_neighbors([
        sce.get_alternative_experiment("ADT").get_reduced_dimension("PCA2").transpose(),
        sce.get_alternative_experiment("other").get_reduced_dimension("PCA2").transpose()
    ])
    assert (out.get_reduced_dimension("combined") == ref["combined"].transpose()).all()
    assert len(out.get_metadata()["combined"]["main_scaling"]) == 0
    assert len(out.get_metadata()["combined"]["altexp_scaling"]["ADT"]) == 1
    assert len(out.get_metadata()["combined"]["altexp_scaling"]["other"]) == 1
    expected = [
        out.get_metadata()["combined"]["altexp_scaling"]["ADT"]["PCA2"],
        out.get_metadata()["combined"]["altexp_scaling"]["other"]["PCA2"]
    ]
    assert expected == list(ref["scaling"])

    # Nothing. 
    out = scranpy.scale_by_neighbors_se(sce, altexp_reddims={ "ADT": "PCA2", "other": "PCA2" }, main_reddims="PCA1", meta_name=None)
    assert "combined" not in out.get_metadata()


def test_scale_by_neighbors_se_dedup():
    sce = create_test_se()

    out = scranpy.scale_by_neighbors_se(sce, altexp_reddims=biocutils.NamedList([["PCA1", "PCA1"], ["PCA2", "PCA2"]], names=["ADT", "ADT"]), main_reddims=["PCA2", "PCA2"])
    ref = scranpy.scale_by_neighbors_se(sce, altexp_reddims={ "ADT": "PCA1" }, main_reddims="PCA2")

    assert (out.get_reduced_dimension("combined") == ref.get_reduced_dimension("combined")).all()
    assert out.get_metadata()["combined"] == ref.get_metadata()["combined"]
