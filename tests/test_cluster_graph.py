import scranpy
import numpy
import copy
import biocutils


def test_cluster_graph_multilevel():
    data = numpy.random.randn(10, 1000)
    g = scranpy.build_snn_graph(data)

    clust = scranpy.cluster_graph(g, method="multilevel")
    assert (clust["membership"] >= 0).all()
    assert (clust["membership"] < 1000).all()

    for i, lev in enumerate(clust["levels"]):
        assert (lev >= 0).all()
        assert (lev < 1000).all()

    # Works without weights.
    ug = copy.copy(g)
    ug.weights = None
    uclust = scranpy.cluster_graph(ug, method="multilevel")
    assert (uclust["membership"] >= 0).all()
    assert (uclust["membership"] < 1000).all()


def test_cluster_graph_leiden():
    data = numpy.random.randn(10, 1000)
    g = scranpy.build_snn_graph(data)

    clust = scranpy.cluster_graph(g, method="leiden")
    assert (clust["membership"] >= 0).all()
    assert (clust["membership"] < 1000).all()

    # Works without weights.
    ug = copy.copy(g)
    ug.weights = None
    uclust = scranpy.cluster_graph(ug, leiden_objective="cpm", method="leiden")
    assert (uclust["membership"] >= 0).all()
    assert (uclust["membership"] < 1000).all()


def test_cluster_graph_walktrap():
    data = numpy.random.randn(10, 1000)
    g = scranpy.build_snn_graph(data)

    clust = scranpy.cluster_graph(g, method="walktrap")
    assert (clust["membership"] >= 0).all()
    assert (clust["membership"] < 1000).all()

    # Works without weights.
    ug = copy.copy(g)
    ug.weights = None
    uclust = scranpy.cluster_graph(ug, method="walktrap")
    assert (uclust["membership"] >= 0).all()
    assert (uclust["membership"] < 1000).all()


def test_cluster_graph_se():
    import singlecellexperiment
    sce = singlecellexperiment.SingleCellExperiment({ "counts": numpy.random.rand(0, 200) })
    sce.set_reduced_dimension("PCA", numpy.random.rand(200, 10), in_place=True)

    sce = scranpy.cluster_graph_se(sce)
    assert len(sce.get_column_data()["clusters"]) == 200

    sce2 = scranpy.cluster_graph_se(sce, resolution=2)
    assert not (sce2.get_column_data()["clusters"] == sce.get_column_data()["clusters"]).all()

    sce = scranpy.cluster_graph_se(sce, graph_name="graph", meta_name="cluster")
    assert isinstance(sce.get_metadata()["cluster"], biocutils.NamedList)
    assert isinstance(sce.get_metadata()["graph"], biocutils.NamedList)
