import scranpy
import numpy
import singlecellexperiment


def test_cluster_kmeans_se():
    sce = singlecellexperiment.SingleCellExperiment({ "counts": numpy.random.rand(0, 200) })
    sce = sce.set_reduced_dimension("PCA", numpy.random.rand(200, 5))

    out = scranpy.cluster_kmeans_se(sce, k=2)
    assert len(set(out.get_column_data()["clusters"])) == 2

    out = scranpy.cluster_kmeans_se(sce, k=10, meta_name="kmeans")
    assert len(set(out.get_column_data()["clusters"])) == 10
    assert out.get_metadata()["kmeans"]["centers"].shape[1] == 10
