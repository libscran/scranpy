import scranpy
import singlecellexperiment
import numpy


def test_run_all_neighbor_steps_se():
    sce = singlecellexperiment.SingleCellExperiment({ "counts": numpy.random.rand(0, 200) })
    sce = sce.set_reduced_dimension("PCA", numpy.random.rand(200, 10))

    out = scranpy.run_all_neighbor_steps_se(sce, num_threads=2)
    assert "TSNE" in out.get_reduced_dimension_names()
    assert "UMAP" in out.get_reduced_dimension_names()
    assert len(out.get_column_data()["clusters"]) == 200

    null = scranpy.run_all_neighbor_steps_se(sce, umap_output_name=None, tsne_output_name=None, cluster_output_name=None, num_threads=2)
    assert not "TSNE" in null.get_reduced_dimension_names()
    assert not "UMAP" in null.get_reduced_dimension_names()
    assert not null.get_column_data().has_column("clusters")

    # Checking that we can get the graph.
    wtg = scranpy.run_all_neighbor_steps_se(sce, umap_output_name=None, tsne_output_name=None, build_graph_name="graph", num_threads=2)
    assert not "TSNE" in wtg.get_reduced_dimension_names()
    assert not "UMAP" in wtg.get_reduced_dimension_names()
    assert len(wtg.get_column_data()["clusters"]) == 200
    assert "graph" in wtg.get_metadata()
