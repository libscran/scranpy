import numpy
import delayedarray
import summarizedexperiment 
import singlecellexperiment
import scranpy
import pytest

from scranpy.se_analyze import _delayify_assays, _define_single_target_embedding, _add_source_embedding_to_scale


se = None
default = None

def get_test_se():
    global se
    if se is None:
        numpy.random.seed(69)
        mean = numpy.random.rand(100) * 5
        mat = numpy.ndarray((len(mean), 100))
        for i in range(mat.shape[1]):
            mat[:,i] = numpy.random.poisson(lam=mean, size=len(mean))
        se = summarizedexperiment.SummarizedExperiment({ "counts": mat })
        se.set_row_names(["GENE_" + str(i) for i in range(mat.shape[0])], in_place=True)
    return se


def get_default_analysis():
    global default
    if default is None:
        se = get_test_se()
        default = scranpy.analyze_se(
            se,
            more_tsne_args={ "max_iterations": 10 },
            more_umap_args={ "num_epochs": 5 },
            num_threads=1
        )
    return default


def test_analyze_se_basic():
    se = get_test_se()
    res = get_default_analysis()

    assert res["x"].get_row_names() == se.get_row_names()
    assert res["x"].get_column_data()["keep"].all()
    assert not res["x"].get_column_data().has_column("combined_keep")

    assert isinstance(res["x"].get_assay("counts"), delayedarray.DelayedArray)
    assert isinstance(res["x"].get_assay("logcounts"), delayedarray.DelayedArray)

    assert numpy.issubdtype(res["x"].get_row_data()["residual"].dtype, numpy.dtype("float64"))

    assert "PCA" in res["x"].get_reduced_dimension_names()
    assert "TSNE" in res["x"].get_reduced_dimension_names()
    assert "UMAP" in res["x"].get_reduced_dimension_names()
    assert "combined" not in res["x"].get_reduced_dimension_names()
    assert "MNN" not in res["x"].get_reduced_dimension_names()

    clustering = res["x"].get_column_data()["graph_cluster"]
    print(clustering)
    assert res["markers"]["rna"].get_names().as_list() == [str(i) for i in sorted(list(set(clustering)))]


def test_analyze_se_no_filter():
    se = get_test_se()
    res = scranpy.analyze_se(
        se,
        more_tsne_args={ "max_iterations": 10 },
        filter_cells=False,
        more_umap_args={ "num_epochs": 5 },
        num_threads=1
    )
    assert res["x"].shape == se.shape
    assert isinstance(res["x"].get_assay(0), numpy.ndarray)


def test_analyze_se_kmeans():
    se = get_test_se()
    res = scranpy.analyze_se(
        se,
        more_tsne_args={ "max_iterations": 10 },
        more_umap_args={ "num_epochs": 5 },
        cluster_graph_output_name=None,
        kmeans_clusters=10,
        num_threads=1
    )
    assert not res["x"].get_column_data().has_column("graph_cluster")
    clustering = res["x"].get_column_data()["kmeans_cluster"]
    assert res["markers"]["rna"].get_names().as_list() == [str(i) for i in sorted(list(set(clustering)))]


def test_analyze_se_no_cluster():
    se = get_test_se()
    res = scranpy.analyze_se(
        se,
        more_tsne_args={ "max_iterations": 10 },
        more_umap_args={ "num_epochs": 5 },
        cluster_graph_output_name=None,
        num_threads=1
    )
    assert not res["x"].get_column_data().has_column("graph_cluster")
    assert not res["x"].get_column_data().has_column("kmeans_cluster")
    assert res["markers"] is None


def test_analyze_se_blocking():
    se = get_test_se()
    block = ["A", "B", "C", "D"] * (se.shape[1] // 4)

    res = scranpy.analyze_se(
        se,
        block=block,
        more_tsne_args={ "max_iterations": 10 },
        more_umap_args={ "num_epochs": 5 },
        num_threads=1
    )
    assert "MNN" in res["x"].get_reduced_dimensions()
    assert res["x"].get_column_data().has_column("block")
    assert res["x"].get_metadata()["qc"]["thresholds"]["sum"].get_names().as_list() == ["A", "B", "C", "D"]

    # Confirm that blocked coordinates are actually used downstream.
    downstream = scranpy.run_all_neighbor_steps(
        res["x"].get_reduced_dimension("MNN").transpose(),
        run_tsne_options={ "max_iterations": 10 },
        run_umap_options={ "num_epochs": 5 },
        num_threads=1
    )
    assert (downstream["cluster_graph"]["membership"] == res["x"].get_column_data()["graph_cluster"]).all()
    assert (downstream["run_tsne"] == res["x"].get_reduced_dimension("TSNE")).all()


def test_analyze_se_rna_adt():
    se = get_test_se()
    sce = singlecellexperiment.SingleCellExperiment(
        se.get_assays(),
        row_data=se.get_row_data(),
        column_data=se.get_column_data(),
        row_names=se.get_row_names()
    )

    numpy.random.rand(42)
    amat = numpy.random.poisson(lam=10, size=1000).reshape(10, 100)
    ase = summarizedexperiment.SummarizedExperiment({ "counts": amat })
    ase.set_row_names(["TAG_" + str(i) for i in range(amat.shape[0])], in_place=True)
    sce.set_alternative_experiment("ADT", ase, in_place=True)

    # Works with RNA + ADT. We use a different number of PCs for the ADTs to get some variety.
    res = scranpy.analyze_se(
        sce,
        adt_altexp="ADT",
        more_adt_pca_args={ "number": 10 },
        more_tsne_args={ "max_iterations": 10 },
        more_umap_args={ "num_epochs": 5 },
        num_threads=1
    )

    assert res["x"].get_alternative_experiment(0).get_row_names() == sce.get_alternative_experiment(0).get_row_names()
    assert numpy.issubdtype(res["x"].get_alternative_experiment("ADT").get_column_data()["keep"].dtype, numpy.bool)
    comkeep = res["x"].get_column_data()["combined_keep"]
    assert numpy.issubdtype(comkeep.dtype, numpy.bool)
    assert comkeep.all()

    altout = res["x"].get_alternative_experiment(0)
    assert isinstance(altout.get_assay("counts"), delayedarray.DelayedArray)
    assert isinstance(altout.get_assay("logcounts"), delayedarray.DelayedArray)
    assert not altout.get_row_data().has_column("hvg")
    assert altout.get_reduced_dimension("PCA").shape[1] == 10

    assert res["x"].get_row_data().has_column("hvg")
    assert res["x"].get_reduced_dimension("PCA").shape[1] == 25

    assert res["x"].get_reduced_dimension("combined").shape[1] == 35
    assert res["x"].get_metadata()["combined"]["main_scaling"].get_names().as_list() == ["PCA"]
    assert res["x"].get_metadata()["combined"]["altexp_scaling"]["ADT"].get_names().as_list() == ["PCA"]

    clustering = res["x"].get_column_data()["graph_cluster"]
    clustlev = [str(i) for i in sorted(list(set(clustering)))]
    assert res["markers"]["rna"].get_names().as_list() == clustlev
    assert res["markers"]["adt"].get_names().as_list() == clustlev

    # Also works with ADT + RNA where ADT is the main experiment.
    sce2 = singlecellexperiment.SingleCellExperiment(
        ase.get_assays(),
        row_data=ase.get_row_data(),
        column_data=ase.get_column_data(),
        row_names=ase.get_row_names(),
        alternative_experiments={ "RNA": se }
    )

    res2 = scranpy.analyze_se(
        sce2,
        rna_altexp="RNA",
        adt_altexp=False,
        more_tsne_args={ "max_iterations": 10 },
        more_umap_args={ "num_epochs": 5 },
        num_threads=1
    )
    assert not res2["x"].get_row_data().has_column("hvg")
    assert numpy.issubdtype(res2["x"].get_alternative_experiment(0).get_row_data()["hvg"].dtype, numpy.bool)

    assert (res2["x"].get_reduced_dimension("PCA") == res["x"].get_alternative_experiment("ADT").get_reduced_dimension("PCA")).all()
    assert (res2["x"].get_alternative_experiment("RNA").get_reduced_dimension("PCA") == res["x"].get_reduced_dimension("PCA")).all()
    assert res2["x"].get_reduced_dimension("combined").shape == res["x"].get_reduced_dimension("combined").shape

    assert res["markers"]["rna"].get_names() == res2["markers"]["rna"].get_names()
    assert (res["markers"]["rna"][0]["cohens_d_mean"] == res2["markers"]["rna"][0]["cohens_d_mean"]).all()
    assert res["markers"]["adt"].get_names() == res2["markers"]["adt"].get_names()
    assert (res["markers"]["adt"][1]["auc_median"] == res2["markers"]["adt"][1]["auc_median"]).all()

    # Confirm that scaled coordinates are actually used downstream.
    ref = scranpy.run_all_neighbor_steps(
        res["x"].get_reduced_dimension("combined").transpose(),
        run_tsne_options={ "max_iterations": 10 },
        run_umap_options= { "num_epochs": 5 },
        num_threads=1
    )
    assert (ref["cluster_graph"]["membership"] == res["x"].get_column_data()["graph_cluster"]).all()
    assert (ref["run_tsne"] == res["x"].get_reduced_dimension("TSNE")).all()

    # Fails if neither ADT or RNA is supplied.
    with pytest.raises(Exception, match="at least one"):
        scranpy.analyze_se(sce, use_rna_pcs=False, use_adt_pcs=False)


def test_analyze_se_adt_only():
    numpy.random.rand(42)
    amat = numpy.random.poisson(lam=10, size=1000).reshape(10, 100)
    ase = summarizedexperiment.SummarizedExperiment({ "counts": amat })
    ase.set_row_names(["TAG_" + str(i) for i in range(amat.shape[0])], in_place=True)

    # Works with ADT in the main experiment.
    res = scranpy.analyze_se(
        ase,
        rna_altexp=None,
        adt_altexp=False,
        more_tsne_args={ "max_iterations": 10 },
        more_umap_args={ "num_epochs": 5 },
        num_threads=1
    )
    assert numpy.issubdtype(res["x"].get_column_data()["keep"].dtype, numpy.bool)
    assert "PCA" in res["x"].get_reduced_dimension_names()
    assert "combined" not in res["x"].get_reduced_dimension_names()
    assert res["markers"].get_names().as_list() == ["adt"]

    # Works after ignoring RNA in the main experiment.
    # This checks that the use of a named character vector for 'target.embedding' is respected within analyze.se().
    sce = singlecellexperiment.SingleCellExperiment({ "counts": numpy.random.rand(0, 100) })
    sce.set_alternative_experiment("ADT", ase, in_place=True)

    res2 = scranpy.analyze_se(
        sce,
        rna_altexp=None,
        adt_altexp="ADT",
        more_tsne_args={ "max_iterations": 10 },
        more_umap_args={ "num_epochs": 5 },
        num_threads=1
    )
    assert (res2["x"].get_alternative_experiment("ADT").get_reduced_dimension("PCA") == res["x"].get_reduced_dimension("PCA")).all()
    assert (res2["x"].get_alternative_experiment("ADT").get_column_data()["size_factor"] == res["x"].get_column_data()["size_factor"]).all()
    assert (res2["x"].get_reduced_dimension("TSNE") == res["x"].get_reduced_dimension("TSNE")).all() # UMAP, TSNE, clusters still stored in the main experiment, though.
    assert (res2["x"].get_reduced_dimension("UMAP") == res["x"].get_reduced_dimension("UMAP")).all()
    assert (res2["x"].get_column_data()["graph_cluster"] == res["x"].get_column_data()["graph_cluster"]).all()
    assert res["markers"]["adt"].get_names() == res2["markers"]["adt"].get_names()


def test_analyze_se_crispr():
    se = get_test_se()
    sce = singlecellexperiment.SingleCellExperiment(
        se.get_assays(),
        row_data=se.get_row_data(),
        column_data=se.get_column_data(),
        row_names=se.get_row_names()
    )

    numpy.random.rand(27)
    cmat = numpy.random.poisson(lam=10, size=2000).reshape(20, 100)
    cse = summarizedexperiment.SummarizedExperiment({ "counts": cmat })
    cse.set_row_names(["GUIDE_" + str(i) for i in range(cmat.shape[0])], in_place=True)
    sce.set_alternative_experiment("CRISPR", cse, in_place=True)

    res = scranpy.analyze_se(
        sce,
        crispr_altexp="CRISPR",
        more_tsne_args={ "max_iterations": 10 },
        more_umap_args={ "num_epochs": 5 },
        num_threads=1
    )
    assert res["x"].get_alternative_experiment("CRISPR").get_row_names() == cse.get_row_names() 
    assert numpy.issubdtype(res["x"].get_alternative_experiment("CRISPR").get_column_data()["keep"].dtype, numpy.bool)
    assert res["x"].get_column_data()["combined_keep"].all()

    assert isinstance(res["x"].get_alternative_experiment("CRISPR").get_assay("logcounts"), delayedarray.DelayedArray)
    assert res["markers"]["crispr"].get_names().as_list() == [str(i) for i in sorted(set(res["x"].get_column_data()["graph_cluster"]))]


def test_delayify_assays():
    se = get_test_se()
    sce = singlecellexperiment.SingleCellExperiment(
        se.get_assays(),
        row_data=se.get_row_data(),
        column_data=se.get_column_data(),
        row_names=se.get_row_names()
    )

    sce = sce.set_assay("logcounts", numpy.random.rand(100, 100), in_place=True) 
    sce.set_alternative_experiment("FOOBAR", se, in_place=True)
    delayed = _delayify_assays(sce)

    assert isinstance(delayed.get_assay('counts'), delayedarray.DelayedArray)
    assert (delayedarray.to_dense_array(delayed.get_assay('counts')) == sce.get_assay("counts")).all()

    assert isinstance(delayed.get_assay('logcounts'), delayedarray.DelayedArray)
    assert (delayedarray.to_dense_array(delayed.get_assay('logcounts')) == sce.get_assay("logcounts")).all()

    assert isinstance(delayed.get_alternative_experiment(0).get_assay('logcounts'), delayedarray.DelayedArray)
    assert (delayedarray.to_dense_array(delayed.get_alternative_experiment(0).get_assay('counts')) == sce.get_alternative_experiment(0).get_assay("counts")).all()


def test_define_single_target_embedding():
    se = get_test_se()
    sce = singlecellexperiment.SingleCellExperiment(
        se.get_assays(),
        row_data=se.get_row_data(),
        column_data=se.get_column_data(),
        row_names=se.get_row_names(),
        alternative_experiments={ "FOOBAR": se }
    )

    assert _define_single_target_embedding(sce, False, "PCA") == "PCA"
    assert _define_single_target_embedding(sce, 0, "PCA") == ("FOOBAR", "PCA")
    assert _define_single_target_embedding(sce, "STUFF", "PCA") == ("STUFF", "PCA")


def test_add_source_embedding_to_scale():
    se = get_test_se()
    sce = singlecellexperiment.SingleCellExperiment(
        se.get_assays(),
        row_data=se.get_row_data(),
        column_data=se.get_column_data(),
        row_names=se.get_row_names(),
        alternative_experiments={ "FOOBAR": se }
    )

    all_main = ["A"];
    all_altexp = { "YAY": [2] };
    _add_source_embedding_to_scale(sce, False, "PCA", all_main, all_altexp)
    assert all_main == ["A", "PCA"]
    assert all_altexp == { "YAY": [2] }

    all_main = ["A"];
    all_altexp = { "YAY": [2] };
    _add_source_embedding_to_scale(sce, 0, "PCA", all_main, all_altexp)
    assert all_main == ["A"]
    assert all_altexp == { "YAY": [2], "FOOBAR": ["PCA"] }

    all_main = ["A"];
    all_altexp = { "YAY": [2] };
    _add_source_embedding_to_scale(sce, "STUFF", "PCA", all_main, all_altexp)
    assert all_main == ["A"]
    assert all_altexp == { "YAY": [2], "STUFF": ["PCA"] }
