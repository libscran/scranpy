from typing import Union, Optional

import numpy
import knncolle
import copy
import biocutils

from .run_tsne import run_tsne, tsne_perplexity_to_neighbors
from .run_umap import run_umap
from .build_snn_graph import build_snn_graph 
from .cluster_graph import cluster_graph


def _get_default(f, arg: str):
    import inspect
    sig = inspect.signature(f)
    return sig.parameters[arg].default


def _run_graph(nn, build_args, cluster_args):
    graph = build_snn_graph(nn, **build_args)
    return graph, cluster_graph(graph, **cluster_args)


def run_all_neighbor_steps(
    x: Union[knncolle.Index, numpy.ndarray],
    run_umap_options: Optional[dict] = {},
    run_tsne_options: Optional[dict] = {},
    build_snn_graph_options: Optional[dict] = {},
    cluster_graph_options: dict = {},
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters(),
    collapse_search: bool = False,
    num_threads: int = 3,
) -> biocutils.NamedList:
    """
    Run all steps that depend on the nearest neighbor search,
    i.e., :py:func:`~scranpy.run_tsne.run_tsne`, :py:func:`~scranpy.run_umap.run_umap`, :py:func:`~scranpy.build_snn_graph.build_snn_graph`, and :py:func:`~scranpy.cluster_graph.cluster_graph`.
    This builds the index once and re-uses it for the neighbor search in each step; the various steps are also run in parallel to save more time. 

    Args:
        x:
            Matrix of principal components where rows are cells and columns are PCs, typically produced by :py:func:`~scranpy.run_pca.run_pca`.

            Alternatively, a :py:class:`~knncolle.Index.Index` instance containing a prebuilt search index for the cells.

        run_umap_options:
            Optional arguments for :py:func:`~scranpy.run_umap.run_umap`.
            If ``None``, UMAP is not performed.

        run_tsne_options:
            Optional arguments for :py:func:`~scranpy.run_tsne.run_tsne`.
            If ``None``, t-SNE is not performed.

        build_snn_graph_options:
            Optional arguments for :py:func:`~scranpy.build_snn_graph.build_snn_graph`.
            Ignored if ``cluster_graph_options = None``.

        cluster_graph_options:
            Optional arguments for :py:func:`~scranpy.cluster_graph.cluster_graph`.
            If ``None``, graph-based clustering is not performed.

        nn_parameters:
            Parameters for the nearest-neighbor search.

        collapse_search:
            Whether to collapse the nearest-neighbor search for each step into a single search.
            Steps that need fewer neighbors will use a subset of the neighbors from the collapsed search.
            This is faster but may not give the same results as separate searches for some approximate search algorithms.

        num_threads:
            Number of threads to use for the parallel execution of UMAP, t-SNE and SNN graph construction.
            This overrides the specified number of threads in the various ``*_options`` arguments.

    Returns:
        A :py:class:`~biocutils.NamedList.NamedList` containing the results of each step:

        - ``run_tsne``: results of :py:func:`~scranpy.run_tsne.run_tsne`.
          Omitted if t-SNE was not performed.
        - ``run_umap``: results of :py:func:`~scranpy.run_tsne.run_tsne`.
          Omitted if UMAP was not performed.
        - ``build_snn_graph``: results of :py:func:`~scranpy.build_snn_graph.build_snn_graph`.
          Omitted if graph-based clustering was not performed.
        - ``cluster_graph``: results of :py:func:`~scranpy.cluster_graph.cluster_graph`.
          Omitted if graph-based clustering was not performed.

        If ``collapse_search = False``, results should be identical to the result of running each step in serial.

    Examples:
        >>> import numpy
        >>> pcs = numpy.random.rand(10, 200)
        >>> import scranpy
        >>> output = scranpy.run_all_neighbor_steps(pcs)
        >>> print(output["run_tsne"][:5,:])
        >>> print(output["run_umap"][:5,:])
        >>> print(output["cluster_graph"]["membership"])
    """

    if isinstance(x, knncolle.Index):
        index = x
    else:
        index = knncolle.build_index(nn_parameters, x.T)

    k_choices = {}
    if run_umap_options is not None:
        if "num_neighbors" in run_umap_options:
            umap_k = umap_args["num_neighbors"]
        else:
            umap_k = _get_default(run_umap, "num_neighbors")
        k_choices["run_umap"] = umap_k

    if run_tsne_options is not None:
        if "num_neighbors" in run_tsne_options:
            tsne_k = run_tsne_options["num_neighbors"]
        else:
            if "perplexity" in run_tsne_options:
                tsne_perp = run_tsne_options["perplexity"]
            else:
                tsne_perp = _get_default(run_tsne, "perplexity")
            tsne_k = tsne_perplexity_to_neighbors(tsne_perp)
        k_choices["run_tsne"] = tsne_k

    if cluster_graph_options is not None:
        if "num_neighbors" in build_snn_graph_options:
            cluster_k = build_snn_graph_options["num_neighbors"]
        else:
            cluster_k = _get_default(build_snn_graph, "num_neighbors")
        k_choices["cluster_graph"] = cluster_k

    nn_res = {}
    if collapse_search:
        all_res = knncolle.find_knn(
            index,
            num_neighbors=max(k_choices.values()),
            num_threads=num_threads,
        )
        for n, curk in k_choices.items():
            curi = all_res.index
            curd = all_res.distance
            if curk != curi.shape[1]:
                nn_res[n] = knncolle.FindKnnResults(curi[:,:curk], curd[:,:curk])
            else:
                nn_res[n] = all_res

    else:
        precomputed = {}
        for n, curk in k_choices.items():
            if curk in precomputed:
                curres = precomputed[curk]
            else:
                curres = knncolle.find_knn(
                    index,
                    num_neighbors=curk,
                    num_threads=num_threads,
                )
                precomputed[curk] = curres
            nn_res[n] = curres


    _tasks = []
    _ids = []

    run_tsne_options = copy.copy(run_tsne_options)
    run_umap_options = copy.copy(run_umap_options)
    build_snn_graph_options = copy.copy(build_snn_graph_options)

    num_tasks = len(k_choices)
    concurrent = (num_threads > 1 and num_tasks > 1)
    if concurrent:
        import multiprocessing as mp
        import platform
        from concurrent.futures import ProcessPoolExecutor, wait

        pp = platform.platform()
        extra_args = {}
        if "macos" in pp.lower():
            extra_args["mp_context"] = mp.get_context("fork")

        executor = ProcessPoolExecutor(max_workers=min(num_tasks, num_threads), **extra_args)
        threads_per_task = max(1, int(num_threads / num_tasks))

        if run_tsne_options is not None: 
            run_tsne_options["num_threads"] = threads_per_task
            _tasks.append(
                executor.submit(
                    run_tsne,
                    nn_res["run_tsne"],
                    **run_tsne_options,
                )
            )
            _ids.append("run_tsne")

        if run_umap_options is not None: 
            run_umap_options["num_threads"] = threads_per_task
            _tasks.append(
                executor.submit(
                    run_umap,
                    nn_res["run_umap"],
                    **run_umap_options,
                )
            )
            _ids.append("run_umap")

        if cluster_graph_options is not None:
            build_snn_graph_options["num_threads"] = threads_per_task
            _tasks.append(
                executor.submit(
                    _run_graph,
                    nn_res["cluster_graph"],
                    build_snn_graph_options,
                    cluster_graph_options
                )
            )
            _ids.append("cluster_graph")

    else:
        if run_tsne_options is not None: 
            run_tsne_options["num_threads"] = num_threads
            _tasks.append(run_tsne(nn_res["run_tsne"], **run_tsne_options))
            _ids.append("run_tsne")

        if run_umap_options is not None: 
            run_umap_options["num_threads"] = num_threads
            _tasks.append(run_umap(nn_res["run_umap"], **run_umap_options))
            _ids.append("run_umap")

        if cluster_graph_options is not None:
            build_snn_graph_options["num_threads"] = num_threads
            _tasks.append(_run_graph(nn_res["cluster_graph"], build_snn_graph_options, cluster_graph_options))
            _ids.append("cluster_graph")

    output = biocutils.NamedList([], [])
    if concurrent:
        wait(_tasks)
        executor.shutdown()
        for i, task in enumerate(_tasks):
            output[_ids[i]] = task.result()
    else:
        for i, task in enumerate(_tasks):
            output[_ids[i]] = task

    if "cluster_graph" in output.get_names():
        graph, clusters = output["cluster_graph"]
        output["build_snn_graph"] = graph
        output["cluster_graph"] = clusters 

    return output
