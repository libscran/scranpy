# Changelog

## Version 0.3.0

- Updated to the latest version of all C++ libraries in the **assorthead** package.
  This should not affect results except for MNN correction where the underlying algorithm has changed.
- Updated to version 1.0.0 of **igraph**, which may result in some small changes to the graph clustering results.
- Functions that previously returned a dedicated `*Result` dataclass will now return a `BiocFrame` or `NamedList`.
  This is simpler and improves consistency with the R package. 
- Added `SummarizedExperiment`-compatible wrappers for each function.
  These are identifiable by the `*_se` suffix in their name and will accept (and often return) a `SummarizedExperiment` for improved compatibility with **BiocPy** workflows.
- Replaced the `analyze()` function with `analyze_se()`, which converts the analysis results in a more convenient format for downstream inspection.
- Set `collapse_search=True` in `run_all_neighbor_steps()`, which improves speed at the cost of some reproducibility when compared to running each function separately.
- Added a `delayed=` option to `normalize_counts()` to avoid returning a `DelayedArray`.
- Transpose the arrays returned by `run_tsne()` and `run_umap()` so that each row corresponds to a cell and each column is a dimension.

## Version 0.2.2

- Version bump to recompile for **assorthead** updates.

## Version 0.2.1

- Minor bugfixes when converting some `*Results` to BiocPy classes.

## Version 0.2.0

- Major refactor to use the new [**libscran**](https://github.com/libscran) C++ libraries.
  Functions are now aligned with those in the [**scrapper**](https://bioconductor.org/packages/scrapper) package.
- Removed support for Python 3.8 (EOL).

## Version 0.1.0

- Added overlord functions for basic, multi-modal and multi-sample analyses from matrices, SummarizedExperiments and SingleCellExperiments.
