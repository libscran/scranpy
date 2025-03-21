# Changelog

## Version 0.2.3

- Added a `delayed=` option to `normalize_counts()` to avoid returning a `DelayedArray`.

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
