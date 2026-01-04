<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/scranpy.svg?branch=main)](https://cirrus-ci.com/github/<USER>/scranpy)
[![ReadTheDocs](https://readthedocs.org/projects/scranpy/badge/?version=latest)](https://scranpy.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/scranpy/main.svg)](https://coveralls.io/r/<USER>/scranpy)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/scranpy.svg)](https://anaconda.org/conda-forge/scranpy)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/scranpy)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/scranpy.svg)](https://pypi.org/project/scranpy/)
[![Downloads](https://static.pepy.tech/badge/scranpy/month)](https://pepy.tech/project/scranpy)
![Unit tests](https://github.com/libscran/scranpy/actions/workflows/run-tests.yml/badge.svg)

# scran, in Python

## Overview

The **scranpy** package provides Python bindings to the single-cell analysis methods in the [**libscran**](https://github.com/libscran) C++ libraries.
It performs the standard steps in a typical single-cell analysis including quality control, normalization, feature selection, dimensionality reduction, clustering and marker detection.
This package is effectively a mirror of its counterparts in Javascript ([**scran.js**](https://npmjs.com/package/scran.js)) and R ([**scrapper**](https://github.com/libscran/scrapper)),
which are based on the same underlying C++ libraries and concepts.

## Quick start

Let's fetch a dataset from the [**scrnaseq**](https://github.com/BiocPy/scrnaseq) package:

```python
import scrnaseq 
sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
print(sce)
## class: SingleCellExperiment
## dimensions: (20006, 3005)
## assays(1): ['counts']
## row_data columns(1): ['featureType']
## row_names(20006): ['Tspan12', 'Tshz1', 'Fnbp1l', ..., 'mt-Rnr2', 'mt-Rnr1', 'mt-Nd4l']
## column_data columns(9): ['tissue', 'group #', 'total mRNA mol', 'well', 'sex', 'age', 'diameter', 'level1class', 'level2class']
## column_names(3005): ['1772071015_C02', '1772071017_G12', '1772071017_A05', ..., '1772063068_D01', '1772066098_A12', '1772058148_F03']
## main_experiment_name: gene
## reduced_dims(0): []
## alternative_experiments(2): ['repeat', 'ERCC']
## row_pairs(0): []
## column_pairs(0): []
## metadata(0):
```

Then we call **scranpy**'s `analyze()` functions, with some additional information about the mitochondrial subset for quality control purposes.
This will perform all of the usual steps for a routine single-cell analysis, 
as described in Bioconductor's [Orchestrating single cell analysis](https://bioconductor.org/books/OSCA) book.

```python
import scranpy
results = scranpy.analyze_se(
    sce,
    rna_qc_subsets = {
        "mito": [name.startswith("mt-") for name in sce.get_row_names()]
    }
)
print(results["x"])
## class: SingleCellExperiment
## dimensions: (20006, 2874)
## assays(2): ['counts', 'logcounts']
## row_data columns(6): ['featureType', 'mean', 'variance', 'fitted', 'residual', 'hvg']
## row_names(20006): ['Tspan12', 'Tshz1', 'Fnbp1l', ..., 'mt-Rnr2', 'mt-Rnr1', 'mt-Nd4l']
## column_data columns(15): ['tissue', 'group #', 'total mRNA mol', ..., 'keep', 'size_factor', 'graph_cluster']
## column_names(2874): ['1772071015_C02', '1772071017_G12', '1772071017_A05', ..., '1772066097_D04', '1772063068_D01', '1772066098_A12']
## main_experiment_name: gene
## reduced_dims(3): ['PCA', 'TSNE', 'UMAP']
## alternative_experiments(2): ['repeat', 'ERCC']
## row_pairs(0): []
## column_pairs(0): []
## metadata(2): qc PCA
```

We can extract useful bits and pieces from the [`SingleCellExperiment`](https://github.com/BiocPy/singlecellexperiment) stored as `x`:

```python
print(results["x"].get_column_data()[:,["sum", "detected", "subset_proportion_mito", "size_factor", "graph_cluster"])
##                               sum          detected subset_proportion_mito         size_factor    graph_cluster
##                <ndarray[float64]> <ndarray[uint32]>     <ndarray[float64]>  <ndarray[float64]> <ndarray[int64]>
## 1772071015_C02            22354.0              4871    0.03462467567325758  1.4587034836358181                0
## 1772071017_G12            22869.0              4712    0.04901832174559447  1.4923096522889652                0
## 1772071017_A05            32594.0              6055   0.029207829661900962  2.1269115749139242                0
##                               ...               ...                    ...                 ...              ...
## 1772066097_D04             2574.0              1441   0.005827505827505828 0.16796558856932076                6
## 1772063068_D01             4993.0              2001    0.19587422391347886 0.32581669919449047                6
## 1772066098_A12             3099.0              1510    0.06550500161342368 0.20222430418660645               13

print(results["x"].get_reduced_dimension("TSNE"))
## [[  5.56742365 -28.68868021]
##  [  5.60398273 -28.02309408]
##  [  4.77422687 -28.70557818]
##  ...
##  [ 18.76434892   8.48223628]
##  [ 17.69108131   3.82950607]
##  [ 14.09402365   6.71953971]]

print(results["x"].get_reduced_dimension("UMAP"))
## [[10.36782455 -1.76653302]
##  [10.24362564 -1.65133715]
##  [10.46131039 -1.62113261]
##  ...
##  [-3.18933988 -6.42624807]
##  [-2.90072227 -5.33980703]
##  [-5.86072397 -7.62296963]]
```

We can also inspect the top markers for each cluster in each modality:

```python
print(results["markers"]["rna"].get_names())
## ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']

print(scranpy.preview_markers(results["markers"]["rna"]["0"]))
## BiocFrame with 10 rows and 3 columns
##                       mean           detected                lfc
##         <ndarray[float64]> <ndarray[float64]> <ndarray[float64]>
##    Gad1  4.722606264371632 0.9967105263157895 4.5144234560400935
##    Gad2 4.3881067939564335 0.9967105263157895  4.212918994316072
##   Ndrg4  4.336870587905828 0.9967105263157895 2.5370838460299865
##   Stmn3  4.676829189457486  0.993421052631579 2.6318375392719537
##  Vstm2a  2.866641571231044 0.9572368421052632 2.6178131058482004
##  Nap1l5    4.2569402746771                1.0 3.0585216754783877
##  Slc6a1 3.6726655518924356 0.9901315789473684 3.0484528790789396
##   Rab3c 3.8515412644109457 0.9802631578947368 2.9407642089786044
##  Tspyl4 3.3162021744969783                1.0   2.10987030054009
## Slc32a1    1.9797938787573 0.8947368421052633 1.9555828312546393
```

Check out the [reference documentation](https://libscran.github.io/scranpy) for more details.

## Multiple batches

To demonstrate, let's grab two pancreas datasets from the **scrnaseq** package.
Each dataset represents a separate batch of cells generated in different studies.

```python
import scrnaseq 
gsce = scrnaseq.fetch_dataset("grun-pancreas-2016", "2023-12-14", realize_assays=True)
msce = scrnaseq.fetch_dataset("muraro-pancreas-2016", "2023-12-19", realize_assays=True)
```

They don't have the same features, so we'll just take the intersection of their row names before combining them into a single `SingleCellExperiment` object:

```python
import biocutils
common = biocutils.intersect(gsce.get_row_names(), msce.get_row_names())
combined = biocutils.relaxed_combine_columns(
    gsce[biocutils.match(common, gsce.get_row_names()), :],
    msce[biocutils.match(common, msce.get_row_names()), :]
)
print(combined)
## class: SingleCellExperiment
## dimensions: (18499, 4800)
## assays(1): ['counts']
## row_data columns(2): ['symbol', 'chr']
## row_names(18499): ['A1BG-AS1__chr19', 'A1BG__chr19', 'A1CF__chr10', ..., 'ZYX__chr7', 'ZZEF1__chr17', 'ZZZ3__chr1']
## column_data columns(4): ['donor', 'sample', 'label', 'plate']
## column_names(4800): ['D2ex_1', 'D2ex_2', 'D2ex_3', ..., 'D30-8_94', 'D30-8_95', 'D30-8_96']
## main_experiment_name: endogenous
## reduced_dims(0): []
## alternative_experiments(0): []
## row_pairs(0): []
## column_pairs(0): []
## metadata(0):
```

We can now perform a batch-aware analysis, where the blocking factor is also used in relevant functions to avoid problems with batch effects.
This yields mostly the same set of results as before, but with an extra MNN-corrected embedding for clustering, visualization, etc.

```python
import scranpy
block = ["grun"] * gsce.shape[1] + ["muraro"] * msce.shape[1]
results = scranpy.analyze(combined, block=block) # no mitochondrial genes in this case...
print(results["x"].get_reduced_dimension_names())
## ['PCA', 'MNN', 'TSNE', 'UMAP']
```

The blocking factor is also used during marker detection to ensure that any batch effects do not interfere with marker scoring.

```python
print(results["markers"]["rna"].get_names())
## ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

print(scranpy.preview_markers(results["markers"]["rna"]["0"]))
## BiocFrame with 10 rows and 3 columns
##                              mean           detected                lfc
##                <ndarray[float64]> <ndarray[float64]> <ndarray[float64]>
##    PRSS1__chr7 7.4960140622383635                1.0  6.112707795838959
## PLA2G1B__chr12 5.3256781931032435 0.9767441860465116 4.9768663423161525
##   SPINK1__chr5  7.226025132762255                1.0  5.484853630696096
##  PRSS3P2__chr7  6.414465893460902                1.0  5.474642067942027
##   CTRB1__chr16  6.242340887932875                1.0  5.439850813378681
##   CTRB2__chr16  6.211977585350613 0.9767441860465116  5.390247433870743
##   CELA3A__chr1  5.749981006377225 0.9922480620155039  5.379827932369857
##     CPA1__chr7  5.956052489801642 0.9922480620155039  5.398848064205473
##    REG1A__chr2  8.505688048945858                1.0   5.69363720328606
##     CPA2__chr7  4.394240756079104 0.9224806201550388  4.167356591402403
```

## Multiple modalities

Let's grab a 10X Genomics immune profiling dataset (see [here](https://github.com/kanaverse/random-test-files/releases/download/10x-immune-v1.0.0/immune_3.0.0-tenx.h5)),
which contains count data for the entire transcriptome and targeted proteins:

```python
import singlecellexperiment
input = singlecellexperiment.read_tenx_h5("immune_3.0.0-tenx.h5", realize_assays=True)
input.set_row_names(input.get_row_data().get_column("id"), in_place=True)
```

We split our `SingleCellExperiment` into separate objects for genes and ADTs.
We use the gene dataset as the main experiment and store the ADT object as an alternative experiment.

```python
feattypes = input.get_row_data().get_column("feature_type")
gene_data = input[[x == "Gene Expression" for x in feattypes],:]
adt_data = input[[x == "Antibody Capture" for x in feattypes],:]

sce = gene_data
sce.set_alternative_experiment("ADT", adt_data, in_place=True)
print(sce)
## class: SingleCellExperiment
## dimensions: (33555, 8258)
## assays(1): ['counts']
## row_data columns(7): ['feature_type', 'genome', 'id', 'name', 'pattern', 'read', 'sequence']
## row_names(33555): ['ENSG00000243485', 'ENSG00000237613', 'ENSG00000186092', ..., 'IgG2b', 'CD127', 'CD15']
## column_data columns(1): ['barcodes']
## column_names(0):
## main_experiment_name:
## reduced_dims(0): []
## alternative_experiments(0): []
## row_pairs(0): []
## column_pairs(0): []
## metadata(0):

print(sce.get_alternative_experiment("ADT"))
## class: SingleCellExperiment
## dimensions: (17, 8258)
## assays(1): ['counts']
## row_data columns(7): ['feature_type', 'genome', 'id', 'name', 'pattern', 'read', 'sequence']
## row_names(17): ['CD3', 'CD19', 'CD45RA', ..., 'IgG2b', 'CD127', 'CD15']
## column_data columns(1): ['barcodes']
## column_names(0):
## main_experiment_name:
## reduced_dims(0): []
## alternative_experiments(0): []
## row_pairs(0): []
## column_pairs(0): []
## metadata(0):
```

And now we can run the analysis, with some additional specification of the IgG subsets for ADT-related quality control.
ADT-specific results are stored in the alternative experiment of the output `x`:

```python
import scranpy
results = scranpy.analyze_se(
    sce,
    adt_altexp="ADT",
    rna_qc_subsets = { 
        "mito": [n.startswith("MT-") for n in gene_data.get_row_data().get_column("name")]
    },
    adt_qc_subsets = {
        "igg": [n.startswith("IgG") for n in adt_data.get_row_data().get_column("name")]
    }
)

print(results["x"].get_alternative_experiment("ADT").get_column_data()[:,["sum", "detected", "subset_sum_igg", "size_factor"]])
## BiocFrame with 6779 rows and 4 columns
##                       sum          detected     subset_sum_igg        size_factor
##        <ndarray[float64]> <ndarray[uint32]> <ndarray[float64]> <ndarray[float64]>
##    [0]             2410.0                17               21.0 0.7935940817266915
##    [1]             2637.0                17               30.0 0.7941033166783458
##    [2]             5551.0                17               29.0  0.895364130394313
##                       ...               ...                ...                ...
## [6776]             5079.0                17               11.0 0.7920783948957917
## [6777]             1757.0                17               12.0 0.6649272283074951
## [6778]             2312.0                17               33.0 0.7684763715276961

print(results["x"].get_alternative_experiment("ADT").get_reduced_dimension_names())
## ['PCA']
```

`analyze_se()` combines the RNA and ADT data into a single embedding for downstream steps like visualization and clustering.
This ensures that those steps will use information from both modalities.

```python
print(results["x"].get_reduced_dimension_names())
## ['PCA', 'combined', 'TSNE', 'UMAP']

import biocutils
print(biocutils.table(results["x"].get_column_data()["graph_cluster"]))
## ['0'=401, '1'=801, '2'=282, '3'=1030, '4'=1142, '5'=279, '6'=922, '7'=198, '8'=375, '9'=212, '10'=884, '11'=47, '12'=13, '13'=48, '14'=58, '15'=87]
```

Similarly, `analyze_se()` will compute marker statistics for the ADT data:

```python
print(scranpy.preview_markers(results["markers"]["adt"]["0"]))
## BiocFrame with 10 rows and 3 columns
##                      mean           detected                  lfc
##        <ndarray[float64]> <ndarray[float64]>   <ndarray[float64]>
##    CD3 10.710798038087853                1.0    3.073794979159913
##  CD127  7.202083836362728                1.0   2.1670932585149587
## CD45RO  7.062122120526255                1.0   1.6391671901349703
##   CD8a  7.530206276892356                1.0   1.8636563581508114
##   CD56  5.110427162087626                1.0   0.6727102618980374
##   PD-1  5.739783018333582                1.0  0.19200190878211132
##   IgG1  4.330276551600134                1.0 -0.02840790745859983
##   CD25  5.088783799767125                1.0  -0.1269303811218677
##  TIGIT 3.6629345003643983 0.9975062344139651 -0.28573778753415674
##  IgG2a  3.143818721927284 0.9900249376558603  -0.3191747955354897
```

## Customizing the analysis

Most parameters can be changed by modifying the relevant arguments in `analyze_se()`.
For example:

```python
import scrnaseq 
sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
is_mito = [name.startswith("mt-") for name in sce.get_row_names()]

import scranpy
results = scranpy.analyze(
    sce,
    rna_qc_subsets = {
        "mito": is_mito
    },
    more_build_graph_args = {
        "num_neighbors": 10
    },
    more_cluster_graph_args = {
        "multilevel_resolution": 2
    },
    more_rna_pca_args = {
        "number": 15
    },
    more_tsne_args = {
        "perplexity": 25
    },
    more_umap_args = {
        "min_dist": 0.05
    }
)
```

For finer control, users can call each step individually via lower-level functions.
A typical RNA analysis might be implemented as:

```python
res = scranpy.quick_rna_qc_se(sce, subsets={ "mito": is_mito })
res = res[:,res.get_column_data()["keep"]]
res = scranpy.normalize_rna_counts_se(res, size_factors=res.get_column_data()["sum"])
res = scranpy.choose_rna_hvgs_se(res)
res = scranpy.run_pca_se(res, features=res.get_row_data()["hvg"])
res = scranpy.run_all_neighbor_steps_se(res)
markers = scranpy.score_markers_se(res, groups=res.get_column_data()["clusters"])
```

Check out [`analyze_se()` source code](src/scranpy/_se_analyze.py) for more details.
