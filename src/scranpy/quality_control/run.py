from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

from biocframe import BiocFrame

from .._abstract import AbstractStepOptions
from ..types import validate_object_type
from .rna import CreateRnaQcFilter, PerCellRnaQcMetricsOptions, SuggestRnaQcFilters

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class RnaQualityControlOptions(AbstractStepOptions):
    """Arguments to filter out low quality cells.,

    Attributes:
        per_cell_rna_qc_metrics (PerCellRnaQcMetricsOptions): Arguments to compute
            per cell qc metrics.
            (:py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics`).
        create_rna_qc_filters (CreateRnaQcFilter): Arguments to create qc filter
            (:py:meth:`~scranpy.quality_control.rna.create_rna_qc_filters`)
        suggest_rna_qc_filters (SuggestRnaQcFilters): Arguments to
            :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.
        mito_subset (Union[str, bool], optional): subset mitochondrial genes.
    """

    per_cell_rna_qc_metrics: PerCellRnaQcMetricsOptions = PerCellRnaQcMetricsOptions()
    create_rna_qc_filters: CreateRnaQcFilter = CreateRnaQcFilter()
    suggest_rna_qc_filters: SuggestRnaQcFilters = SuggestRnaQcFilters()
    mito_subset: Optional[Union[str, int]] = None
    custom_thresholds: Optional[BiocFrame] = None

    def __post_init__(self):
        validate_object_type(self.per_cell_rna_qc_metrics, PerCellRnaQcMetricsOptions)
        validate_object_type(self.create_rna_qc_filters, CreateRnaQcFilter)
        validate_object_type(self.suggest_rna_qc_filters, SuggestRnaQcFilters)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.per_cell_rna_qc_metrics.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        self.per_cell_rna_qc_metrics.verbose = verbose
        self.suggest_rna_qc_filters.verbose = verbose
        self.create_rna_qc_filters.verbose = verbose

    def set_block(self, block: Optional[Sequence] = None):
        """Set block.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        self.suggest_rna_qc_filters.block = block
        self.create_rna_qc_filters.block = block

    def set_subset(self, subset: Optional[Mapping] = None):
        """Set subsets.

        Args:
            subset (Mapping, optional): Set subsets. Defaults to None.
        """
        self.per_cell_rna_qc_metrics.subsets = subset
