"""Spec4ML's reusable Python API.

Evaluation helpers are loaded lazily so the core preprocessing package does not
require optional TPOT and visualization dependencies at import time.
"""

from importlib import import_module

from .preprocessing import (
    ALSBaseline,
    AsymmetricLeastSquares,
    EMSC,
    MSC,
    PolynomialDetrend,
    RubberbandBaseline,
    SNV,
    SavitzkyGolay,
    UnitVectorNormalize,
    UnitVectorNormalizer,
)

__version__ = "0.2.0"

_EVALUATION_EXPORTS = {
    "EnsembleML",
    "FeatureImportanceEvaluation_Retrain",
    "aggregate_sample_predictions",
    "evaluate_pipelines",
    "evaluate_predictions",
    "feature_block_importance",
    "feature_block_importance2",
    "get_first_float_column_index",
    "pipeline_LOOCV_evaluation",
    "pipeline_LOOCV_evaluation_with_residual_correction",
    "pipeline_testsets_evaluation",
}

__all__ = [
    "__version__",
    "SavitzkyGolay",
    "SNV",
    "UnitVectorNormalizer",
    "UnitVectorNormalize",
    "MSC",
    "EMSC",
    "RubberbandBaseline",
    "ALSBaseline",
    "AsymmetricLeastSquares",
    "PolynomialDetrend",
    *_EVALUATION_EXPORTS,
]


def __getattr__(name):
    """Load legacy evaluation functions only when they are requested."""
    if name in _EVALUATION_EXPORTS:
        module = import_module(".evaluation_functions", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
