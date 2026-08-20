"""Leakage-safe, scikit-learn-compatible spectral preprocessing."""

from ._transformers import (
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

__all__ = [
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
]
