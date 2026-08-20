# Spectral preprocessing

`spec4ml_py.preprocessing` provides deterministic, scikit-learn-compatible
transformers for spectroscopy workflows. Each transformer operates on a
two-dimensional matrix with samples in rows and wavelengths/features in
columns. They can be cloned, serialized with joblib, and used directly in a
scikit-learn `Pipeline`.

```python
from sklearn.pipeline import Pipeline
from spec4ml_py.preprocessing import SavitzkyGolay, SNV

pipeline = Pipeline(
    [
        ("derivative", SavitzkyGolay(window_length=11, polyorder=2, deriv=1)),
        ("snv", SNV()),
    ]
)
X_preprocessed = pipeline.fit_transform(X_train)
X_validation_preprocessed = pipeline.transform(X_validation)
```

## Transformers

- `SavitzkyGolay(window_length=11, polyorder=2, deriv=0, delta=1.0)` applies
  smoothing or first/second derivatives using SciPy's Savitzky-Golay filter.
- `SNV()` centers and scales each nonconstant spectrum independently using
  population standard deviation. Constant rows become zero rows.
- `UnitVectorNormalizer()` (alias `UnitVectorNormalize`) applies row-wise L2
  normalization. Zero rows remain zero.
- `MSC(reference=None)` performs multiplicative scatter correction. With the
  default, `reference_` is learned from the training-row mean during `fit`.
- `EMSC(degree=2, reference=None)` extends MSC with a per-spectrum polynomial
  baseline. Its `reference_`, coordinate `axis_`, and regression `design_` are
  fixed during `fit` and reused by `transform`.
- `RubberbandBaseline()` removes a piecewise-linear lower-convex-hull baseline.
- `ALSBaseline(lam=1e5, p=0.01, n_iter=10)` (alias
  `AsymmetricLeastSquares`) removes a sparse asymmetric-least-squares baseline.
- `PolynomialDetrend(degree=2)` removes a row-wise polynomial trend.

MSC and EMSC deliberately learn their default references only from training
data. Fit these steps inside the cross-validation pipeline; do not fit them on
the complete dataset before splitting.
