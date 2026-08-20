"""Scikit-learn transformers for common spectroscopy preprocessing steps."""

from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class _BaseSpectralTransformer(TransformerMixin, BaseEstimator):
    """Shared numeric validation and fitted feature-count handling."""

    def _validate_X(self, X, *, reset):
        try:
            array = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("X must contain only numeric values") from error
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not np.isfinite(array).all():
            raise ValueError("X must contain only finite values")
        if reset:
            self.n_features_in_ = array.shape[1]
        else:
            check_is_fitted(self, "n_features_in_")
            if array.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {array.shape[1]} features, but {self.__class__.__name__} "
                    f"was fitted with {self.n_features_in_} features"
                )
        return array

    def fit(self, X, y=None):
        self._validate_parameters()
        self._validate_X(X, reset=True)
        return self

    def _validate_parameters(self):
        """Validate constructor parameters in concrete transformers."""


class SavitzkyGolay(_BaseSpectralTransformer):
    """Apply row-wise Savitzky-Golay smoothing or differentiation."""

    def __init__(self, window_length=11, polyorder=2, deriv=0, delta=1.0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    def _validate_parameters(self):
        if not isinstance(self.window_length, Integral) or isinstance(
            self.window_length, bool
        ):
            raise ValueError("window_length must be an integer")
        if self.window_length < 3 or self.window_length % 2 == 0:
            raise ValueError(
                "window_length must be an odd integer greater than or equal to 3"
            )
        if not isinstance(self.polyorder, Integral) or isinstance(self.polyorder, bool):
            raise ValueError("polyorder must be an integer")
        if self.polyorder < 0:
            raise ValueError("polyorder must be non-negative")
        if self.window_length <= self.polyorder:
            raise ValueError("window_length must be greater than polyorder")
        if (
            not isinstance(self.deriv, Integral)
            or isinstance(self.deriv, bool)
            or self.deriv not in (0, 1, 2)
        ):
            raise ValueError("deriv must be one of 0, 1, or 2")
        if self.polyorder < self.deriv:
            raise ValueError("polyorder must be greater than or equal to deriv")
        if (
            not isinstance(self.delta, Real)
            or isinstance(self.delta, bool)
            or not np.isfinite(self.delta)
            or self.delta <= 0
        ):
            raise ValueError("delta must be a positive number")

    def fit(self, X, y=None):
        super().fit(X, y)
        if self.window_length > self.n_features_in_:
            raise ValueError("window_length must not exceed the number of features")
        return self

    def transform(self, X):
        array = self._validate_X(X, reset=False)
        return savgol_filter(
            array,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            axis=1,
        )


class SNV(_BaseSpectralTransformer):
    """Standard-normal-variate correction independently for each spectrum."""

    def transform(self, X):
        array = self._validate_X(X, reset=False)
        centered = array - array.mean(axis=1, keepdims=True)
        scale = array.std(axis=1, ddof=0, keepdims=True)
        return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)


class UnitVectorNormalizer(_BaseSpectralTransformer):
    """Normalize each spectrum to unit Euclidean norm; leave zero rows unchanged."""

    def transform(self, X):
        array = self._validate_X(X, reset=False)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


UnitVectorNormalize = UnitVectorNormalizer


def _validate_reference(reference, n_features):
    try:
        result = np.asarray(reference, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("reference must contain only numeric values") from error
    if result.ndim != 1 or result.shape[0] != n_features:
        raise ValueError(f"reference must be one-dimensional with {n_features} values")
    if not np.isfinite(result).all():
        raise ValueError("reference must contain only finite values")
    return result.copy()


class MSC(_BaseSpectralTransformer):
    """Multiplicative scatter correction using a training-only reference."""

    def __init__(self, reference=None):
        self.reference = reference

    def fit(self, X, y=None):
        array = self._validate_X(X, reset=True)
        self.reference_ = (
            array.mean(axis=0)
            if self.reference is None
            else _validate_reference(self.reference, self.n_features_in_)
        )
        if np.ptp(self.reference_) == 0:
            raise ValueError("reference must not be constant")
        return self

    def transform(self, X):
        array = self._validate_X(X, reset=False)
        check_is_fitted(self, "reference_")
        design = np.column_stack((np.ones(self.n_features_in_), self.reference_))
        coefficients = np.linalg.lstsq(design, array.T, rcond=None)[0]
        intercepts, slopes = coefficients
        if np.any(np.isclose(slopes, 0.0)):
            raise ValueError(
                "MSC cannot correct a spectrum with zero multiplicative slope"
            )
        return (array - intercepts[:, None]) / slopes[:, None]


class EMSC(_BaseSpectralTransformer):
    """Extended MSC with a polynomial baseline and multiplicative correction."""

    def __init__(self, degree=2, reference=None):
        self.degree = degree
        self.reference = reference

    def _validate_parameters(self):
        if not isinstance(self.degree, Integral) or isinstance(self.degree, bool):
            raise ValueError("degree must be an integer")
        if self.degree < 0:
            raise ValueError("degree must be non-negative")

    def fit(self, X, y=None):
        self._validate_parameters()
        array = self._validate_X(X, reset=True)
        if self.degree + 2 > self.n_features_in_:
            raise ValueError("degree is too large for the number of features")
        self.reference_ = (
            array.mean(axis=0)
            if self.reference is None
            else _validate_reference(self.reference, self.n_features_in_)
        )
        self.axis_ = np.linspace(-1.0, 1.0, self.n_features_in_)
        baseline = np.vander(self.axis_, N=self.degree + 1, increasing=True)
        self.design_ = np.column_stack((self.reference_, baseline))
        if np.linalg.matrix_rank(self.design_) != self.design_.shape[1]:
            raise ValueError(
                "reference and polynomial baseline terms must be independent"
            )
        return self

    def transform(self, X):
        array = self._validate_X(X, reset=False)
        check_is_fitted(self, ("reference_", "axis_", "design_"))
        coefficients = np.linalg.lstsq(self.design_, array.T, rcond=None)[0]
        slopes = coefficients[0]
        if np.any(np.isclose(slopes, 0.0)):
            raise ValueError(
                "EMSC cannot correct a spectrum with zero multiplicative slope"
            )
        polynomial = self.design_[:, 1:] @ coefficients[1:]
        return ((array.T - polynomial) / slopes).T


class RubberbandBaseline(_BaseSpectralTransformer):
    """Remove a piecewise-linear lower-convex-hull baseline from each row."""

    @staticmethod
    def _baseline(spectrum):
        x = np.arange(spectrum.size, dtype=float)
        hull = []
        for index in range(spectrum.size):
            while len(hull) >= 2:
                left, middle = hull[-2], hull[-1]
                cross = (middle - left) * (spectrum[index] - spectrum[left]) - (
                    spectrum[middle] - spectrum[left]
                ) * (index - left)
                if cross > 0:
                    break
                hull.pop()
            hull.append(index)
        return np.interp(x, x[hull], spectrum[hull])

    def transform(self, X):
        array = self._validate_X(X, reset=False)
        return np.vstack([row - self._baseline(row) for row in array])


class ALSBaseline(_BaseSpectralTransformer):
    """Remove a baseline estimated by asymmetric least squares smoothing."""

    def __init__(self, lam=1e5, p=0.01, n_iter=10):
        self.lam = lam
        self.p = p
        self.n_iter = n_iter

    def _validate_parameters(self):
        if (
            not isinstance(self.lam, Real)
            or isinstance(self.lam, bool)
            or not np.isfinite(self.lam)
            or self.lam <= 0
        ):
            raise ValueError("lam must be a positive number")
        if (
            not isinstance(self.p, Real)
            or isinstance(self.p, bool)
            or not np.isfinite(self.p)
            or not 0 < self.p < 1
        ):
            raise ValueError("p must be between 0 and 1")
        if (
            not isinstance(self.n_iter, Integral)
            or isinstance(self.n_iter, bool)
            or self.n_iter < 1
        ):
            raise ValueError("n_iter must be a positive integer")

    def fit(self, X, y=None):
        super().fit(X, y)
        if self.n_features_in_ < 3:
            raise ValueError("ALSBaseline requires at least three features")
        return self

    def _estimate_baseline(self, spectrum):
        length = spectrum.size
        differences = sparse.diags(
            [np.ones(length - 2), -2 * np.ones(length - 2), np.ones(length - 2)],
            [0, 1, 2],
            shape=(length - 2, length),
            format="csc",
        )
        penalty = self.lam * (differences.T @ differences)
        weights = np.ones(length)
        baseline = spectrum.copy()
        for _ in range(self.n_iter):
            system = sparse.spdiags(weights, 0, length, length) + penalty
            baseline = spsolve(system, weights * spectrum)
            weights = np.where(spectrum > baseline, self.p, 1.0 - self.p)
        return baseline

    def transform(self, X):
        array = self._validate_X(X, reset=False)
        return np.vstack([row - self._estimate_baseline(row) for row in array])


AsymmetricLeastSquares = ALSBaseline


class PolynomialDetrend(_BaseSpectralTransformer):
    """Remove a per-spectrum polynomial trend of the requested degree."""

    def __init__(self, degree=2):
        self.degree = degree

    def _validate_parameters(self):
        if not isinstance(self.degree, Integral) or isinstance(self.degree, bool):
            raise ValueError("degree must be an integer")
        if self.degree < 0:
            raise ValueError("degree must be non-negative")

    def fit(self, X, y=None):
        super().fit(X, y)
        if self.degree >= self.n_features_in_:
            raise ValueError("degree must be less than the number of features")
        self.axis_ = np.linspace(-1.0, 1.0, self.n_features_in_)
        return self

    def transform(self, X):
        array = self._validate_X(X, reset=False)
        check_is_fitted(self, "axis_")
        coefficients = np.polynomial.polynomial.polyfit(
            self.axis_, array.T, self.degree
        )
        trends = np.polynomial.polynomial.polyval(self.axis_, coefficients)
        return array - trends
