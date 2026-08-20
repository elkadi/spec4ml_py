import copy

import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from spec4ml_py.preprocessing import (
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


@pytest.fixture
def spectra():
    x = np.linspace(-1.0, 1.0, 31)
    return np.vstack((np.sin(2 * x) + 2, np.cos(3 * x) + 3, x**3 + x + 4))


@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_savitzky_golay_matches_scipy(spectra, deriv):
    transformer = SavitzkyGolay(7, 3, deriv=deriv, delta=0.2)
    expected = savgol_filter(spectra, 7, 3, deriv=deriv, delta=0.2, axis=1)
    assert_allclose(transformer.fit_transform(spectra), expected)


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"window_length": 2}, "odd integer"),
        ({"window_length": 4}, "odd integer"),
        ({"window_length": 5, "polyorder": 5}, "greater than polyorder"),
        ({"window_length": 5, "polyorder": 1, "deriv": 2}, "polyorder"),
        ({"deriv": 3}, "deriv"),
        ({"delta": 0}, "delta"),
    ],
)
def test_savitzky_golay_rejects_invalid_parameters(spectra, parameters, message):
    with pytest.raises(ValueError, match=message):
        SavitzkyGolay(**parameters).fit(spectra)


def test_savitzky_golay_rejects_window_larger_than_spectrum(spectra):
    with pytest.raises(ValueError, match="number of features"):
        SavitzkyGolay(33, 2).fit(spectra)


def test_snv_is_row_wise_and_handles_constant_rows():
    X = np.array([[1.0, 2.0, 4.0, 8.0], [7.0, 7.0, 7.0, 7.0]])
    transformed = SNV().fit_transform(X)
    assert_allclose(transformed[0].mean(), 0.0, atol=1e-14)
    assert_allclose(transformed[0].std(ddof=0), 1.0)
    assert_allclose(transformed[1], 0.0)


def test_unit_vector_normalization_is_row_wise_and_handles_zero_rows():
    X = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    transformed = UnitVectorNormalizer().fit_transform(X)
    assert_allclose(np.linalg.norm(transformed[0]), 1.0)
    assert_allclose(transformed[1], 0.0)
    assert UnitVectorNormalize is UnitVectorNormalizer


def test_msc_recovers_reference_from_offset_and_scale():
    reference = np.sin(np.linspace(0, 2 * np.pi, 41)) + np.linspace(0, 0.2, 41)
    X = np.vstack((1.7 + 2.3 * reference, -0.4 + 0.6 * reference))
    corrected = MSC(reference=reference).fit_transform(X)
    assert_allclose(corrected, np.vstack((reference, reference)), atol=1e-12)


def test_msc_reference_is_learned_only_during_fit():
    train = np.vstack((np.arange(8.0), np.arange(8.0) * 1.2 + 1))
    validation = np.vstack((np.arange(8.0) * 7 - 3, np.arange(8.0) * 0.4))
    transformer = MSC().fit(train)
    learned = transformer.reference_.copy()
    transformer.transform(validation)
    assert_allclose(transformer.reference_, learned)
    assert_allclose(learned, train.mean(axis=0))


def test_emsc_recovers_reference_with_polynomial_baseline_and_scale():
    axis = np.linspace(-1.0, 1.0, 61)
    reference = np.exp(-(((axis - 0.15) / 0.22) ** 2)) + 0.3 * np.sin(8 * axis)
    spectra = []
    for scale, c0, c1, c2 in [(1.8, 0.4, -0.2, 0.3), (0.7, -0.1, 0.5, -0.2)]:
        spectra.append(scale * reference + c0 + c1 * axis + c2 * axis**2)
    corrected = EMSC(degree=2, reference=reference).fit_transform(np.vstack(spectra))
    assert_allclose(corrected, np.vstack((reference, reference)), atol=1e-11)


def test_emsc_reference_and_design_are_not_changed_by_transform():
    axis = np.linspace(-1.0, 1.0, 41)
    reference = np.exp(-((axis / 0.25) ** 2))
    train = np.vstack((reference + 0.1, 1.2 * reference - 0.2 * axis))
    transformer = EMSC(degree=1).fit(train)
    state = (
        transformer.reference_.copy(),
        transformer.axis_.copy(),
        transformer.design_.copy(),
    )
    transformer.transform(np.vstack((3 * reference + axis, 0.4 * reference - 2 * axis)))
    for actual, expected in zip(
        (transformer.reference_, transformer.axis_, transformer.design_), state
    ):
        assert_allclose(actual, expected)


def test_rubberband_removes_linear_baseline_and_preserves_peak():
    baseline = np.linspace(1.0, 3.0, 21)
    signal = baseline.copy()
    signal[10] += 5.0
    corrected = RubberbandBaseline().fit_transform(signal[None, :])[0]
    expected = np.zeros(21)
    expected[10] = 5.0
    assert_allclose(corrected, expected, atol=1e-12)


def test_als_removes_smooth_baseline_while_retaining_peak():
    axis = np.linspace(-1.0, 1.0, 101)
    baseline = 1.0 + 0.3 * axis + 0.2 * axis**2
    peak = 3.0 * np.exp(-(((axis - 0.1) / 0.06) ** 2))
    corrected = ALSBaseline(lam=1e5, p=0.01, n_iter=15).fit_transform(
        (baseline + peak)[None, :]
    )[0]
    assert corrected.max() > 2.5
    assert np.mean(np.abs(corrected[np.abs(axis - 0.1) > 0.3])) < 0.08
    assert AsymmetricLeastSquares is ALSBaseline


def test_polynomial_detrend_removes_known_trend():
    axis = np.linspace(-1.0, 1.0, 51)
    X = np.vstack((2 + 3 * axis - axis**2, -4 + 0.2 * axis + 2 * axis**2))
    assert_allclose(PolynomialDetrend(2).fit_transform(X), 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "transformer",
    [
        SavitzkyGolay(5, 2),
        SNV(),
        UnitVectorNormalizer(),
        MSC(),
        EMSC(degree=1),
        RubberbandBaseline(),
        ALSBaseline(n_iter=2),
        PolynomialDetrend(1),
    ],
)
def test_sklearn_contract_and_deterministic_transform(transformer, spectra):
    cloned = clone(transformer)
    cloned.set_params(**cloned.get_params())
    fitted = cloned.fit(spectra)
    first = fitted.transform(spectra)
    second = fitted.transform(spectra)
    assert fitted is cloned
    assert fitted.n_features_in_ == spectra.shape[1]
    assert_allclose(first, second)
    assert_allclose(first, clone(transformer).fit_transform(spectra))


def test_transformers_work_in_sklearn_pipeline(spectra):
    pipeline = Pipeline(
        [("smooth", SavitzkyGolay(5, 2)), ("normalize", UnitVectorNormalizer())]
    )
    transformed = pipeline.fit_transform(spectra)
    assert transformed.shape == spectra.shape
    assert_allclose(np.linalg.norm(transformed, axis=1), 1.0)


@pytest.mark.parametrize(
    "transformer",
    [
        SavitzkyGolay(5, 2),
        SNV(),
        UnitVectorNormalizer(),
        MSC(),
        EMSC(degree=1),
        RubberbandBaseline(),
        ALSBaseline(n_iter=2),
        PolynomialDetrend(1),
    ],
)
def test_wrong_feature_count_is_rejected(transformer, spectra):
    transformer.fit(spectra)
    with pytest.raises(ValueError, match="was fitted with"):
        transformer.transform(spectra[:, :-1])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_are_rejected(spectra, bad_value):
    X = spectra.copy()
    X[0, 0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        SNV().fit(X)


@pytest.mark.parametrize(
    "transformer",
    [MSC(), EMSC(degree=1), PolynomialDetrend(2), SavitzkyGolay(5, 2)],
)
def test_joblib_serialization_preserves_predictions_and_learned_state(
    transformer, spectra, tmp_path
):
    fitted = transformer.fit(spectra)
    expected = fitted.transform(spectra)
    path = tmp_path / "transformer.joblib"
    joblib.dump(fitted, path)
    restored = joblib.load(path)
    assert restored.n_features_in_ == fitted.n_features_in_
    assert_allclose(restored.transform(spectra), expected)
    if hasattr(fitted, "reference_"):
        assert_allclose(restored.reference_, fitted.reference_)


def test_pipeline_can_fit_regressor_and_predict(spectra):
    target = np.array([1.0, 2.0, 3.0])
    pipeline = Pipeline([("snv", SNV()), ("regressor", LinearRegression())]).fit(
        spectra, target
    )
    assert pipeline.predict(spectra).shape == target.shape


def test_fit_state_is_unchanged_by_repeated_transform(spectra):
    transformer = EMSC(degree=1).fit(spectra)
    before = copy.deepcopy(transformer.__dict__)
    transformer.transform(spectra * 2 + 1)
    for name, expected in before.items():
        actual = transformer.__dict__[name]
        if isinstance(expected, np.ndarray):
            assert_allclose(actual, expected)
        else:
            assert actual == expected
