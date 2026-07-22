import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from spec4ml_py.loocv import pipeline_LOOCV_evaluation


def _write_preprocessing_file(tmp_path):
    data = pd.DataFrame(
        {
            "Sample": ["A", "A", "B", "B", "C", "C"],
            "Spectra": ["A1", "A1", "B1", "B1", "C1", "C1"],
            "Target": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "1000.0": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
            "1001.0": [1.5, 1.6, 2.5, 2.6, 3.5, 3.6],
        }
    )
    data.to_csv(tmp_path / "prep.csv", index=False)


def test_evaluation_ids_restrict_testing_but_not_training(tmp_path):
    _write_preprocessing_file(tmp_path)

    results = pipeline_LOOCV_evaluation(
        Selected_Preprocessings=["prep"],
        Selected_Pipelines=[DummyRegressor(strategy="mean")],
        Sample_ID="Spectra",
        target="Target",
        Spectra_Start_Index=3,
        data_folder=str(tmp_path),
        evaluation_ids=["A1"],
        index_col=None,
    )

    assert results["Sample_IDs"].tolist() == ["A1"]
    assert np.isclose(results.loc[0, "Groundtruths"], 1.0)
    # Both B1 and C1 remain in training, so the training mean is 2.5.
    assert np.isclose(results.loc[0, "Predictions"], 2.5)


def test_exclude_ids_remove_outer_test_rows_before_loocv(tmp_path):
    _write_preprocessing_file(tmp_path)

    results = pipeline_LOOCV_evaluation(
        Selected_Preprocessings=["prep"],
        Selected_Pipelines=[DummyRegressor(strategy="mean")],
        Sample_ID="Spectra",
        target="Target",
        Spectra_Start_Index=3,
        data_folder=str(tmp_path),
        evaluation_ids=["A1"],
        exclude_column="Sample",
        exclude_ids=["C"],
        index_col=None,
    )

    assert results["Sample_IDs"].tolist() == ["A1"]
    # C is removed first; when A1 is left out, only B1 remains in training.
    assert np.isclose(results.loc[0, "Predictions"], 2.0)


def test_missing_evaluation_id_raises(tmp_path):
    _write_preprocessing_file(tmp_path)

    try:
        pipeline_LOOCV_evaluation(
            Selected_Preprocessings=["prep"],
            Selected_Pipelines=[DummyRegressor(strategy="mean")],
            Sample_ID="Spectra",
            target="Target",
            Spectra_Start_Index=3,
            data_folder=str(tmp_path),
            evaluation_ids=["missing"],
            index_col=None,
        )
    except ValueError as exc:
        assert "evaluation_ids" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing evaluation ID")
