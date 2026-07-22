import importlib
import sys
import types

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
sklearn_linear = pytest.importorskip("sklearn.linear_model")
sklearn_pipeline = pytest.importorskip("sklearn.pipeline")
sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
joblib = pytest.importorskip("joblib")
LinearRegression = sklearn_linear.LinearRegression
make_pipeline = sklearn_pipeline.make_pipeline
RandomForestRegressor = sklearn_ensemble.RandomForestRegressor


class ColumnSumRegressor:
    def __init__(self, expected_columns):
        self.expected_columns = expected_columns

    def predict(self, features):
        assert features.columns.tolist() == self.expected_columns
        return features.sum(axis=1).to_numpy()


def _evaluation_functions():
    if "xgboost" not in sys.modules:
        xgboost = types.ModuleType("xgboost")
        xgboost.XGBRegressor = object
        sys.modules["xgboost"] = xgboost
    return importlib.import_module("spec4ml_py.evaluation_functions")


def _write_toy_spectra(tmp_path):
    folder = tmp_path / "SelectedSpectra"
    folder.mkdir()
    rows = []
    for sample_id, target in [
        ("A", 1.0),
        ("B", 2.0),
        ("C", 3.0),
        ("D", 4.0),
        ("E", 5.0),
    ]:
        for rep in range(2):
            rows.append(
                {
                    "Spectra": f"{sample_id}_{rep}",
                    "Sample_ID": sample_id,
                    "target": target,
                    "Index": f"{sample_id}_{rep}",
                    "f1": target + rep * 0.01,
                    "f2": target * 2 + rep * 0.01,
                }
            )
    pd.DataFrame(rows).to_csv(folder / "toy.csv", index=False)
    return folder


def test_evaluation_functions_imports_without_tpot(monkeypatch):
    monkeypatch.setitem(sys.modules, "tpot", None)
    monkeypatch.setitem(sys.modules, "tpot.builtins", None)
    monkeypatch.setitem(sys.modules, "tpot.export_utils", None)
    sys.modules.pop("spec4ml_py.evaluation_functions", None)

    ef = _evaluation_functions()

    assert hasattr(ef, "pipeline_testsets_evaluation")


def test_clone_with_random_state_sets_nested_random_state():
    ef = _evaluation_functions()
    pipe = make_pipeline(RandomForestRegressor())

    cloned = ef._clone_with_random_state(pipe, seed=11)

    assert cloned.get_params()["randomforestregressor__random_state"] == 11


def test_aggregate_and_evaluate_predictions_preserve_columns_and_keys():
    ef = _evaluation_functions()
    predictions = pd.DataFrame(
        {
            "Set": [1, 1, 1, 1],
            "Sample_IDs": ["A", "A", "B", "B"],
            "Groundtruths": [1.0, 1.0, 2.0, 2.0],
            "Predictions": [1.0, 1.2, 1.9, 2.1],
        }
    )

    aggregated = ef.aggregate_sample_predictions(predictions)

    assert list(aggregated.columns) == [
        "Set",
        "Test_Samples_ID",
        "Groundtruths",
        "Predictions_Means",
        "Predictions_Mean_Corrected",
        "Predictions_Medians",
        "Predictions_Medians_Corrected",
    ]
    metrics = ef.evaluate_predictions(
        aggregated["Groundtruths"], aggregated["Predictions_Means"]
    )
    assert set(metrics) == {"r2", "MAE", "RMSE", "r", "p_value", "NMAE (%)", "MAPE"}


@pytest.mark.parametrize("mode", ["group", "spectra"])
def test_evaluate_pipelines_modes_work(tmp_path, monkeypatch, mode):
    ef = _evaluation_functions()
    _write_toy_spectra(tmp_path)
    monkeypatch.chdir(tmp_path)
    spectra_start_index = 3 if mode == "group" else 4

    result = ef.evaluate_pipelines(
        OverAllResults=pd.DataFrame({"pipeline": [1]}),
        pipelines=[LinearRegression()],
        Selected_Preprocessings=["toy"],
        TestSets=[["E"]],
        Sample_ID="Sample_ID",
        target="target",
        Index_column="Index",
        Spectra_Start_Index=spectra_start_index,
        mode=mode,
        grouping_column="Sample_ID",
    )

    assert set(result) == {
        "PipelineIndex",
        "Spectral_Preprocessing",
        "Pipeline_MAEs",
        "Pipeline_MAEsStd",
        "Pipeline_NMAEs",
        "Pipeline_NMAEsStd",
        "Pipeline_R2s",
        "Pipeline_R2sStd",
        "training_times",
    }
    assert result["PipelineIndex"] == [0]
    assert len(result["Pipeline_MAEs"]) == 1


def test_group_mode_loads_index_column_as_index_and_uses_numeric_features(
    tmp_path, monkeypatch
):
    ef = _evaluation_functions()
    _write_toy_spectra(tmp_path)
    monkeypatch.chdir(tmp_path)

    observed_feature_columns = []

    def fake_cross_val_predict(model, training_features, training_target, **kwargs):
        observed_feature_columns.extend(training_features.columns.tolist())
        assert training_features.select_dtypes(include="number").shape[1] == 2
        assert "Index" not in training_features.columns
        return np.asarray(training_target)

    monkeypatch.setattr(ef, "cross_val_predict", fake_cross_val_predict)

    ef.evaluate_pipelines(
        OverAllResults=pd.DataFrame({"pipeline": [1]}),
        pipelines=[LinearRegression()],
        Selected_Preprocessings=["toy"],
        TestSets=[["E"]],
        Sample_ID="Sample_ID",
        target="target",
        Index_column="Index",
        Spectra_Start_Index=3,
        mode="group",
        grouping_column="Sample_ID",
    )

    assert observed_feature_columns == ["f1", "f2"]


def test_evaluate_pipelines_invalid_mode_raises(tmp_path, monkeypatch):
    ef = _evaluation_functions()
    _write_toy_spectra(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="mode must be either"):
        ef.evaluate_pipelines(
            OverAllResults=pd.DataFrame({"pipeline": [1]}),
            pipelines=[LinearRegression()],
            Selected_Preprocessings=["toy"],
            TestSets=[["E"]],
            Sample_ID="Sample_ID",
            target="target",
            Index_column="Index",
            Spectra_Start_Index=4,
            mode="bad",
            grouping_column="Sample_ID",
        )


def _write_whole_dataset_models(tmp_path):
    target = "EqModulus"
    model_folder = tmp_path / f"TPOT_WholeMD_{target}"
    model_folder.mkdir()

    pd.DataFrame(
        {
            "Task": [2],
            "Preprocessing": ["source/prep.csv"],
            "Mode": ["NIR_Time"],
            "Spectral_scale": [14.0],
        }
    ).to_csv(model_folder / "Result_task_2.csv", index=False)
    pd.DataFrame(
        {
            "Task": [1],
            "Preprocessing": ["source/prep.csv"],
            "Mode": ["NIR"],
            "Spectral_scale": [14.0],
        }
    ).to_csv(model_folder / "Result_task_1.csv", index=False)

    joblib.dump(
        ColumnSumRegressor(["1000.0", "1001.0"]),
        model_folder / f"NIR_prep_{target}_fitted.joblib",
    )
    joblib.dump(
        ColumnSumRegressor(["1000.0", "1001.0", "Time_scaled"]),
        model_folder / f"NIR_Time_prep_{target}_fitted.joblib",
    )

    spectra_folder = tmp_path / "TC_SelectedSpectra" / target
    spectra_folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "Spectra": ["A_1", "B_1"],
            "Sample": ["A", "B"],
            "Day": [14, 28],
            target: [3.0, 7.0],
            "1000.0": [1.0, 3.0],
            "1001.0": [2.0, 4.0],
        }
    ).to_csv(spectra_folder / "prep.csv", index=False)

    return target, model_folder, spectra_folder


def test_load_all_models_combines_and_sorts_task_results(tmp_path):
    ef = _evaluation_functions()
    target, model_folder, _ = _write_whole_dataset_models(tmp_path)

    models, results = ef.load_all_models(target, model_folder=model_folder)

    assert list(results["Task"]) == [1, 2]
    assert set(models) == {("prep", "NIR"), ("prep", "NIR_Time")}
    assert models[("prep", "NIR_Time")]["scale"] == 14.0


def test_predict_all_models_handles_spectral_and_scaled_time_modes(tmp_path):
    ef = _evaluation_functions()
    target, model_folder, spectra_folder = _write_whole_dataset_models(tmp_path)

    predictions = ef.predict_all_models(
        target,
        spectra_folder=spectra_folder,
        model_folder=model_folder,
    )

    nir = predictions[predictions["Mode"] == "NIR"].reset_index(drop=True)
    nir_time = predictions[predictions["Mode"] == "NIR_Time"].reset_index(drop=True)
    assert nir["Prediction"].tolist() == [3.0, 7.0]
    assert nir_time["Prediction"].tolist() == [10.0, 21.0]
    assert nir["Groundtruth"].tolist() == [3.0, 7.0]
    assert predictions["Target"].unique().tolist() == [target]
