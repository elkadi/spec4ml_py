# spec4ml-py Documentation

`spec4ml-py` is the Python package for spectral-data machine-learning workflows in the Spec4ML project. It provides utilities for evaluating regression pipelines, handling technical replicates during validation, aggregating predictions, computing metrics, and performing spectral feature-importance experiments.

This repository is separate from:

- the R package: `elkadi/spec4ml`, and
- the Streamlit application: `elkadi/SpecML-Studio`.

## Package identity

- Distribution name: `spec4ml-py`
- Import name: `spec4ml_py`
- Repository: `https://github.com/elkadi/spec4ml_py`
- Python requirement: `>=3.9`
- License: BSD-3-Clause
- Build backend: Hatchling

## Installation

Install from PyPI:

```bash
pip install spec4ml-py
```

Install from GitHub:

```bash
pip install git+https://github.com/elkadi/spec4ml_py.git
```

Install for development:

```bash
git clone https://github.com/elkadi/spec4ml_py.git
cd spec4ml_py
pip install -e ".[dev]"
```

Verify installation:

```python
import spec4ml_py
print(spec4ml_py.__version__)
```

## Runtime dependencies

The package metadata declares:

- `numpy>=1.23`
- `scipy>=1.9`

The current evaluation module also imports several scientific and ML packages used by the available workflows, including:

- `pandas`
- `scikit-learn`
- `tpot`
- `xgboost`

If you use the evaluation functions directly, install the full scientific stack required by your workflow. The Studio repository’s full environment uses TPOT and XGBoost for AutoML-related workflows.

## Public API overview

The package currently exposes:

```python
from spec4ml_py import __version__
from spec4ml_py import *
```

The package `__init__.py` imports all names from:

- `spec4ml_py.evaluation_functions`
- `spec4ml_py.libraries`

The most important user-facing functions are in `evaluation_functions.py`.

## Data conventions

Most workflows expect tabular CSV data where:

- rows represent spectra or technical replicates,
- metadata columns come before spectral feature columns,
- the target column is numeric for regression workflows,
- a sample identifier column identifies samples or technical replicate groups,
- spectral features begin at `Spectra_Start_Index`, commonly defaulting to `16`,
- preprocessed spectra are often stored in a folder such as `SelectedSpectra`, with one CSV file per preprocessing method.

Several functions assume files are indexed by a column named `Spectra` when reading CSV files:

```python
pd.read_csv(file_path, sep=",", index_col="Spectra")
```

Adjust your input files or arguments accordingly.

## Core workflow patterns

### 1. Evaluate selected pipelines on external test sets

Use `pipeline_testsets_evaluation()` when you already have:

- selected preprocessing names,
- trained or configured scikit-learn-compatible pipelines,
- one or more external test sets defined by sample IDs.

```python
from spec4ml_py import pipeline_testsets_evaluation

metrics_df, predictions_df = pipeline_testsets_evaluation(
    Selected_Preprocessings=["SNV", "SG1"],
    Selected_Pipelines=[pipeline_1, pipeline_2],
    TestSets=[["S01", "S02"], ["S03", "S04"]],
    Sample_ID="Sample_ID",
    target="Target",
    Spectra_Start_Index=16,
    data_folder="SelectedSpectra",
    seed=11,
)
```

Returns:

- `metrics_df`: per-pipeline, per-preprocessing, per-test-set metrics.
- `predictions_df`: sample IDs, ground-truth values, and prediction series.

The function averages technical replicate spectra for each test sample before prediction.

### 2. Run leave-one-sample-out evaluation

Use `pipeline_LOOCV_evaluation()` to evaluate selected preprocessing/pipeline combinations by leaving out one sample ID at a time.

```python
from spec4ml_py import pipeline_LOOCV_evaluation

loocv_df = pipeline_LOOCV_evaluation(
    Selected_Preprocessings=["SNV", "SG1"],
    Selected_Pipelines=[pipeline_1, pipeline_2],
    Sample_ID="Sample_ID",
    target="Target",
    Spectra_Start_Index=16,
    data_folder="SelectedSpectra",
)
```

Returns a flattened `pandas.DataFrame` with one row per prediction and columns such as:

- `Pipeline`
- `Preprocessing`
- `Sample_IDs`
- `Groundtruths`
- `Predictions`
- `Training_Time`

### 3. Run LOOCV with residual correction

`pipeline_LOOCV_evaluation_with_residual_correction()` extends the LOOCV workflow by fitting an inner residual-correction model.

```python
from spec4ml_py import pipeline_LOOCV_evaluation_with_residual_correction

corrected_df = pipeline_LOOCV_evaluation_with_residual_correction(
    Selected_Preprocessings=["SNV"],
    Selected_Pipelines=[pipeline_1],
    Sample_ID="Sample_ID",
    target="Target",
)
```

The output includes both raw predictions and `Corrected_Predictions` where correction succeeds.

### 4. Aggregate repeated predictions per sample

Use `aggregate_sample_predictions()` when long-format predictions contain multiple predictions per sample.

```python
from spec4ml_py import aggregate_sample_predictions

summary = aggregate_sample_predictions(
    predictions_long,
    grouping_column="Sample_IDs",
    prediction_column="Predictions",
    groundtruth_column="Groundtruths",
    Set_column="Set",
    z_thresh=3.5,
)
```

The function computes:

- mean prediction,
- median prediction,
- modified-Z-score outlier-filtered mean,
- modified-Z-score outlier-filtered median.

Returned columns include:

- `Set`
- `Test_Samples_ID`
- `Groundtruths`
- `Predictions_Means`
- `Predictions_Mean_Corrected`
- `Predictions_Medians`
- `Predictions_Medians_Corrected`

### 5. Compute standard regression metrics

Use `evaluate_predictions()` for a compact metrics dictionary.

```python
from spec4ml_py import evaluate_predictions

metrics = evaluate_predictions(true_values, predictions)
print(metrics)
```

Returns:

- `r2`
- `MAE`
- `RMSE`
- `r`
- `p_value`
- `NMAE (%)`
- `MAPE`

### 6. Compare pipelines with replicate-aware modes

`evaluate_pipelines()` supports two modes:

- `mode="group"`: row-level prediction with `LeaveOneGroupOut`, then average predictions by group.
- `mode="spectra"`: average spectra for the held-out sample before prediction.

```python
from spec4ml_py import evaluate_pipelines

results = evaluate_pipelines(
    OverAllResults=overall_results_df,
    pipelines=pipelines,
    Selected_Preprocessings=selected_preprocessings,
    TestSets=test_sets,
    Sample_ID="Sample_ID",
    target="Target",
    Index_column="Sample_ID",
    Spectra_Start_Index=16,
    mode="group",
    grouping_column="Sample_ID",
)
```

Returns a dictionary with pipeline-level means and standard deviations for:

- MAE,
- NMAE,
- R²,
- training time.

## Technical replicate handling

The Python package supports replicate-aware evaluation through function arguments and grouping behavior rather than a formal configuration class.

Common patterns:

### Average spectra before prediction

Functions such as `pipeline_testsets_evaluation()` and `pipeline_LOOCV_evaluation()` average technical replicate feature rows for each test sample before prediction.

This is useful when the final prediction should represent the sample rather than each individual replicate spectrum.

### Average predictions after modeling

`evaluate_pipelines(mode="group", grouping_column="Sample_ID")` uses `LeaveOneGroupOut()` to generate row-level predictions while keeping groups intact, then averages predictions by group for metrics.

This is useful when the model should see replicate-level variation during training, but final evaluation should be sample-level.

### Avoiding leakage

Always keep technical replicates from the same physical/biological sample in the same fold. Do not allow replicate spectra from one sample to appear in both training and test data.

## Feature importance and interval ablation

The package includes functions for spectral-region importance using feature-block ablation.

### `FeatureImportanceEvaluation_Retrain()`

Retrains pipelines after replacing selected spectral features with training-set means. This evaluates how much model performance changes when a spectral region is removed or neutralized.

```python
excluded_features, r2, mae, r = FeatureImportanceEvaluation_Retrain(
    Selected_Preprocessings=selected_preprocessings,
    Selected_Pipelines=selected_pipelines,
    TestSets=test_sets,
    Sample_ID="Sample_ID",
    target="Target",
    Spectra_Start_Index=16,
    Excluded_Feature=["1200", "1201", "1202"],
    Prediction_Type="Predictions_Medians",
    data_folder="SelectedSpectra",
    ablation_sets="all",
)
```

### `feature_block_importance()`

Runs sliding-window ablation across spectral columns.

```python
importance = feature_block_importance(
    File=df,
    Spectra_Start_Index=16,
    Selected_Preprocessings_SA=selected_preprocessings,
    Selected_Pipelines_SA=selected_pipelines,
    TestSets=test_sets,
    Sample_ID="Sample_ID",
    Target="Target",
    Prediction_Type="Predictions_Medians",
    Data_folder="SelectedSpectra",
    interval_size=10,
    step_size=5,
)
```

Returns a dictionary with:

- `Excluded_Feature_Groups`
- `R2ns`
- `MAEns`
- `Rns`

### `feature_block_importance2()`

Similar to `feature_block_importance()`, with optional tail handling through `include_tail=True`.

## Ensemble workflow

`EnsembleML()` combines predictions from multiple preprocessing/pipeline combinations and summarizes performance using the selected prediction aggregation column.

```python
from spec4ml_py import EnsembleML

selected_predictions = EnsembleML(
    Selected_Preprocessings=selected_preprocessings,
    Selected_Pipelines=selected_pipelines,
    TestSets=test_sets,
    Sample_ID="Sample_ID",
    target="Target",
    Spectra_Start_Index=16,
    Prediction_Type="Predictions_Medians",
    data_folder="SelectedSpectra",
    seed=11,
    Output="predictions",
)
```

Valid `Prediction_Type` values are:

- `Predictions_Medians`
- `Predictions_Means`
- `Predictions_Mean_Corrected`
- `Predictions_Medians_Corrected`

## Utility functions

### `get_first_float_column_index()`

Detects the first column whose name can be converted to a float. This is useful when spectral feature columns are named by wavelength or wavenumber.

```python
idx = get_first_float_column_index(df)
```

Raises `ValueError` if no column names can be converted to floats.

## Input and output examples

### Example input layout

```text
Spectra,Sample_ID,Group,Target,...metadata...,1000,1001,1002,1003
sp_001,S01,A,2.1,...,0.12,0.13,0.15,0.16
sp_002,S01,A,2.1,...,0.11,0.14,0.15,0.15
sp_003,S02,B,4.7,...,0.32,0.34,0.35,0.37
```

### Recommended output files

For reproducibility, save:

```python
metrics_df.to_csv("metrics.csv", index=False)
predictions_df.to_csv("predictions.csv", index=False)
```

For aggregated predictions:

```python
summary.to_csv("aggregated_predictions.csv", index=False)
```

## Reproducibility

Recommended practice:

```python
import sys
import numpy as np
import pandas as pd
import sklearn
import spec4ml_py

print("python", sys.version)
print("spec4ml_py", spec4ml_py.__version__)
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("sklearn", sklearn.__version__)
```

Set seeds where available:

```python
metrics_df, predictions_df = pipeline_testsets_evaluation(..., seed=11)
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'tpot'`

Some evaluation workflows import TPOT. Install TPOT in your active environment:

```bash
pip install tpot
```

For the full Studio-style environment, use the Studio repository’s `requirements-full.txt`.

### `ModuleNotFoundError: No module named 'xgboost'`

Install XGBoost:

```bash
pip install xgboost
```

### `FileNotFoundError` for `SelectedSpectra/*.csv`

Many functions read files from `data_folder`, defaulting to `SelectedSpectra`. Make sure the folder exists and contains one CSV per preprocessing name, for example:

```text
SelectedSpectra/SNV.csv
SelectedSpectra/SG1.csv
```

### `KeyError: 'Spectra'`

Some functions read CSVs using `index_col="Spectra"`. Ensure your CSV has a `Spectra` column, or adjust the function/source workflow to match your file layout.

### Leakage or overly optimistic metrics

Check whether:

- technical replicates from the same sample were split across train and test,
- preprocessing was fitted on the entire dataset before validation,
- target values or group identifiers were accidentally included as features,
- `Spectra_Start_Index` points to the correct first spectral feature column.

## Relationship to other repositories

- `elkadi/spec4ml`: R implementation and R-specific spectral preprocessing/modeling functions.
- `elkadi/spec4ml_py`: this Python package.
- `elkadi/SpecML-Studio`: Streamlit application built on the Python package and app-specific services.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Recommended checks:

```bash
ruff check .
black --check .
mypy spec4ml_py
```

Build package:

```bash
python -m build
```

## Documentation maintenance checklist

When changing the package:

1. Update this file.
2. Update the root `README.md` if installation or quickstart behavior changes.
3. Add or update tests under `tests/`.
4. Keep `pyproject.toml` dependencies synchronized with imports used by public functions.
5. Document any workflow that can introduce cross-validation leakage.
6. Add examples for new public functions.
