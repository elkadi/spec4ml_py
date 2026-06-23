# System Modules
import os
import sys
import time
import logging
import glob
import random
import warnings
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd

# Statistics Modules
from math import sqrt
from scipy.stats import pearsonr

# Machine Learning
##TPOT
from tpot import TPOTRegressor
from tpot.builtins import OneHotEncoder, StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive

##SkLearn
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    cross_val_score,
    cross_val_predict,
)
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.decomposition import FastICA
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNetCV, SGDRegressor, RidgeCV, LinearRegression
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    StandardScaler,
    Binarizer,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
)

from sklearn.feature_selection import (
    SelectFwe,
    f_regression,
    SelectPercentile,
    VarianceThreshold,
    SelectFromModel,
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.base import clone
from xgboost import XGBRegressor


def _load_spectra(
    preprocessing_name, data_folder="SelectedSpectra", index_col="Spectra"
):
    """Load a preprocessed spectra CSV using the package's historical defaults."""
    file_path = os.path.join(data_folder, f"{preprocessing_name}.csv")
    return pd.read_csv(file_path, sep=",", index_col=index_col)


def _clone_with_random_state(pipeline, seed=11):
    """Clone a pipeline/estimator and set random_state recursively when supported."""
    model = clone(pipeline)
    if hasattr(model, "steps"):
        set_param_recursive(model.steps, "random_state", seed)
    elif hasattr(model, "random_state"):
        model.random_state = seed
    return model


def _split_by_test_index(file, Sample_ID, TestIndex):
    """Split spectra rows into train/test partitions using sample identifiers."""
    return file[~file[Sample_ID].isin(TestIndex)], file[file[Sample_ID].isin(TestIndex)]


def _xy(data, target, Spectra_Start_Index):
    """Return feature matrix and target vector from the configured spectra start column."""
    return data.iloc[:, Spectra_Start_Index:], data[target]


def _average_by_sample(data, Sample_ID, target, Spectra_Start_Index):
    """Average replicate spectra and targets by sample, preserving existing groupby behavior."""
    testing_target = data.groupby(Sample_ID)[target].mean()
    testing_features = data.groupby(Sample_ID).apply(
        lambda x: x.iloc[:, Spectra_Start_Index:].mean()
    )
    return testing_features, testing_target


def _regression_metrics(y_true, y_pred):
    """Compute the MAE, normalized MAE, and R2 tuple used throughout this module."""
    mae = mean_absolute_error(y_true, y_pred)
    nmae = mae / (max(y_true) - min(y_true))
    r2 = r2_score(y_true, y_pred)
    return mae, nmae, r2


def _flatten_predictions(predictions_df):
    """Flatten list/Series prediction rows to one row per sample prediction."""
    return predictions_df.explode(
        ["Sample_IDs", "Groundtruths", "Predictions"]
    ).reset_index(drop=True)


def _evaluate_prediction_columns(final_results, prediction_columns):
    """Evaluate aggregate prediction columns against the Groundtruths column."""
    return {
        col: evaluate_predictions(final_results["Groundtruths"], final_results[col])
        for col in prediction_columns
    }


def _build_prediction_summary(predictions_df):
    """Flatten fold predictions and aggregate them to sample-level summaries."""
    return aggregate_sample_predictions(_flatten_predictions(predictions_df))


def _evaluate_group_averaged_fold(
    training_data, pipeline, Sample_ID, target, Spectra_Start_Index, grouping_column
):
    """Evaluate one fold using leave-one-group-out predictions averaged by group."""
    if grouping_column is None:
        raise ValueError("grouping_column is required when mode='group'")
    training_features, training_target = _xy(training_data, target, Spectra_Start_Index)
    predictions = cross_val_predict(
        _clone_with_random_state(pipeline),
        training_features,
        training_target,
        groups=training_data[grouping_column],
        cv=LeaveOneGroupOut(),
        n_jobs=-1,
    )
    df = training_data.copy()
    df["predictions"] = predictions
    avg_pred = df.groupby(grouping_column)["predictions"].mean()
    avg_tgt = df.groupby(grouping_column)[target].mean()
    return _regression_metrics(avg_tgt, avg_pred)


def _evaluate_spectra_averaged_fold(
    training_data, pipeline, Index_column, target, Spectra_Start_Index
):
    """Evaluate one fold by leaving out each spectrum index and predicting averaged replicates."""
    preds, ground_truth = [], []
    for si in training_data[Index_column].unique():
        testing_set = training_data[training_data[Index_column] == si]
        training_set = training_data[training_data[Index_column] != si]
        training_features, training_target = _xy(
            training_set, target, Spectra_Start_Index
        )
        testing_features = (
            testing_set.iloc[:, Spectra_Start_Index:].mean(axis=0).to_frame().T
        )
        testing_target = testing_set[target].mean()
        model = _clone_with_random_state(pipeline)
        model.fit(training_features, training_target)
        preds.append(model.predict(testing_features).item())
        ground_truth.append(testing_target)
    return _regression_metrics(ground_truth, preds)


#######################################


def evaluate_pipelines(
    OverAllResults,
    pipelines,
    Selected_Preprocessings,
    TestSets,
    Sample_ID,
    target,
    Index_column,
    Spectra_Start_Index,
    mode="group",  # <-- NEW ARGUMENT: "group" or "spectra"
    grouping_column=None,  # only used for mode="group"
):
    if mode not in {"group", "spectra"}:
        raise ValueError("mode must be either 'group' or 'spectra'")

    PipelineIndex = []
    Spectral_Preprocessing = []
    Pipeline_MAEs = []
    Pipeline_NMAEs = []
    Pipeline_R2s = []
    Pipeline_MAEsStd = []
    Pipeline_NMAEsStd = []
    Pipeline_R2sStd = []
    training_times = []

    for i in range(OverAllResults.shape[0]):
        print(f"Evaluating pipeline {i+1}/{OverAllResults.shape[0]}")
        start_time_n = time.time()
        PipelineIndex.append(i)
        exported_pipeline = pipelines[i]
        Spectral_Preprocessing.append(Selected_Preprocessings[i])
        index_col = Index_column if mode == "group" else None
        file = _load_spectra(Selected_Preprocessings[i], index_col=index_col)

        T_MAEs, T_NMAEs, T_R2s = [], [], []
        for TestIndex in TestSets:
            Training_data, _ = _split_by_test_index(file, Sample_ID, TestIndex)
            Training_data = Training_data.sort_values(by=[Sample_ID])

            if mode == "group":
                mae, nmae, r2 = _evaluate_group_averaged_fold(
                    Training_data,
                    exported_pipeline,
                    Sample_ID,
                    target,
                    Spectra_Start_Index,
                    grouping_column,
                )
            else:
                mae, nmae, r2 = _evaluate_spectra_averaged_fold(
                    Training_data,
                    exported_pipeline,
                    Index_column,
                    target,
                    Spectra_Start_Index,
                )

            T_MAEs.append(mae)
            T_NMAEs.append(nmae)
            T_R2s.append(r2)

        training_times.append(time.time() - start_time_n)
        Pipeline_MAEs.append(np.mean(T_MAEs))
        Pipeline_MAEsStd.append(np.std(T_MAEs))
        Pipeline_NMAEs.append(np.mean(T_NMAEs))
        Pipeline_NMAEsStd.append(np.std(T_NMAEs))
        Pipeline_R2s.append(np.mean(T_R2s))
        Pipeline_R2sStd.append(np.std(T_R2s))

    return {
        "PipelineIndex": PipelineIndex,
        "Spectral_Preprocessing": Spectral_Preprocessing,
        "Pipeline_MAEs": Pipeline_MAEs,
        "Pipeline_MAEsStd": Pipeline_MAEsStd,
        "Pipeline_NMAEs": Pipeline_NMAEs,
        "Pipeline_NMAEsStd": Pipeline_NMAEsStd,
        "Pipeline_R2s": Pipeline_R2s,
        "Pipeline_R2sStd": Pipeline_R2sStd,
        "training_times": training_times,
    }


#######################################


def pipeline_testsets_evaluation(
    Selected_Preprocessings,
    Selected_Pipelines,
    TestSets,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    data_folder="SelectedSpectra",
    seed=11,
):
    """
    Evaluate multiple ML pipelines and their crossponding preprocessing on external test sets where each sample has technical replicates.
    Technical replicates are averaged before prediction.

    Parameters:
        Selected_Preprocessings (list): List of preprocessing names (filenames without `.csv`).
        Selected_Pipelines (list): List of trained pipelines (e.g., from TPOT).
        TestSets (list of lists): External test sets with sample IDs.
        Sample_ID (str): Name of the column indicating sample IDs.
        target (str): Name of the target column.
        Spectra_Start_Index (int): Index in the CSV from which spectral features begin.
        data_folder (str): Folder where CSVs are located.

    Returns:
        metrics_df (pd.DataFrame): DataFrame with MAE, NMAE, R², training time, etc.
        predictions_df (pd.DataFrame): DataFrame with per-sample predictions.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()

    # Output collectors
    Pipelineindex, Preprocessing, Set = [], [], []
    Test_Samples_ID, Groundtruths, Predictions = [], [], []
    training_times, MAEs, NMAEs, R2s = [], [], [], []

    for p, s, pin in zip(
        range(1, len(Selected_Preprocessings) + 1),
        Selected_Preprocessings,
        Selected_Pipelines,
    ):
        print(f"Evaluating pipeline {p}/{len(Selected_Preprocessings)}", end="\r")

        file = _load_spectra(s, data_folder=data_folder)

        for i, TestIndex in enumerate(TestSets):
            Training_data, Testing_data = _split_by_test_index(
                file, Sample_ID, TestIndex
            )
            training_features, training_target = _xy(
                Training_data, target, Spectra_Start_Index
            )
            testing_features, testing_target = _average_by_sample(
                Testing_data, Sample_ID, target, Spectra_Start_Index
            )

            # Store test IDs and ground truths
            Groundtruths.append(testing_target)
            Test_Samples_ID.append(testing_target.index.tolist())
            pi = _clone_with_random_state(pin, seed=seed)

            # Train
            t_start = time.time()
            pi.fit(training_features, training_target)
            t_end = time.time()

            training_times.append(t_end - t_start)

            # Predict
            prediction = pi.predict(testing_features)
            Predictions.append(pd.Series(prediction, index=testing_target.index))

            mae, nmae, r2 = _regression_metrics(testing_target, prediction)

            MAEs.append(mae)
            NMAEs.append(nmae)
            R2s.append(r2)

            # Record identifiers
            Pipelineindex.append(p)
            Preprocessing.append(s)
            Set.append(i + 1)

    # Create result DataFrames
    metrics_df = pd.DataFrame(
        {
            "Pipeline": Pipelineindex,
            "Preprocessing": Preprocessing,
            "Set": Set,
            "MAE": MAEs,
            "NMAE": NMAEs,
            "R2": R2s,
            "training_time": training_times,
        }
    )
    predictions_df = pd.DataFrame(
        {
            "Pipeline": Pipelineindex,
            "Preprocessing": Preprocessing,
            "Set": Set,
            "Sample_IDs": Test_Samples_ID,
            "Groundtruths": Groundtruths,
            "Predictions": Predictions,
        }
    )

    print("Total Time Elapsed: {:.2f} min".format((time.time() - start_time) / 60))
    return metrics_df, predictions_df


###########################


def pipeline_LOOCV_evaluation(
    Selected_Preprocessings,
    Selected_Pipelines,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    data_folder="SelectedSpectra",
):
    """
    Evaluate multiple ML pipelines and their corresponding preprocessing using LOOCV.
    Technical replicates are averaged before prediction.

    Returns:
        pd.DataFrame: Flattened results with one row per prediction.
    """

    start_time = time.time()

    # Collectors
    results = []

    for p_idx, (preprocessing_name, pipeline) in enumerate(
        zip(Selected_Preprocessings, Selected_Pipelines), start=1
    ):
        print(f"Evaluating pipeline {p_idx}/{len(Selected_Preprocessings)}")

        file = _load_spectra(preprocessing_name, data_folder=data_folder)

        for test_sample in file[Sample_ID].unique():
            Training_data = file[file[Sample_ID] != test_sample]
            Testing_data = file[file[Sample_ID] == test_sample]
            training_features, training_target = _xy(
                Training_data, target, Spectra_Start_Index
            )
            testing_features, testing_target = _average_by_sample(
                Testing_data, Sample_ID, target, Spectra_Start_Index
            )
            model = _clone_with_random_state(pipeline)

            # Train
            t_start = time.time()
            model.fit(training_features, training_target)
            training_time = time.time() - t_start

            # Predict
            prediction = model.predict(testing_features)

            # Store results (flattened)
            for sample_id in testing_target.index:
                results.append(
                    {
                        "Pipeline": p_idx,
                        "Preprocessing": preprocessing_name,
                        "Sample_IDs": sample_id,
                        "Groundtruths": testing_target.loc[sample_id],
                        "Predictions": prediction[
                            0
                        ],  # Only one prediction here due to LOOCV
                        "Training_Time": training_time,
                    }
                )

    print("Total Time Elapsed: {:.2f} min".format((time.time() - start_time) / 60))
    return pd.DataFrame(results)


#########################################################################


def pipeline_LOOCV_evaluation_with_residual_correction(
    Selected_Preprocessings,
    Selected_Pipelines,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    data_folder="SelectedSpectra",
):
    start_time = time.time()
    results = []

    for p_idx, (preprocessing_name, pipeline) in enumerate(
        zip(Selected_Preprocessings, Selected_Pipelines), start=1
    ):
        print(f"Evaluating pipeline {p_idx}/{len(Selected_Preprocessings)}")

        file = _load_spectra(preprocessing_name, data_folder=data_folder)

        for test_sample in file[Sample_ID].unique():
            Training_data = file[file[Sample_ID] != test_sample]
            Testing_data = file[file[Sample_ID] == test_sample]
            training_features, training_target = _xy(
                Training_data, target, Spectra_Start_Index
            )
            testing_features, testing_target = _average_by_sample(
                Testing_data, Sample_ID, target, Spectra_Start_Index
            )
            model = _clone_with_random_state(pipeline)

            # Fit main model
            t_start = time.time()
            model.fit(training_features, training_target)
            training_time = time.time() - t_start

            # Predict test sample
            prediction = model.predict(testing_features)

            # Residual correction
            try:
                # Use inner 5-fold CV to get out-of-fold predictions on training data
                oof_preds = cross_val_predict(
                    model,
                    training_features,
                    training_target,
                    cv=KFold(n_splits=5, shuffle=True, random_state=11),
                )
                residuals = training_target.values - oof_preds

                # Fit correction model (e.g., residual ~ y_true)
                residual_model = LinearRegression()
                residual_model.fit(training_target.values.reshape(-1, 1), residuals)

                # TODO: This preserves historical behavior, but uses held-out ground truth
                # during correction and should be revisited in a behavior-changing release.
                correction = residual_model.predict(
                    testing_target.values.reshape(-1, 1)
                )
                corrected_prediction = prediction[0] - correction[0]
            except Exception as e:
                print(f"Residual correction failed for sample {test_sample}: {e}")
                corrected_prediction = None  # fallback

            for sample_id in testing_target.index:
                results.append(
                    {
                        "Pipeline": p_idx,
                        "Preprocessing": preprocessing_name,
                        "Sample_IDs": sample_id,
                        "Groundtruths": testing_target.loc[sample_id],
                        "Predictions": prediction[0],
                        "Corrected_Predictions": corrected_prediction,
                        "Training_Time": training_time,
                    }
                )

    print("Total Time Elapsed: {:.2f} min".format((time.time() - start_time) / 60))
    return pd.DataFrame(results)


########################################################################


def aggregate_sample_predictions(
    predictions_long,
    grouping_column="Sample_IDs",
    prediction_column="Predictions",
    groundtruth_column="Groundtruths",
    Set_column="Set",
    z_thresh=3.5,
):
    """
    Aggregates predictions per sample with outlier removal using Modified Z-score.

    Parameters:
    - predictions_long (pd.DataFrame): Long-format DataFrame
    - z_thresh (float): Threshold for Modified Z-score to detect outliers (default=3.5)

    Returns:
    - pd.DataFrame: Aggregated results with corrected and uncorrected statistics
    """

    def remove_outliers_mzscore(predictions, threshold=z_thresh):
        median = np.median(predictions)
        mad = np.median(np.abs(predictions - median))
        if mad == 0:
            return predictions  # Avoid division by zero
        modified_z = 0.6745 * np.abs(predictions - median) / mad
        return predictions[modified_z <= threshold]

    results = []
    grouped = predictions_long.groupby(grouping_column)

    for sample_id, group in grouped:
        prediction_values = group[prediction_column].values
        filtered = remove_outliers_mzscore(prediction_values)

        results.append(
            {
                "Set": group[Set_column].iloc[0],
                "Test_Samples_ID": sample_id,
                "Groundtruths": group[groundtruth_column].iloc[0],
                "Predictions_Means": np.mean(prediction_values),
                "Predictions_Mean_Corrected": np.mean(filtered),
                "Predictions_Medians": np.median(prediction_values),
                "Predictions_Medians_Corrected": np.median(filtered),
                # Optionally add other fields like training time if present
            }
        )

    return pd.DataFrame(results)


##################################################################################


def evaluate_predictions(true_values, predictions):
    return {
        "r2": r2_score(true_values, predictions),
        "MAE": mean_absolute_error(true_values, predictions),
        "RMSE": sqrt(mean_squared_error(true_values, predictions)),
        "r": pearsonr(true_values, predictions)[0],
        "p_value": pearsonr(true_values, predictions)[1],
        "NMAE (%)": 100
        * mean_absolute_error(true_values, predictions)
        / (np.max(true_values) - np.min(true_values)),
        "MAPE": mean_absolute_percentage_error(true_values, predictions),
    }


####################################################################################
def get_first_float_column_index(df):
    """
    Returns the index of the first column whose name can be converted to a float.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        int: Index of the first float-convertible column name.

    Raises:
        ValueError: If no column names can be converted to float.
    """
    for i, name in enumerate(df.columns):
        try:
            float(name)
            return i
        except ValueError:
            continue
    raise ValueError("No column names can be converted to float.")


##################################################################################
def FeatureImportanceEvaluation_Retrain(
    Selected_Preprocessings,
    Selected_Pipelines,
    TestSets,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    Excluded_Feature=None,
    Prediction_Type="Predictions_Medians",
    data_folder="SelectedSpectra",
    ablation_sets="all",
):
    """
    Interval ablation by mean-replacement in train and test or test alone (retrain per fold).

    For each preprocessing 's' and its pipeline 'pi':
      - Load s.csv
      - For each external split in TestSets:
           * Split train/test
           * For each feature f in Excluded_Feature:
                mean_f = TRAIN[f].mean()
                TEST[f]  = mean_f +- TRAIN[f] = mean_f
           * Train cloned pipeline on TRAIN
           * Predict on averaged TEST replicates
      - Aggregate all fold predictions, compute metrics and return (Excluded Features, R2, MAE, r)
    """

    # Output collectors
    Pipelineindex, Preprocessing, Set = [], [], []
    Test_Samples_ID, Groundtruths, Predictions = [], [], []
    training_times = []

    for p, s, pi in zip(
        range(1, len(Selected_Preprocessings) + 1),
        Selected_Preprocessings,
        Selected_Pipelines,
    ):
        file = _load_spectra(s, data_folder=data_folder)

        for i, TestIndex in enumerate(TestSets):
            Training_data, Testing_data = _split_by_test_index(
                file, Sample_ID, TestIndex
            )
            Training_data = Training_data.copy()
            Testing_data = Testing_data.copy()

            # Replace selected features in testing set with training mean
            if Excluded_Feature:
                for feat in Excluded_Feature:
                    mean_val = Training_data[feat].mean()
                    Testing_data.loc[:, feat] = mean_val
                    if ablation_sets == "all":
                        Training_data.loc[:, feat] = mean_val

            training_features, training_target = _xy(
                Training_data, target, Spectra_Start_Index
            )
            testing_features, testing_target = _average_by_sample(
                Testing_data, Sample_ID, target, Spectra_Start_Index
            )

            # Store test IDs and ground truths
            Groundtruths.append(testing_target)
            Test_Samples_ID.append(testing_target.index.tolist())
            model = _clone_with_random_state(pi)

            # Train
            t_start = time.time()
            model.fit(training_features, training_target)
            t_end = time.time()
            training_times.append(t_end - t_start)

            # Predict
            prediction = model.predict(testing_features)
            Predictions.append(pd.Series(prediction, index=testing_target.index))

            # Record run info
            Pipelineindex.append(p)
            Preprocessing.append(s)
            Set.append(i + 1)

    # Create prediction summary
    predictions_df = pd.DataFrame(
        {
            "Pipeline": Pipelineindex,
            "Preprocessing": Preprocessing,
            "Set": Set,
            "Sample_IDs": Test_Samples_ID,
            "Groundtruths": Groundtruths,
            "Predictions": Predictions,
        }
    )

    Final_Results_5CV_ALL_SA = _build_prediction_summary(predictions_df)

    # Evaluate predictions
    prediction_columns = [
        "Predictions_Medians",
        "Predictions_Means",
        "Predictions_Mean_Corrected",
        "Predictions_Medians_Corrected",
    ]
    results_summary_SA = _evaluate_prediction_columns(
        Final_Results_5CV_ALL_SA, prediction_columns
    )

    R2n = results_summary_SA[Prediction_Type]["r2"]
    MAEn = results_summary_SA[Prediction_Type]["MAE"]
    Rn = results_summary_SA[Prediction_Type]["r"]

    return Excluded_Feature, R2n, MAEn, Rn


##################################################################################
def feature_block_importance(
    File,
    Spectra_Start_Index,
    Selected_Preprocessings_SA,
    Selected_Pipelines_SA,
    TestSets,
    Sample_ID,
    Target,
    Prediction_Type,
    Data_folder,
    interval_size=10,
    step_size=5,
    verbose=True,
    logger=None,
):
    """Backward-compatible wrapper for sliding-window feature-block importance."""
    return feature_block_importance2(
        File=File,
        Spectra_Start_Index=Spectra_Start_Index,
        Selected_Preprocessings_SA=Selected_Preprocessings_SA,
        Selected_Pipelines_SA=Selected_Pipelines_SA,
        TestSets=TestSets,
        Sample_ID=Sample_ID,
        Target=Target,
        Prediction_Type=Prediction_Type,
        Data_folder=Data_folder,
        interval_size=interval_size,
        step_size=step_size,
        include_tail=False,
        start_index=1,
        verbose=verbose,
        logger=logger,
    )


##################################################################################
def feature_block_importance2(
    File,
    Spectra_Start_Index,
    Selected_Preprocessings_SA,
    Selected_Pipelines_SA,
    TestSets,
    Sample_ID,
    Target,
    Prediction_Type,
    Data_folder,
    interval_size=10,
    step_size=5,
    include_tail: bool = True,
    start_index: int = 0,
    verbose=True,
    logger=None,
):
    """
    Perform sliding-window exclusion of spectral features and evaluate model performance.
    Now includes all features if include_tail=True.
    Parameters
    ----------
    filex : DataFrame
        Input dataset with spectral features.
    Spectra_Start_Index : int
        Column index where spectral features start.
    Selected_Preprocessings_SA : list
        Preprocessing options to be used in evaluation.
    Selected_Pipelines_SA : list
        Pipelines to evaluate.
    TestSets : list
        Test sets for validation.
    Sample_ID : str
        Identifier for samples.
    target : str
        Target variable name.
    Prediction_Type : str
        Type of prediction output (e.g. "Predictions_Medians_Corrected").
    data_folder : str
        Folder where spectra or intermediate results are stored.
    interval_size : int, optional
        Number of consecutive features per region to exclude. Default = 10.
    step_size : int, optional
        Sliding window step size. Default = 5.
    verbose : bool, optional
        Whether to print progress to console. Default = True.
    logger : logging.Logger, optional
        Logger for recording progress. If None, falls back to print when verbose=True.

    Returns
    -------
    dict
        Dictionary containing:
        - 'Excluded_Feature_Groups'
        - 'R2ns'
        - 'MAEns'
        - 'Rns'
    """
    spectral_columns = File.columns[Spectra_Start_Index:]
    num_features = len(spectral_columns)

    Excluded_Feature_Groups = []
    R2ns, MAEns, Rns = [], [], []

    start_time = time.time()

    # Loop range to include all features if include_tail=True
    if include_tail:
        loop_range = range(start_index, num_features, step_size)
    else:
        loop_range = range(start_index, num_features - interval_size + 1, step_size)

    for start_idx in loop_range:
        elapsed_min = (time.time() - start_time) / 60
        message = f"Evaluating Excluded Features {start_idx + 1}/{num_features} | Elapsed: {elapsed_min:.2f} min"

        if logger is not None:
            logger.info(message)
        elif verbose:
            print(message, end="\r")
        # Compute window boundaries
        end_idx = min(start_idx + interval_size, num_features)
        excluded_features = spectral_columns[start_idx:end_idx].tolist()

        # Skip incomplete window if include_tail=False
        if not include_tail and len(excluded_features) < interval_size:
            continue

        Excluded_Feature, R2n, MAEn, Rn = FeatureImportanceEvaluation_Retrain(
            Selected_Preprocessings=Selected_Preprocessings_SA,
            Selected_Pipelines=Selected_Pipelines_SA,
            TestSets=TestSets,
            Sample_ID=Sample_ID,
            target=Target,
            Spectra_Start_Index=Spectra_Start_Index,
            Excluded_Feature=excluded_features,
            Prediction_Type=Prediction_Type,
            data_folder=Data_folder,
        )

        Excluded_Feature_Groups.append(excluded_features)
        R2ns.append(R2n)
        MAEns.append(MAEn)
        Rns.append(Rn)

    if verbose:
        print(
            f"\nCompleted in {(time.time() - start_time) / 60:.2f} min. "
            f"Generated {len(Excluded_Feature_Groups)} feature blocks."
        )

    return {
        "Excluded_Feature_Groups": Excluded_Feature_Groups,
        "R2ns": R2ns,
        "MAEns": MAEns,
        "Rns": Rns,
    }


##################################################################################
def EnsembleML(
    Selected_Preprocessings,
    Selected_Pipelines,
    TestSets,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    Prediction_Type="Predictions_Medians",
    data_folder="SelectedSpectra",
    seed=11,
    Output="predictions",
):
    """
    Evaluate an ensemble model based on multiple ML pipelines.
    Returns either the selected prediction vector/series or a tuple of
    (R2, MAE, r, selected_predictions, all_predictions_df, summary_dict),
    depending on `Output`.
    """
    np.random.seed(seed)
    random.seed(seed)
    # Get predictions
    predictions_df = pipeline_testsets_evaluation(
        Selected_Preprocessings=Selected_Preprocessings,
        Selected_Pipelines=Selected_Pipelines,
        TestSets=TestSets,
        Sample_ID=Sample_ID,
        target=target,
        Spectra_Start_Index=Spectra_Start_Index,
        data_folder=data_folder,
        seed=seed,
    )[1]

    # 1) Aggregate predictions to sample-level
    Final_Results_5CV_ALL_SA = _build_prediction_summary(predictions_df)

    # 3) Evaluate different aggregates
    prediction_columns = [
        "Predictions_Medians",
        "Predictions_Means",
        "Predictions_Mean_Corrected",
        "Predictions_Medians_Corrected",
    ]

    # Validate requested Prediction_Type
    if Prediction_Type not in prediction_columns:
        raise ValueError(
            f"Prediction_Type '{Prediction_Type}' not in {prediction_columns}"
        )

    results_summary_SA = _evaluate_prediction_columns(
        Final_Results_5CV_ALL_SA, prediction_columns
    )

    # 4) Pull metrics for the requested aggregate
    R2n = results_summary_SA[Prediction_Type]["r2"]
    MAEn = results_summary_SA[Prediction_Type]["MAE"]
    Rn = results_summary_SA[Prediction_Type]["r"]

    # 5) Return according to Output flag
    if Output == "predictions":
        return Final_Results_5CV_ALL_SA[Prediction_Type]
    else:
        return (
            R2n,
            MAEn,
            Rn,
            Final_Results_5CV_ALL_SA[Prediction_Type],
            Final_Results_5CV_ALL_SA,
            results_summary_SA,
        )
