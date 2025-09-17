#System Modules
import os
import sys
import time
import glob
import random
import warnings
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
#Statistics Modules
from math import sqrt
from scipy.stats import pearsonr
#Machine Learning
##TPOT
from tpot import TPOTRegressor
from tpot.builtins import OneHotEncoder, StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive
##SkLearn
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_score, cross_val_predict
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.decomposition import FastICA
from sklearn.ensemble import (
    AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import ElasticNetCV, SGDRegressor, RidgeCV
from sklearn.preprocessing import (
    FunctionTransformer, PolynomialFeatures, StandardScaler, Binarizer,
    MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
)

from sklearn.feature_selection import (
    SelectFwe, f_regression, SelectPercentile, VarianceThreshold, SelectFromModel
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.base import clone
from xgboost import XGBRegressor



def evaluate_pipelines_performance_averaged(
    OverAllResults,
    pipelines,
    Selected_Preprocessings,
    TestSets,
    Sample_ID,
    target,
    grouping_column,
    Index_column,
    Spectra_Start_Index
):
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
        # Set random state
        if hasattr(exported_pipeline, 'steps'):
                set_param_recursive(exported_pipeline.steps, 'random_state', 11)
        elif hasattr(exported_pipeline, 'random_state'):
                setattr(exported_pipeline, 'random_state', 11)
        Spectral_Preprocessing.append(Selected_Preprocessings[i])
        
        # Load data
        file_name = os.path.join("SelectedSpectra", Selected_Preprocessings[i] + ".csv")
        file = pd.read_csv(file_name, sep=",", index_col=Index_column)
        
        T_MAEs = []
        T_NMAEs = []
        T_R2s = []

        for t in TestSets:
            TestIndex = t
            Training_data = file[~file[Sample_ID].isin(TestIndex)]
            Training_data = Training_data.sort_values(by=[Sample_ID])
            
            # Get features and targets
            training_features = Training_data.iloc[:, Spectra_Start_Index:]
            training_target = Training_data[target]


            groups = Training_data[grouping_column]
            logo = LeaveOneGroupOut()
            model = clone(exported_pipeline)
            predictions = cross_val_predict(model, training_features, training_target, groups=groups, cv=logo, n_jobs=-1)

            Training_data_predictions = Training_data.copy()
            Training_data_predictions['predictions'] = predictions
            avg_predictions = Training_data_predictions.groupby(grouping_column)['predictions'].mean()
            avg_target = Training_data_predictions.groupby(grouping_column)[target].mean()

            T_MAEs.append(mean_absolute_error(avg_target, avg_predictions))
            T_NMAEs.append(mean_absolute_error(avg_target, avg_predictions) / (max(avg_target) - min(avg_target)))
            T_R2s.append(r2_score(avg_target, avg_predictions))

        end_time_n = time.time()
        training_times.append(end_time_n - start_time_n)
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
        "training_times": training_times
    }
    
############################

def evaluate_pipelines_spectra_averaged(
    OverAllResults,
    pipelines,
    Selected_Preprocessings,
    TestSets,
    Sample_ID,
    target,
    Index_column,
    Spectra_Start_Index
):
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
        # Set random state
        if hasattr(exported_pipeline, 'steps'):
                set_param_recursive(exported_pipeline.steps, 'random_state', 11)
        elif hasattr(exported_pipeline, 'random_state'):
                setattr(exported_pipeline, 'random_state', 11)
        Spectral_Preprocessing.append(Selected_Preprocessings[i])
        
        # Load data
        file_name = os.path.join("SelectedSpectra", Selected_Preprocessings[i] + ".csv")
        file = pd.read_csv(file_name, sep=",")
        
        T_MAEs = []
        T_NMAEs = []
        T_R2s = []

        for t in TestSets:
            TestIndex = t
            Training_data = file[~file[Sample_ID].isin(TestIndex)]
            Training_data = Training_data.sort_values(by=[Sample_ID])
            Spectra_Index=Training_data[Index_column].unique()
            predictions=[]
            GT=[]
            for si in Spectra_Index:
                testing_set=Training_data[Training_data[Index_column]==si]
                training_set=Training_data[~(Training_data[Index_column]==si)]
                #getting the features and targets
                training_features=training_set.iloc[:,Spectra_Start_Index:]
                training_target=training_set[target]
                testing_features=testing_set.iloc[:,Spectra_Start_Index:]
                testing_features = testing_features.mean(axis=0).to_frame().T
                testing_target=testing_set[target].mean()
                model = clone(exported_pipeline)
                tpot=model.fit(training_features, training_target)
                #Statitics and predictions collections
                predictions.append(tpot.predict(testing_features).item())
                GT.append(testing_target)
            T_MAEs.append(mean_absolute_error(GT, predictions))
            T_NMAEs.append(mean_absolute_error(GT, predictions)/(max(GT)-min(GT)))
            T_R2s.append(r2_score(GT, predictions))  
        end_time_n = time.time()
        training_times.append(end_time_n - start_time_n)
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
        "training_times": training_times
    }
#######################################

def pipeline_testsets_evaluation(
    Selected_Preprocessings,
    Selected_Pipelines,
    TestSets,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    data_folder="SelectedSpectra"
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

    start_time = time.time()

    # Output collectors
    Pipelineindex, Preprocessing, Set = [], [], []
    Test_Samples_ID, Groundtruths, Predictions = [], [], []
    training_times, MAEs, NMAEs, R2s = [], [], [], []

    for p, s, pi in zip(range(1, len(Selected_Preprocessings)+1), Selected_Preprocessings, Selected_Pipelines):
        print(f"Evaluating pipeline {p}/{len(Selected_Preprocessings)}")

        # Load file
        file_path = os.path.join(data_folder, f"{s}.csv")
        file = pd.read_csv(file_path, sep=",", index_col="Spectra")

        for i, TestIndex in enumerate(TestSets):
            # Split training and testing
            Training_data = file[~file[Sample_ID].isin(TestIndex)]
            Testing_data = file[file[Sample_ID].isin(TestIndex)]

            # Features and target for training
            training_features = Training_data.iloc[:, Spectra_Start_Index:]
            training_target = Training_data[target]

            # Average testing features and target by sample
            testing_target = Testing_data.groupby(Sample_ID)[target].mean()
            testing_features = Testing_data.groupby(Sample_ID).apply(
                lambda x: x.iloc[:, Spectra_Start_Index:].mean()
            )

            # Store test IDs and ground truths
            Groundtruths.append(testing_target)
            Test_Samples_ID.append(testing_target.index.tolist())

            # Set random state if possible
            if hasattr(pi, 'steps'):
                set_param_recursive(pi.steps, 'random_state', 11)
            elif hasattr(pi, 'random_state'):
                pi.random_state = 11

            # Train
            t_start = time.time()
            pi.fit(training_features, training_target)
            t_end = time.time()

            training_times.append(t_end - t_start)

            # Predict
            prediction = pi.predict(testing_features)
            Predictions.append(pd.Series(prediction, index=testing_target.index))

            # Metrics
            mae = mean_absolute_error(testing_target, prediction)
            nmae = mae / (max(testing_target) - min(testing_target))
            r2 = r2_score(testing_target, prediction)

            MAEs.append(mae)
            NMAEs.append(nmae)
            R2s.append(r2)

            # Record identifiers
            Pipelineindex.append(p)
            Preprocessing.append(s)
            Set.append(i + 1)

    # Create result DataFrames
    metrics_df = pd.DataFrame({
        'Pipeline': Pipelineindex,
        'Preprocessing': Preprocessing,
        'Set': Set,
        'MAE': MAEs,
        'NMAE': NMAEs,
        'R2': R2s,
        'training_time': training_times })
    predictions_df = pd.DataFrame({
        'Pipeline': Pipelineindex,
        'Preprocessing': Preprocessing,
        'Set': Set,
        'Sample_IDs': Test_Samples_ID,
        'Groundtruths': Groundtruths,
        'Predictions': Predictions
    })

    print("Total Time Elapsed: {:.2f} min".format((time.time() - start_time) / 60))
    return metrics_df, predictions_df
    

###########################

def pipeline_LOOCV_evaluation(
    Selected_Preprocessings,
    Selected_Pipelines,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    data_folder="SelectedSpectra"
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

    for p_idx, (preprocessing_name, pipeline) in enumerate(zip(Selected_Preprocessings, Selected_Pipelines), start=1):
        print(f"Evaluating pipeline {p_idx}/{len(Selected_Preprocessings)}")

        file_path = os.path.join(data_folder, f"{preprocessing_name}.csv")
        file = pd.read_csv(file_path, sep=",", index_col="Spectra")

        for test_sample in file[Sample_ID].unique():
            # Train/test split
            Training_data = file[file[Sample_ID] != test_sample]
            Testing_data = file[file[Sample_ID] == test_sample]

            training_features = Training_data.iloc[:, Spectra_Start_Index:]
            training_target = Training_data[target]

            # Averaging replicates for test sample
            testing_target = Testing_data.groupby(Testing_data[Sample_ID])[target].mean()
            testing_features = Testing_data.iloc[:, Spectra_Start_Index:].groupby(Testing_data[Sample_ID]).mean()

            # Set random state
            if hasattr(pipeline, 'steps'):
                set_param_recursive(pipeline.steps, 'random_state', 11)
            elif hasattr(pipeline, 'random_state'):
                pipeline.random_state = 11

            # Train
            t_start = time.time()
            pipeline.fit(training_features, training_target)
            training_time = time.time() - t_start

            # Predict
            prediction = pipeline.predict(testing_features)

            # Store results (flattened)
            for sample_id in testing_target.index:
                results.append({
                    'Pipeline': p_idx,
                    'Preprocessing': preprocessing_name,
                    'Sample_IDs': sample_id,
                    'Groundtruths': testing_target.loc[sample_id],
                    'Predictions': prediction[0],  # Only one prediction here due to LOOCV
                    'Training_Time': training_time
                })

    print("Total Time Elapsed: {:.2f} min".format((time.time() - start_time) / 60))
    return pd.DataFrame(results)
#########################################################################

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold

def pipeline_LOOCV_evaluation_with_residual_correction(
    Selected_Preprocessings,
    Selected_Pipelines,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    data_folder="SelectedSpectra"
):
    import time
    import os
    import pandas as pd
    from sklearn.utils import all_estimators

    start_time = time.time()
    results = []

    for p_idx, (preprocessing_name, pipeline) in enumerate(zip(Selected_Preprocessings, Selected_Pipelines), start=1):
        print(f"Evaluating pipeline {p_idx}/{len(Selected_Preprocessings)}")

        file_path = os.path.join(data_folder, f"{preprocessing_name}.csv")
        file = pd.read_csv(file_path, sep=",", index_col="Spectra")

        for test_sample in file[Sample_ID].unique():
            # LOOCV split
            Training_data = file[file[Sample_ID] != test_sample]
            Testing_data = file[file[Sample_ID] == test_sample]

            training_features = Training_data.iloc[:, Spectra_Start_Index:]
            training_target = Training_data[target]

            # Averaging replicates
            testing_target = Testing_data.groupby(Testing_data[Sample_ID])[target].mean()
            testing_features = Testing_data.iloc[:, Spectra_Start_Index:].groupby(Testing_data[Sample_ID]).mean()

            # Reset random state if possible
            if hasattr(pipeline, 'steps'):
                set_param_recursive(pipeline.steps, 'random_state', 11)
            elif hasattr(pipeline, 'random_state'):
                pipeline.random_state = 11

            # Fit main model
            t_start = time.time()
            pipeline.fit(training_features, training_target)
            training_time = time.time() - t_start

            # Predict test sample
            prediction = pipeline.predict(testing_features)

            # Residual correction
            try:
                # Use inner 5-fold CV to get out-of-fold predictions on training data
                oof_preds = cross_val_predict(pipeline, training_features, training_target, cv=KFold(n_splits=5, shuffle=True, random_state=11))
                residuals = training_target.values - oof_preds

                # Fit correction model (e.g., residual ~ y_true)
                residual_model = LinearRegression()
                residual_model.fit(training_target.values.reshape(-1, 1), residuals)

                # Apply correction to prediction
                correction = residual_model.predict(testing_target.values.reshape(-1, 1))
                corrected_prediction = prediction[0] - correction[0]
            except Exception as e:
                print(f"Residual correction failed for sample {test_sample}: {e}")
                corrected_prediction = None  # fallback

            for sample_id in testing_target.index:
                results.append({
                    'Pipeline': p_idx,
                    'Preprocessing': preprocessing_name,
                    'Sample_IDs': sample_id,
                    'Groundtruths': testing_target.loc[sample_id],
                    'Predictions': prediction[0],
                    'Corrected_Predictions': corrected_prediction,
                    'Training_Time': training_time
                })

    print("Total Time Elapsed: {:.2f} min".format((time.time() - start_time) / 60))
    return pd.DataFrame(results)


########################################################################

def aggregate_sample_predictions(predictions_long, grouping_column="Sample_IDs", prediction_column="Predictions",groundtruth_column="Groundtruths",Set_column= "Set", z_thresh=3.5):
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

        results.append({
            "Set": group[Set_column].iloc[0],
            "Test_Samples_ID": sample_id,
            "Groundtruths": group[groundtruth_column].iloc[0],
            "Predictions_Means": np.mean(prediction_values),
            "Predictions_Mean_Corrected": np.mean(filtered),
            "Predictions_Medians": np.median(prediction_values),
            "Predictions_Medians_Corrected": np.median(filtered),
            # Optionally add other fields like training time if present
        })

    return pd.DataFrame(results)

##################################################################################

def evaluate_predictions(true_values, predictions):
    return {
        'r2': r2_score(true_values, predictions),
        'MAE': mean_absolute_error(true_values, predictions),
        'RMSE': sqrt(mean_squared_error(true_values, predictions)),
        'r': pearsonr(true_values, predictions)[0],
        'p_value': pearsonr(true_values, predictions)[1],
        'NMAE (%)': 100 * mean_absolute_error(true_values, predictions) /(np.max(true_values)-np.min(true_values)),
        'MAPE': mean_absolute_percentage_error(true_values, predictions)
        
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
def FeatureImportanceEvaluation2(
    Selected_Preprocessings,
    Selected_Pipelines,
    TestSets,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    Excluded_Feature=None,
    Prediction_Type='Predictions_Medians',
    data_folder="SelectedSpectra"
):
    """
    Evaluate multiple ML pipelines with specific features replaced by their mean (interval ablation by substitution).
    """

    # Output collectors
    Pipelineindex, Preprocessing, Set = [], [], []
    Test_Samples_ID, Groundtruths, Predictions = [], [], []
    training_times = []

    for p, s, pi in zip(range(1, len(Selected_Preprocessings) + 1), Selected_Preprocessings, Selected_Pipelines):
        # Load preprocessed spectra file
        file_path = os.path.join(data_folder, f"{s}.csv")
        file = pd.read_csv(file_path, sep=",", index_col="Spectra")

        for i, TestIndex in enumerate(TestSets):
            # Train-test split
            Training_data = file[~file[Sample_ID].isin(TestIndex)].copy()
            Testing_data = file[file[Sample_ID].isin(TestIndex)].copy()

            # Replace selected features in testing set with training mean
            if Excluded_Feature:
                for feat in Excluded_Feature:
                    mean_val = Training_data[feat].mean()
                    Testing_data.loc[:, feat] = mean_val

            # Extract training and testing features
            training_features = Training_data.iloc[:, Spectra_Start_Index:]
            training_target = Training_data[target]

            testing_target = Testing_data.groupby(Sample_ID)[target].mean()
            testing_features = Testing_data.groupby(Sample_ID).apply(
                lambda x: x.iloc[:, Spectra_Start_Index:].mean()
            )

            # Store test IDs and ground truths
            Groundtruths.append(testing_target)
            Test_Samples_ID.append(testing_target.index.tolist())

            # Set random state
            if hasattr(pi, 'steps'):
                set_param_recursive(pi.steps, 'random_state', 11)
            elif hasattr(pi, 'random_state'):
                pi.random_state = 11

            # Train
            t_start = time.time()
            pi.fit(training_features, training_target)
            t_end = time.time()
            training_times.append(t_end - t_start)

            # Predict
            prediction = pi.predict(testing_features)
            Predictions.append(pd.Series(prediction, index=testing_target.index))

            # Record run info
            Pipelineindex.append(p)
            Preprocessing.append(s)
            Set.append(i + 1)

    # Create prediction summary
    predictions_df = pd.DataFrame({
        'Pipeline': Pipelineindex,
        'Preprocessing': Preprocessing,
        'Set': Set,
        'Sample_IDs': Test_Samples_ID,
        'Groundtruths': Groundtruths,
        'Predictions': Predictions
    })

    # Flatten lists for evaluation
    predictions_SA_long = predictions_df.explode(['Sample_IDs', 'Groundtruths', 'Predictions']).reset_index(drop=True)

    # Aggregate predictions
    from evaluation_functions import aggregate_sample_predictions, evaluate_predictions
    Final_Results_5CV_ALL_SA = aggregate_sample_predictions(predictions_SA_long)

    # Evaluate predictions
    results_summary_SA = {}
    prediction_columns = ["Predictions_Medians", "Predictions_Means", "Predictions_Mean_Corrected", "Predictions_Medians_Corrected"]
    for col in prediction_columns:
        results_summary_SA[col] = evaluate_predictions(Final_Results_5CV_ALL_SA["Groundtruths"], Final_Results_5CV_ALL_SA[col])

    R2n = results_summary_SA[Prediction_Type]["r2"]
    MAEn = results_summary_SA[Prediction_Type]["MAE"]
    Rn = results_summary_SA[Prediction_Type]["r"]

    return Excluded_Feature, R2n, MAEn, Rn

##################################################################################
def FeatureImportanceEvaluation_Retrain(
    Selected_Preprocessings,
    Selected_Pipelines,
    TestSets,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    Excluded_Feature=None,
    Prediction_Type='Predictions_Medians',
    data_folder="SelectedSpectra"
):
    """
    Interval ablation by mean-replacement in BOTH train and test (retrain per fold).

    For each preprocessing 's' and its pipeline 'pi':
      - Load s.csv
      - For each external split in TestSets:
           * Split train/test
           * For each feature f in Excluded_Feature:
                mean_f = TRAIN[f].mean()
                TRAIN[f] = mean_f
                TEST[f]  = mean_f
           * Train cloned pipeline on TRAIN
           * Predict on averaged TEST replicates
      - Aggregate all fold predictions, compute metrics and return (Excluded_Feature, R2, MAE, r)
    """

    # Output collectors
    Pipelineindex, Preprocessing, Set = [], [], []
    Test_Samples_ID, Groundtruths, Predictions = [], [], []
    training_times = []

    for p, s, pi in zip(range(1, len(Selected_Preprocessings) + 1), Selected_Preprocessings, Selected_Pipelines):
        # Load preprocessed spectra file
        file_path = os.path.join(data_folder, f"{s}.csv")
        file = pd.read_csv(file_path, sep=",", index_col="Spectra")

        for i, TestIndex in enumerate(TestSets):
            # Train-test split
            Training_data = file[~file[Sample_ID].isin(TestIndex)].copy()
            Testing_data = file[file[Sample_ID].isin(TestIndex)].copy()

            # Replace selected features in testing set with training mean
            if Excluded_Feature:
                for feat in Excluded_Feature:
                    mean_val = Training_data[feat].mean()
                    Testing_data.loc[:, feat] = mean_val
                    Training_data.loc[:, feat] = mean_val

            # Extract training and testing features
            training_features = Training_data.iloc[:, Spectra_Start_Index:]
            training_target = Training_data[target]

            testing_target = Testing_data.groupby(Sample_ID)[target].mean()
            testing_features = Testing_data.groupby(Sample_ID).apply(
                lambda x: x.iloc[:, Spectra_Start_Index:].mean()
            )

            # Store test IDs and ground truths
            Groundtruths.append(testing_target)
            Test_Samples_ID.append(testing_target.index.tolist())
            model = clone(pi)
            # Set random state
            if hasattr(model, 'steps'):
                set_param_recursive(model.steps, 'random_state', 11)
            elif hasattr(model, 'random_state'):
                model.random_state = 11

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
    predictions_df = pd.DataFrame({
        'Pipeline': Pipelineindex,
        'Preprocessing': Preprocessing,
        'Set': Set,
        'Sample_IDs': Test_Samples_ID,
        'Groundtruths': Groundtruths,
        'Predictions': Predictions
    })

    # Flatten lists for evaluation
    predictions_SA_long = predictions_df.explode(['Sample_IDs', 'Groundtruths', 'Predictions']).reset_index(drop=True)

    # Aggregate predictions
    from evaluation_functions import aggregate_sample_predictions, evaluate_predictions
    Final_Results_5CV_ALL_SA = aggregate_sample_predictions(predictions_SA_long)

    # Evaluate predictions
    results_summary_SA = {}
    prediction_columns = ["Predictions_Medians", "Predictions_Means", "Predictions_Mean_Corrected", "Predictions_Medians_Corrected"]
    for col in prediction_columns:
        results_summary_SA[col] = evaluate_predictions(Final_Results_5CV_ALL_SA["Groundtruths"], Final_Results_5CV_ALL_SA[col])

    R2n = results_summary_SA[Prediction_Type]["r2"]
    MAEn = results_summary_SA[Prediction_Type]["MAE"]
    Rn = results_summary_SA[Prediction_Type]["r"]

    return Excluded_Feature, R2n, MAEn, Rn

##################################################################################
def EnsembelML(
    Selected_Preprocessings,
    Selected_Pipelines,
    TestSets,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    Prediction_Type='Predictions_Medians',
    data_folder="SelectedSpectra"
):
    """
    Evaluate an ensemble model based on multiple ML pipelines.
    """

    # Output collectors
    Pipelineindex, Preprocessing, Set = [], [], []
    Test_Samples_ID, Groundtruths, Predictions = [], [], []
    training_times = []

    for p, s, pi in zip(range(1, len(Selected_Preprocessings) + 1), Selected_Preprocessings, Selected_Pipelines):
        # Load preprocessed spectra file
        file_path = os.path.join(data_folder, f"{s}.csv")
        file = pd.read_csv(file_path, sep=",", index_col="Spectra")

        for i, TestIndex in enumerate(TestSets):
            # Train-test split
            Training_data = file[~file[Sample_ID].isin(TestIndex)].copy()
            Testing_data = file[file[Sample_ID].isin(TestIndex)].copy()

            # Extract training and testing features
            training_features = Training_data.iloc[:, Spectra_Start_Index:]
            training_target = Training_data[target]

            testing_target = Testing_data.groupby(Sample_ID)[target].mean()
            testing_features = Testing_data.groupby(Sample_ID).apply(
                lambda x: x.iloc[:, Spectra_Start_Index:].mean()
            )

            # Store test IDs and ground truths
            Groundtruths.append(testing_target)
            Test_Samples_ID.append(testing_target.index.tolist())

            # Set random state
            if hasattr(pi, 'steps'):
                set_param_recursive(pi.steps, 'random_state', 11)
            elif hasattr(pi, 'random_state'):
                pi.random_state = 11

            # Train
            t_start = time.time()
            pi.fit(training_features, training_target)
            t_end = time.time()
            training_times.append(t_end - t_start)

            # Predict
            prediction = pi.predict(testing_features)
            Predictions.append(pd.Series(prediction, index=testing_target.index))

            # Record run info
            Pipelineindex.append(p)
            Preprocessing.append(s)
            Set.append(i + 1)

    # Create prediction summary
    predictions_df = pd.DataFrame({
        'Pipeline': Pipelineindex,
        'Preprocessing': Preprocessing,
        'Set': Set,
        'Sample_IDs': Test_Samples_ID,
        'Groundtruths': Groundtruths,
        'Predictions': Predictions
    })

    # Flatten lists for evaluation
    predictions_SA_long = predictions_df.explode(['Sample_IDs', 'Groundtruths', 'Predictions']).reset_index(drop=True)

    # Aggregate predictions
    from evaluation_functions import aggregate_sample_predictions, evaluate_predictions
    Final_Results_5CV_ALL_SA = aggregate_sample_predictions(predictions_SA_long)

    # Evaluate predictions
    results_summary_SA = {}
    prediction_columns = ["Predictions_Medians", "Predictions_Means", "Predictions_Mean_Corrected", "Predictions_Medians_Corrected"]
    for col in prediction_columns:
        results_summary_SA[col] = evaluate_predictions(Final_Results_5CV_ALL_SA["Groundtruths"], Final_Results_5CV_ALL_SA[col])

    R2n = results_summary_SA[Prediction_Type]["r2"]
    MAEn = results_summary_SA[Prediction_Type]["MAE"]
    Rn = results_summary_SA[Prediction_Type]["r"]

    return R2n, MAEn, Rn, Final_Results_5CV_ALL_SA[col]
