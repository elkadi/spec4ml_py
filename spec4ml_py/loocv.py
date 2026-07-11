"""LOOCV evaluation helpers with optional subset and exclusion controls."""

import time

import pandas as pd

from .evaluation_functions import (
    _average_by_sample,
    _clone_with_random_state,
    _load_spectra,
    _xy,
)


def pipeline_LOOCV_evaluation(
    Selected_Preprocessings,
    Selected_Pipelines,
    Sample_ID,
    target,
    Spectra_Start_Index=16,
    data_folder="SelectedSpectra",
    evaluation_ids=None,
    exclude_column=None,
    exclude_ids=None,
    index_col="Spectra",
):
    """
    Evaluate multiple preprocessing-pipeline pairs using leave-one-ID-out CV.

    Technical replicates sharing the same ``Sample_ID`` value are removed
    together, averaged before prediction, and returned as one prediction.

    Parameters
    ----------
    Selected_Preprocessings : sequence of str
        Preprocessing file names without the ``.csv`` extension.
    Selected_Pipelines : sequence of estimators
        Pipelines corresponding to ``Selected_Preprocessings``.
    Sample_ID : str
        Column defining the leave-one-out unit. For leave-one-Spectra-out
        evaluation, pass ``Sample_ID="Spectra"`` and ``index_col=None``.
    target : str
        Target column name.
    Spectra_Start_Index : int, default=16
        Positional index of the first spectral feature column.
    data_folder : str, default="SelectedSpectra"
        Folder containing the preprocessing CSV files.
    evaluation_ids : iterable, optional
        Restrict evaluation to these leave-one-out IDs. All other eligible IDs
        remain available for training. This is useful for splitting a long
        LOOCV run into non-overlapping ranges.
    exclude_column : str, optional
        Column used to remove observations before LOOCV, for example ``Sample``
        when excluding one outer test fold.
    exclude_ids : iterable, optional
        Values from ``exclude_column`` to remove before LOOCV.
    index_col : str or None, default="Spectra"
        Column passed to ``pandas.read_csv`` as the index. Use ``None`` when the
        leave-one-out identifier itself must remain a normal dataframe column.

    Returns
    -------
    pandas.DataFrame
        One row per pipeline and evaluated leave-one-out ID, with columns
        ``Pipeline``, ``Preprocessing``, ``Sample_IDs``, ``Groundtruths``,
        ``Predictions``, and ``Training_Time``.

    Notes
    -----
    ``evaluation_ids`` controls only which IDs are tested. It does not remove
    the other IDs from training. In contrast, ``exclude_ids`` removes matching
    rows completely before any leave-one-out split is created.
    """
    if len(Selected_Preprocessings) != len(Selected_Pipelines):
        raise ValueError(
            "Selected_Preprocessings and Selected_Pipelines must have "
            "the same length."
        )

    if (exclude_column is None) != (exclude_ids is None):
        raise ValueError(
            "exclude_column and exclude_ids must either both be provided "
            "or both be None."
        )

    start_time = time.time()
    results = []

    for p_idx, (preprocessing_name, pipeline) in enumerate(
        zip(Selected_Preprocessings, Selected_Pipelines), start=1
    ):
        print(
            f"Evaluating pipeline {p_idx}/{len(Selected_Preprocessings)}",
            flush=True,
        )

        file = _load_spectra(
            preprocessing_name,
            data_folder=data_folder,
            index_col=index_col,
        )

        if exclude_column is not None:
            if exclude_column not in file.columns:
                raise KeyError(
                    f"Exclusion column '{exclude_column}' was not found in "
                    f"{preprocessing_name}.csv."
                )
            file = file.loc[~file[exclude_column].isin(exclude_ids)].copy()

        if Sample_ID not in file.columns:
            raise KeyError(
                f"LOOCV column '{Sample_ID}' was not found in "
                f"{preprocessing_name}.csv. If it is currently used as the "
                "dataframe index, call the function with index_col=None."
            )

        available_ids = list(pd.unique(file[Sample_ID]))

        if evaluation_ids is None:
            ids_to_evaluate = available_ids
        else:
            ids_to_evaluate = list(evaluation_ids)
            missing_ids = [
                evaluation_id
                for evaluation_id in ids_to_evaluate
                if evaluation_id not in available_ids
            ]
            if missing_ids:
                raise ValueError(
                    "The following evaluation_ids were not found after "
                    f"exclusion: {missing_ids}"
                )

        for test_sample in ids_to_evaluate:
            Training_data = file.loc[file[Sample_ID] != test_sample]
            Testing_data = file.loc[file[Sample_ID] == test_sample]

            training_features, training_target = _xy(
                Training_data,
                target,
                Spectra_Start_Index,
            )
            testing_features, testing_target = _average_by_sample(
                Testing_data,
                Sample_ID,
                target,
                Spectra_Start_Index,
            )

            model = _clone_with_random_state(pipeline)

            t_start = time.time()
            model.fit(training_features, training_target)
            training_time = time.time() - t_start

            prediction = model.predict(testing_features)

            for prediction_index, sample_id in enumerate(testing_target.index):
                results.append(
                    {
                        "Pipeline": p_idx,
                        "Preprocessing": preprocessing_name,
                        "Sample_IDs": sample_id,
                        "Groundtruths": testing_target.loc[sample_id],
                        "Predictions": prediction[prediction_index],
                        "Training_Time": training_time,
                    }
                )

    print(
        "Total Time Elapsed: {:.2f} min".format(
            (time.time() - start_time) / 60
        ),
        flush=True,
    )
    return pd.DataFrame(results)
