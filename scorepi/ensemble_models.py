#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to combine forecasts into an ensemble model.

Author: Guillaume St-Onge <stongeg1@gmail.com>
"""

import numpy as np
import pandas as pd
from .base_classes import Predictions


def median_ensemble(predictions_list,**kwargs):
    """median_ensemble.

    Parameters
    ----------
    predictions_list : list of Predictions object
    """
    #we assume all predictions have the same columns
    value_col = predictions_list[0].value_col
    quantile_col = predictions_list[0].quantile_col
    type_col = predictions_list[0].type_col
    t_col = predictions_list[0].t_col
    other_ind_cols = predictions_list[0].other_ind_cols
    ind_cols = predictions_list[0].ind_cols

    #concatenate the predictions
    all_predictions = pd.concat(predictions_list)
    #get median for quantiles
    ensemble_predictions = all_predictions.groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).median(numeric_only=True).reset_index()

    ensemble_predictions = Predictions(ensemble_predictions,value_col=value_col,quantile_col=quantile_col,
                                       type_col=type_col,t_col=t_col,other_ind_cols=other_ind_cols)

    return ensemble_predictions

def mean_ensemble(predictions_list,**kwargs):
    """mean_ensemble.

    Parameters
    ----------
    predictions_list : list of Predictions object
    """
    #we assume all predictions have the same columns
    value_col = predictions_list[0].value_col
    quantile_col = predictions_list[0].quantile_col
    type_col = predictions_list[0].type_col
    t_col = predictions_list[0].t_col
    other_ind_cols = predictions_list[0].other_ind_cols
    ind_cols = predictions_list[0].ind_cols

    #concatenate the predictions
    all_predictions = pd.concat(predictions_list)
    #get mean for quantiles
    ensemble_predictions = all_predictions.groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).mean(numeric_only=True).reset_index()

    ensemble_predictions = Predictions(ensemble_predictions,value_col=value_col,quantile_col=quantile_col,
                                       type_col=type_col,t_col=t_col,other_ind_cols=other_ind_cols)

    return ensemble_predictions

def extreme_ensemble(predictions_list,**kwargs):
    """median_ensemble.

    Parameters
    ----------
    predictions_list : list of Predictions object
    """
    #we assume all predictions have the same columns
    value_col = predictions_list[0].value_col
    quantile_col = predictions_list[0].quantile_col
    type_col = predictions_list[0].type_col
    t_col = predictions_list[0].t_col
    other_ind_cols = predictions_list[0].other_ind_cols
    ind_cols = predictions_list[0].ind_cols

    #concatenate the predictions
    all_predictions = pd.concat(predictions_list)

    #get min/max for quantiles
    ensemble_predictions_low = all_predictions[all_predictions[quantile_col] < 0.5].groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).min().reset_index()
    ensemble_predictions_upp = all_predictions[all_predictions[quantile_col] > 0.5].groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).max().reset_index()
    ensemble_predictions_med = all_predictions[np.isclose(all_predictions[quantile_col],0.5)].groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).median().reset_index()
    ensemble_predictions_point = all_predictions[all_predictions[type_col] == 'point'].groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).median().reset_index()

    ensemble_predictions = pd.concat([ensemble_predictions_low,ensemble_predictions_med,
                                      ensemble_predictions_point, ensemble_predictions_upp])

    ensemble_predictions = Predictions(ensemble_predictions,value_col=value_col,quantile_col=quantile_col,
                                       type_col=type_col,t_col=t_col,other_ind_cols=other_ind_cols)

    return ensemble_predictions

def weighted_ensemble(predictions_list, weights=None, **kwargs):
    """weighted_ensemble.
    Parameters
    ----------
    predictions_list : list of Predictions objects
    weights : list of floats, optional
        Weight for each predictions object in predictions_list.
        Must be the same length as predictions_list.
        Weights do not need to be normalized — they will be normalized internally.
        If None, defaults to equal weights (equivalent to a simple mean).
    """
    if weights is None:
        weights = [1.0] * len(predictions_list)

    if len(weights) != len(predictions_list):
        raise ValueError(
            f"Length of weights ({len(weights)}) must match "
            f"length of predictions_list ({len(predictions_list)})"
        )
    
    # Normalize weights so they sum to 1
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    
    #we assume all predictions have the same columns
    value_col = predictions_list[0].value_col
    quantile_col = predictions_list[0].quantile_col
    type_col = predictions_list[0].type_col
    t_col = predictions_list[0].t_col
    other_ind_cols = predictions_list[0].other_ind_cols
    ind_cols = predictions_list[0].ind_cols

    # Tag each predictions dataframe with its normalized weight
    weighted_predictions_list = []
    for pred, w in zip(predictions_list, norm_weights):
        df = pred.copy() if isinstance(pred, pd.DataFrame) else pd.concat([pred])
        df = df.copy()
        df["_weight"] = w
        weighted_predictions_list.append(df)

    all_predictions = pd.concat(weighted_predictions_list, ignore_index=True)

    # Compute weighted value per row, then sum within each group
    all_predictions["_weighted_value"] = (
        all_predictions[value_col] * all_predictions["_weight"]
    )

    ensemble_predictions = (
        all_predictions
        .groupby(by=ind_cols + [type_col, quantile_col], dropna=False)["_weighted_value"]
        .sum()
        .reset_index()
        .rename(columns={"_weighted_value": value_col})
    )

    ensemble_predictions = Predictions(
        ensemble_predictions,
        value_col=value_col,
        quantile_col=quantile_col,
        type_col=type_col,
        t_col=t_col,
        other_ind_cols=other_ind_cols,
    )
    return ensemble_predictions