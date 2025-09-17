import pandas as pd
import numpy as np

def _validate_inputs(df, aggregation_column, order_by, agg_func, percentile_k):
    if aggregation_column not in df.columns:
        raise ValueError(f"Column '{aggregation_column}' not found in DataFrame.")
    if order_by is None:
        raise ValueError("The 'order_by' column must be specified.")
    if order_by not in df.columns:
        raise ValueError(f"Order by column '{order_by}' not found in DataFrame.")
    if agg_func == 'percentile' and percentile_k is None:
        raise ValueError("You must specify 'percentile_k' when agg_func is 'percentile'.")

def _get_rolling_series(series, window, ignore_nulls):
    if ignore_nulls:
        return series.fillna(np.nan).rolling(window=window, min_periods=1)
    return series.rolling(window=window, min_periods=1)

def apply_rolling_aggregation(
    df,
    aggregation_column,
    agg_func='max',
    percentile_k=None,
    partition_by=None,
    order_by=None,
    window=3,
    ignore_nulls=True
):
    """
    Apply a single rolling aggregation to a DataFrame.

    Parameters:
        df (pd.DataFrame): Input data.
        aggregation_column (str): Column to aggregate.
        agg_func (str): Aggregation type: 'max', 'min', 'sum', 'mean', 'std', 'percentile'.
        percentile_k (float): Required if agg_func='percentile'; value between 0 and 100.
        partition_by (str or list): Column(s) to partition by (optional).
        order_by (str): Column to order within partitions.
        window (int): Rolling window size.
        ignore_nulls (bool): Whether to ignore nulls in the window.

    Returns:
        pd.DataFrame: DataFrame with a new column containing the rolling result.
    """
    _validate_inputs(df, aggregation_column, order_by, agg_func, percentile_k)
    
    df = df.copy()

    if partition_by:
        if isinstance(partition_by, str):
            partition_by = [partition_by]
        grouped = df.sort_values(order_by).groupby(partition_by, group_keys=False)
    else:
        grouped = [(None, df.sort_values(order_by))]

    result_frames = []

    for _, group in grouped:
        col = group[aggregation_column]
        rolling_obj = _get_rolling_series(col, window, ignore_nulls)

        result = group.copy()

        if agg_func == 'max':
            result[f'{aggregation_column}_roll_max'] = rolling_obj.max().values
        elif agg_func == 'min':
            result[f'{aggregation_column}_roll_min'] = rolling_obj.min().values
        elif agg_func == 'sum':
            result[f'{aggregation_column}_roll_sum'] = rolling_obj.sum().values
        elif agg_func == 'mean':
            result[f'{aggregation_column}_roll_mean'] = rolling_obj.mean().values
        elif agg_func == 'std':
            result[f'{aggregation_column}_roll_std'] = rolling_obj.std().values
        elif agg_func == 'percentile':
            percentile_col = rolling_obj.quantile(percentile_k / 100.0).values
            result[f'{aggregation_column}_roll_p{percentile_k}'] = percentile_col
        else:
            raise ValueError(f"Unknown aggregation type: {agg_func}")

        result_frames.append(result)

    final_df = pd.concat(result_frames).sort_index()
    return final_df
