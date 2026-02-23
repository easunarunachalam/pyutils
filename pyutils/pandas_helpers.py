from copy import deepcopy
import numpy as np
import pandas as pd

def normalize_column(df, col_for_scale, other_cols_to_scale=[], how=np.mean, indices_for_scalecomp=None):

    if indices_for_scalecomp is None:
        scale_factor = how(df[col_for_scale].values)
    else:
        scale_factor = how(df.loc[indices_for_scalecomp, col_for_scale].values)

    df[col_for_scale] = df[col_for_scale] / scale_factor

    if not (isinstance(other_cols_to_scale, list) or isinstance(other_cols_to_scale, np.ndarray)):
        other_cols_to_scale = [other_cols_to_scale,]

    for col_name in other_cols_to_scale:
        df[col_name] = df[col_name] / scale_factor

    return df

def propagate_error_product(df, col_a, col_b, mean_suffix=" mean", err_suffix=" sem"):
    
    product_mean = df[col_a + mean_suffix] * df[col_b + mean_suffix]

    a_frac_err = df[col_a + err_suffix] / df[col_a + mean_suffix]
    b_frac_err = df[col_b + err_suffix] / df[col_b + mean_suffix]
    product_frac_err = np.sqrt(a_frac_err**2 + b_frac_err**2)

    product_err = product_frac_err * product_mean

    return product_mean, product_err

def propagate_error_quotient(df, col_a, col_b, mean_suffix=" mean", err_suffix=" sem"):
    
    quotient_mean = df[col_a + mean_suffix] / df[col_b + mean_suffix]

    a_frac_err = df[col_a + err_suffix] / df[col_a + mean_suffix]
    b_frac_err = df[col_b + err_suffix] / df[col_b + mean_suffix]
    quotient_frac_err = np.sqrt(a_frac_err**2 + b_frac_err**2)
    
    quotient_err = quotient_frac_err * quotient_mean

    return quotient_mean, quotient_err

def add_col_prefix(df, prefix):
    new_column_names = [prefix + col for col in df.columns]
    df2 = deepcopy(df)
    df2.columns = new_column_names
    return df2

def merge_multiple(*dataframes, **kwargs):
    """
    Merge multiple pandas DataFrames using a common key or set of keys.

    This function performs sequential merging of the input DataFrames using 
    `pandas.merge`, optionally allowing for dynamic renaming of columns prior to merging.

    Parameters
    ----------
    *dataframes : pandas.DataFrame
        A variable number of pandas DataFrames to merge. The first DataFrame 
        serves as the base, and each subsequent DataFrame is merged in order.

    **kwargs : keyword arguments
        Additional keyword arguments passed directly to `pandas.merge`. 
        Common arguments include:
            - on : str or list
                Column(s) to join on.
            - how : str, default 'inner'
                Type of merge to be performed (e.g., 'inner', 'outer', 'left', 'right').

        Special keyword:
        - col_renaming_map : dict, optional
            A dictionary mapping original column names to lists of new names 
            for each DataFrame. This is useful for disambiguating columns with 
            the same name across DataFrames before merging.

            Example:
                col_renaming_map = {
                    "shared_col": ["df1_col", "df2_col", "df3_col"]
                }

    Returns
    -------
    pandas.DataFrame
        A single merged DataFrame resulting from sequential merges of the input DataFrames.

    Example
    -------
    >>> merge_multiple(df1, df2, df3, on="id", how="outer")
    >>> merge_multiple(df1, df2, on="id", col_renaming_map={"score": ["score1", "score2"]})
    """
    
    dataframes = list(dataframes)

    if "col_renaming_map" in kwargs:

        col_renaming_map = kwargs.get("col_renaming_map")
        kwargs.pop("col_renaming_map")

        col_renaming_map_keys = list(col_renaming_map.keys())
        for i in range(len(col_renaming_map)):
            ikey = col_renaming_map_keys[i]
            for j in range(len(dataframes)):
                dataframes[j] = dataframes[j].rename({ikey: col_renaming_map[ikey][j]}, axis=1)

                # move this column to the end
                column_to_move = col_renaming_map[ikey][j]
                new_order = [col for col in dataframes[j].columns if col != column_to_move] + [column_to_move]
                dataframes[j] = dataframes[j][new_order]

    df_merge = dataframes[0]

    for i, df in enumerate(dataframes[1:]):
        df_merge = pd.merge(left=df_merge, right=df, **kwargs) #, suffixes=["", "_"+str(i+1)])

    return df_merge

def bootstrap_ci_simple(group, colname="A", pct=95):
    """
    Calculate basic summary statistics for a group without bootstrapping.
    
    Parameters
    ----------
    group : pd.DataFrame
        DataFrame containing the data to analyze.
    colname : str, optional
        Name of the column to calculate statistics for. Default is "A".
    pct : int, optional
        Percentile for confidence interval (not used in this function, 
        included for API consistency). Default is 95.
    
    Returns
    -------
    pd.Series
        Series containing:
        - mean: arithmetic mean
        - median: median value
        - std: standard deviation
        - sem: standard error of the mean
        - n: sample size
    """

    mean = np.mean(group[colname].values)
    median = np.median(group[colname].values)
    std = np.std(group[colname].values)
    sem = std / np.sqrt(len(group[colname].values))
    return pd.Series({"mean": mean, "median": median, "std": std, "sem": sem, "n": len(group)})

def bootstrap_ci(group, colname="A", pct=95, n_iterations=1000):
    """
    Calculate summary statistics with bootstrap confidence intervals.
    
    Uses bootstrap resampling to estimate confidence intervals around the mean
    by repeatedly sampling with replacement from the data.
    
    Parameters
    ----------
    group : pd.DataFrame
        DataFrame containing the data to analyze.
    colname : str, optional
        Name of the column to calculate statistics for. Default is "A".
    pct : int, optional
        Confidence level for the interval (e.g., 95 for 95% CI). Default is 95.
    n_iterations : int, optional
        Number of bootstrap iterations to perform. Default is 1000.
    
    Returns
    -------
    pd.Series
        Series containing:
        - mean: arithmetic mean of the original data
        - median: median value of the original data
        - std: standard deviation of the original data
        - sem: standard error of the mean
        - ci_lo: lower bound of the confidence interval
        - ci_hi: upper bound of the confidence interval
        - ci_lo_delta: distance from mean to lower bound
        - ci_hi_delta: distance from upper bound to mean
        - n: sample size
    """

    bootstrap_means = []
    for _ in range(n_iterations):
        # Resample with replacement and calculate the mean
        bootstrap_sample = group.sample(n=len(group), replace=True)
        bootstrap_means.append(bootstrap_sample[colname].mean())

    # Calculate the 95% confidence interval from the bootstrap distribution
    lower_bound = np.percentile(bootstrap_means, (100-pct)/2)
    upper_bound = np.percentile(bootstrap_means, (100+pct)/2)
    mean = np.mean(group[colname].values)
    median = np.median(group[colname].values)
    std = np.std(group[colname].values)
    sem = std / np.sqrt(len(group[colname].values))
    return pd.Series({"mean": mean, "median": median, "std": std, "sem": sem, "ci_lo": lower_bound, "ci_hi": upper_bound, "ci_lo_delta": mean-lower_bound, "ci_hi_delta": upper_bound-mean, "n": len(group)})

def summary_stats(df, colnames_groupby, colname_calc_stats, pct=95, n_iterations=1000):
    """
    Calculate summary statistics with bootstrap confidence intervals for grouped data.
    
    Groups the DataFrame by specified columns and calculates bootstrap statistics
    for each group.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to analyze.
    colnames_groupby : str, list, or np.ndarray
        Column name(s) to group by. Can be a single string or a list of column names.
    colname_calc_stats : str
        Name of the column to calculate statistics for within each group.
    pct : int, optional
        Confidence level for the interval (e.g., 95 for 95% CI). Default is 95.
    n_iterations : int, optional
        Number of bootstrap iterations to perform. Default is 1000.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per group, containing bootstrap statistics
        (mean, median, std, sem, confidence intervals, and sample size).
    """
    
    if not (isinstance(colnames_groupby, list) or isinstance(colnames_groupby, np.ndarray)):
        # if grouping by only a single column, put that colname in a list for consistency
        colnames_groupby_list = [colnames_groupby,]

    return df.groupby(colnames_groupby, group_keys=False).apply(
        lambda a: bootstrap_ci(a.drop(columns=colnames_groupby_list), colname=colname_calc_stats, pct=pct, n_iterations=n_iterations)
    )