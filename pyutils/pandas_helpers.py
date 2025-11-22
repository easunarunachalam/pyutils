from copy import deepcopy
import numpy as np
import pandas as pd

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

def summary_stats(group, colname="A"):

    mean = np.mean(group[colname].values)
    median = np.median(group[colname].values)
    std = np.std(group[colname].values)
    sem = std / np.sqrt(len(group[colname].values))
    return pd.Series({"mean": mean, "median": median, "std": std, "sem": sem, "n": len(group)})

# def bootstrap_ci_simple(arr, func=np.mean, ci=95, n_bstrap_samples=1000):
#     func_values = [func(np.random.choice(arr, size=len(arr))) for _ in range(n_bstrap_samples)]
#     lo, hi = (100-ci)/2, 100 - (100-ci)/2
#     return np.percentile(func_values, lo), np.percentile(func_values, hi)

def bootstrap_ci_simple(group, colname="A", pct=95):

    mean = np.mean(group[colname].values)
    median = np.median(group[colname].values)
    std = np.std(group[colname].values)
    sem = std / np.sqrt(len(group[colname].values))
    return pd.Series({"mean": mean, "median": median, "std": std, "sem": sem, "n": len(group)})

def bootstrap_ci(group, colname="A", pct=95, n_iterations=1000):

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