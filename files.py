import numpy as np
import pandas as pd
import matplotlib
import streamlit

def load_data(filepath: str) -> pd.DataFrame:
    """
    We assume the file will be an  Excel/csv type.
    To read Excel file make sure 'openpyxl' has been installed in the project.
    :param filepath: a path to an Excel/csv file
    :return: dataframe of the file
    """
    path_split = filepath.lower().split('.')
    if path_split[-1] == 'csv':
        return pd.read_csv(filepath)
    else:
        return pd.read_excel(filepath)


def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    """
    :param df: A pandas DataFrame containing the dataset.
    :param group_by_column: The column to group data by.
    :param agg_func: The aggregation function to apply to each group.
    :return: A pandas DataFrame with aggregated results.
    """

    return df.groupby(group_by_column).agg(agg_func)


def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Passing through only the numeric columns (city name will stay)
    :param df: A pandas DataFrame.
    :param threshold: The minimum total sum for a column to be retained in the DataFrame.
    :return: A pandas DataFrame with sparse columns removed.
    """
    res = df
    for column_name in df.select_dtypes(include=['number']).columns:
        if res[column_name].sum() < threshold:
            res = res.drop(columns=column_name)
    return res


def PCA_calculation(df: pd.DataFrame):
    """
    calculate pca of all the columns
    :param df: A pandas DataFrame.
    :return: column name to remove
    """
    pass


def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    """
    :param df: A pandas DataFrame containing the data to be reduced.
    :param num_components: The number of principal components to retain.
    :param meta_columns: A list of metadata columns to exclude from dimensionality reduction (these should be included in the final output without changes).
    :return: A pandas DataFrame with the reduced dimensions and the metadata columns.
    """
    res = df
    for column_name in meta_columns:
        df = df.drop(columns=column_name)
    while len(res.columns) > num_components:
        min_sum_column = df.sum().idxmin() # PCA calculation instead of min
        df = df.drop(columns=min_sum_column)
        res = res.drop(columns=min_sum_column)
    return res


# df_data = load_data("knesset_25.xlsx").head(5)
# print(dimensionality_reduction(df_data, 10, ['city_name', 'ballot_code']))