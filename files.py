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
    pass