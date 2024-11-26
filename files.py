import numpy as np
import pandas as pd
import matplotlib
import streamlit


def load_data(filepath: str) -> pd.DataFrame:
    """
    We assume the file will be an  Excel/csv type.
    To read Excel file make sure
    :param filepath: a path to an Excel/csv file
    :return: dataframe of the file 'openpyxl' has been installed in the project.
    """
    path_split = filepath.lower().split('.')
    if path_split[-1] == 'csv':
        return pd.read_csv(filepath)
    else:
        return pd.read_excel(filepath)

# check:
check = load_data("knesset_25.xlsx")
print(type(check))
print(check['city_name'])

def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    pass