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
    # Separate features (non-meta columns) and meta columns
    non_meta_columns = [col for col in df.columns if col not in meta_columns]
    
    # Get the data to transform (non-meta columns)
    to_transform = df[non_meta_columns].copy()
    to_transform = to_transform.fillna(0)
    
    # Standardize the data (both center and scale)
    m = to_transform.mean(axis=0)
    s = to_transform.std(axis=0)
    to_transform = (to_transform - m) / s
    
    # Calculate covariance matrix
    cov_matrix = np.cov(to_transform.T)
    
    # Get eigenvalues and eigenvectors
    eigvalues, eigvectors = np.linalg.eigh(cov_matrix)
    
    # Select top k components
    top_k_indices = (-eigvalues).argsort()[:num_components]
    top_k_eigenvectors = -eigvectors[:, top_k_indices]
    top_k_eigenvectors[:, 1] = -top_k_eigenvectors[:, -1]  # Flip second component

    # Transform the data using scaled eigenvectors
    reduced_data = np.dot(to_transform, top_k_eigenvectors)
    
    # Convert reduced data to DataFrame
    reduced_df = pd.DataFrame(
        reduced_data,
        index=df.index,
        columns=[f'PC{i+1}' for i in range(num_components)]
    )
    
    if meta_columns:
        result = pd.concat([reduced_df, df[meta_columns]], axis=1)
    else:
        result = reduced_df
        
    return result