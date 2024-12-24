import numpy as np
import pandas as pd
import matplotlib as plt
import streamlit as st
import plotly.express as px

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


# def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
#     """
#     :param df: A pandas DataFrame containing the data to be reduced.
#     :param num_components: The number of principal components to retain.
#     :param meta_columns: A list of metadata columns to exclude from dimensionality reduction (these should be included in the final output without changes).
#     :return: A pandas DataFrame with the reduced dimensions and the metadata columns.
#     """
#     # Separate features (non-meta columns) and meta columns
#     non_meta_columns = [col for col in df.columns if col not in meta_columns]
#
#     # Get the data to transform (non-meta columns)
#     to_transform = df[non_meta_columns].copy()
#     to_transform = to_transform.fillna(0)
#
#     # Standardize the data (both center and scale)
#     m = to_transform.mean(axis=0)
#     s = to_transform.std(axis=0)
#     to_transform = (to_transform - m) / s
#
#     a_at = to_transform @ to_transform.T


def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:
    # Separate features (non-meta columns) and meta columns
    non_meta_columns = [col for col in df.columns if col not in meta_columns]

    # Get the data to transform (non-meta columns)
    to_transform = df[non_meta_columns].select_dtypes(include=['number']).copy()
    to_transform = to_transform.fillna(0)

    # Standardize the data (both center and scale)
    m = to_transform.mean(axis=0)
    s = to_transform.std(axis=0)
    to_transform = (to_transform - m) / s
    to_transform = to_transform.to_numpy()

    # Choose which matrix to decompose based on dimensions
    n, p = to_transform.shape

    if n < p:
        # Compute eigendecomposition of A*A^T (smaller matrix)
        a_at = np.dot(to_transform, to_transform.T)
        eigvalues, U = np.linalg.eigh(a_at)

        # Sort in descending order
        idx = np.argsort(eigvalues)[::-1]
        eigvalues = eigvalues[idx]
        # Keep only top num_components
        U = U[:, idx]

        # Fix signs to match sklearn's convention**********
        # Make largest element in each column positive
        for i in range(U.shape[1]):
            if np.max(np.abs(U[:, i])) != np.max(U[:, i]):
                U[:, i] *= -1

        # Keep only the components we want
        U_reduced = U[:, :num_components]

        # Project the data directly
        reduced_data = U_reduced  # The projection is already in U


    else:
        # Compute eigendecomposition of A^T*A (smaller matrix)
        at_a = np.dot(to_transform.T, to_transform)
        eigvalues, V = np.linalg.eigh(at_a)

        # Sort in descending order
        idx = np.argsort(eigvalues)[::-1]
        eigvalues = eigvalues[idx]
        V = V[:, idx]

        # Fix signs to match sklearn's convention***************
        for i in range(V.shape[1]):
            if np.max(np.abs(V[:, i])) != np.max(V[:, i]):
                V[:, i] *= -1

        # Keep only the components we want
        V_reduced = V[:, :num_components]

        # Project the data
        reduced_data = to_transform @ V_reduced

    reduced_df = pd.DataFrame(
        reduced_data,
        index=df.index,
        columns=[f'PC{i + 1}' for i in range(num_components)]
    )
    if meta_columns:
        result = pd.concat([reduced_df, df[meta_columns]], axis=1)
    else:
        result = reduced_df

    return result



# num_of_remove_sparse = st.slider('Select a column', min_value=1, max_value=10, value=1)
# st.write(num_of_remove_sparse, "the number of sparse columns", num_of_remove_sparse)

# Streamlit app
def main():
    st.title("Dimensionality Reduction with PCA")

    # File upload
    uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx'])
    if uploaded_file is None:
        return
    data = load_data(uploaded_file.name)

    # show the original data - head 5
    st.write(f"### Original Data:")
    st.dataframe(data.head())

    # Sidebar
    st.sidebar.title("Options")
    num_components = st.sidebar.slider("Number of Components", 2, 10, value=2)

    choose_column = st.sidebar.selectbox("Select the column representing cities:", data.columns)
    agg_function = st.sidebar.selectbox("Select the aggregation function:", ['sum', 'mean', 'median'])

    sparse_threshold = st.sidebar.slider("Select Threshold", 100, 2000, value=750)

    data_agg = group_and_aggregate_data(data, choose_column, agg_function)

    sparse_agg_data = remove_sparse_columns(data_agg, sparse_threshold)

    # st.write("Loaded Data:", data_agg.head())
    st.write(f"### Aggregated Data by {choose_column}")
    st.dataframe(sparse_agg_data.head())

    data_reduce = dimensionality_reduction(sparse_agg_data, num_components, ['ballot_code'])
    # PCA
    numeric_data = data_reduce.select_dtypes(include=['number'])
    if not numeric_data.empty:
        # reduced_data = dimensionality_reduction(numeric_data, num_components)
        st.write("### Reduced Dimensional Data")
        st.dataframe(data_reduce)

        # Visualization
        st.write("### PCA Visualization")
        st.bar_chart(data_reduce)
    else:
        st.warning("No numeric columns found for PCA.")

    st.write("### PCA Visualization")
    if num_components == 2:
        # 2D Visualization using Plotly
        fig = px.scatter(data_reduce, x="PC1", y="PC2", title="2D PCA Visualization", opacity=0.7)
        st.plotly_chart(fig)
    elif num_components == 3:
        # 3D Visualization
        fig = px.scatter_3d(data_reduce, x="PC1", y="PC2", z="PC3", title="3D PCA Visualization", opacity=0.7)
        st.plotly_chart(fig)


# Run the app
if __name__ == "__main__":
    main()

