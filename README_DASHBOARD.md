# Data Processing Functions Library

This library provides a collection of utility functions for data loading, processing, and dimensionality reduction using pandas and numpy. It's designed to handle various data preprocessing tasks commonly encountered in data analysis workflows.

## Installation

### Prerequisites
- Python 3.x
- numpy
- pandas
- openpyxl (for Excel file support)

Install the required packages using pip:
```bash
pip install numpy pandas openpyxl
```

## Functions Overview

### `load_data(filepath: str) -> pd.DataFrame`
Loads data from CSV or Excel files into a pandas DataFrame.

**Parameters:**
- `filepath`: Path to the input file (CSV or Excel format)

**Returns:**
- A pandas DataFrame containing the loaded data

**Example:**
```python
df = load_data('path/to/your/data.csv')
```

### `group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame`
Groups and aggregates data while handling both numeric and non-numeric columns appropriately.

**Parameters:**
- `df`: Input DataFrame
- `group_by_column`: Column name to group by
- `agg_func`: Aggregation function to apply to numeric columns

**Returns:**
- A grouped and aggregated DataFrame

**Example:**
```python
grouped_df = group_and_aggregate_data(df, 'category', 'sum')
```

### `remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame`
Removes numeric columns whose sum is below a specified threshold.

**Parameters:**
- `df`: Input DataFrame
- `threshold`: Minimum total sum for a column to be retained

**Returns:**
- DataFrame with sparse columns removed

**Example:**
```python
filtered_df = remove_sparse_columns(df, threshold=100)
```

### `dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame`
Performs PCA-based dimensionality reduction while preserving specified metadata columns.

**Parameters:**
- `df`: Input DataFrame
- `num_components`: Number of principal components to retain
- `meta_columns`: List of column names to preserve unchanged

**Returns:**
- DataFrame with reduced dimensions plus preserved metadata columns

**Example:**
```python
reduced_df = dimensionality_reduction(df, num_components=2, meta_columns=['category', 'date'])
```

## Implementation Details

### Dimensionality Reduction
- Uses eigendecomposition for PCA implementation
- Automatically handles cases where n < p or n ≥ p (n: samples, p: features)
- Standardizes data before reduction
- Preserves specified metadata columns
- Names principal components as PC1, PC2, etc.

### Data Processing
- Handles both numeric and non-numeric columns appropriately in grouping operations
- Provides flexible aggregation function specification
- Preserves data types when possible
- Handles missing values in dimensionality reduction by filling with zeros

## Demonstration

The repository includes a Jupyter notebook (`demonstration.ipynb`) that showcases the usage of these functions with real-world examples. The notebook demonstrates:
- Loading and preprocessing data
- Grouping and aggregating data
- Removing sparse columns
- Performing dimensionality reduction
- Visualizing results

We recommend reviewing the demonstration notebook to better understand how to effectively use these functions in your data analysis workflow.

## Notes
- The dimensionality reduction implementation uses a memory-efficient approach by choosing the smaller covariance matrix based on data dimensions
- Non-numeric columns are handled using 'first' aggregation in grouping operations
- All numeric operations are performed using numpy for efficiency
