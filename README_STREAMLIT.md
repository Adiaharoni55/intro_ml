# Interactive Data Analysis Dashboard

This Streamlit application provides an interactive interface for analyzing and visualizing electoral data, built on top of our data processing library.

## Features

- **Data Upload & Processing**
  - Support for CSV and Excel files
  - Dynamic column selection and aggregation
  - Sparse column removal with adjustable threshold

- **Multiple Visualization Types**
  - PCA Analysis (2D/3D visualization)
  - Party Vote Distribution
  - Party Correlation Analysis
  - Detailed City-Level Analysis

- **Interactive Controls**
  - Customizable aggregation functions
  - Adjustable visualization parameters
  - Column exclusion options
  - City-specific detailed analysis

## Installation

1. Install required packages:
```bash
pip install streamlit pandas numpy plotly
```

2. Install the data processing library (from the main project)

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit.py
```

2. Upload your data file (CSV or Excel)

3. Use the sidebar to:
   - Select visualization type
   - Choose aggregation options
   - Set processing parameters

## Visualization Types

### PCA Analysis
- Interactive 2D/3D scatter plots
- Customizable number of components
- Metadata column preservation
- Dimensionality reduction visualization

### Party Distribution
- Total votes by party
- Horizontal bar charts
- Vote distribution analysis

### Party Correlation
- Correlation heatmap
- Inter-party voting pattern analysis
- Strongest correlations identification

### City Analysis
- Top cities by total votes
- City-specific metrics
- Vote distribution per city
- Polling station analysis

## Interface Components

### Sidebar Options
- Column selection for aggregation
- Aggregation function selection
- Column exclusion
- Threshold adjustment

### Main Interface
- Original data display
- Processed data preview
- Interactive visualizations
- City-specific analysis tools

## Data Requirements
The application expects electoral data with:
- Party vote columns (prefixed with 'party_')
- City and ballot code information
- Numeric vote counts

## Notes
- The application automatically detects party-related columns
- All visualizations are interactive (zoom, hover, selection)
- Processed data can be viewed at each step
