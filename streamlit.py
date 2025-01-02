import streamlit as st
import plotly.express as px
import functions


def setup_sidebar(data):
    """Setup sidebar options"""
    st.sidebar.title("Options")
    
    # Column selection for grouping
    choose_column = st.sidebar.selectbox("Select the column to aggregate by:", data.columns)
    
    # Aggregation function selection (common across all visualizations)
    agg_function = st.sidebar.selectbox("Select the aggregation function:", ['sum', 'mean', 'median'])

    # Columns to exclude
    available_columns = [col for col in data.columns if col != choose_column]
    columns_to_exclude = st.sidebar.multiselect(
        "Select columns to exclude:",
        available_columns,
        help="These columns will be removed before processing the data"
    )

    
    sparse_threshold = st.sidebar.slider("Select Threshold", 0, 4000, value=750)

    
    return {
        'group_column': choose_column,
        'agg_function': agg_function,
        'columns_to_exclude': columns_to_exclude,
        'threshold': sparse_threshold
    }


def render_pca_analysis(data_reduce, num_components, meta_columns):
    """Render PCA visualization section"""
    st.write("### PCA Visualization")
    
    if num_components == 2:
        fig = px.scatter(
            data_reduce, 
            x="PC1", y="PC2", 
            title="2D PCA Visualization",
            opacity=0.7,
            hover_data=meta_columns
        )
        st.plotly_chart(fig)
    elif num_components == 3:
        fig = px.scatter_3d(
            data_reduce, 
            x="PC1", y="PC2", z="PC3", 
            title="3D PCA Visualization",
            opacity=0.7,
            hover_data=meta_columns
        )
        st.plotly_chart(fig)

    st.write("### Reduced Dimensional Data")
    st.dataframe(data_reduce)


def render_party_distribution(data, party_cols):
    """Render party distribution visualization"""
    st.write("### Party Vote Distribution")
    party_totals = data[party_cols].sum().sort_values(ascending=True)
    fig = px.bar(
        party_totals,
        orientation='h',
        title='Total Votes by Party',
        labels={'value': 'Total Votes', 'index': 'Party'}
    )
    fig.update_layout(height=800)
    st.plotly_chart(fig)


def render_strongest_correlations(party_cols, party_corr):
    """Render strongest correlations analysis"""
    correlations = []
    for i in range(len(party_cols)):
        for j in range(i+1, len(party_cols)):
            correlations.append({
                'Party 1': party_cols[i].replace('party_', ''),
                'Party 2': party_cols[j].replace('party_', ''),
                'Correlation': party_corr.iloc[i,j]
            })


def render_party_correlation(data, party_cols):
    """Render party correlation analysis"""
    st.write("### Party Vote Correlation")
    party_corr = data[party_cols].corr()
    
    # Correlation heatmap
    fig = px.imshow(
        party_corr,
        title='Party Vote Correlation Heatmap',
        labels=dict(color="Correlation"),
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig)
    
    render_strongest_correlations(party_cols, party_corr)


def render_city_metrics(city_data, party_cols):
    """Render city metrics"""
    total_city_votes = city_data[party_cols].sum().sum()
    num_polling_stations = len(city_data['ballot_code'].unique())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Votes", f"{total_city_votes:,.0f}")
    with col2:
        st.metric("Polling Stations", num_polling_stations)
    with col3:
        avg_votes_per_station = total_city_votes / num_polling_stations
        st.metric("Avg Votes per Station", f"{avg_votes_per_station:,.1f}")


def render_city_vote_distribution(city_data, party_cols, selected_city):
    """Render city vote distribution"""
    st.write("### Party Vote Distribution in", selected_city)
    city_party_votes = city_data[party_cols].sum()
    city_party_votes = city_party_votes[city_party_votes > 0]  # Remove zero votes
    
    # Calculate percentages
    city_party_pct = (city_party_votes / city_party_votes.sum() * 100).round(2)
    
    col1, col2 = st.columns(2)
    with col1:
        # Bar chart showing all parties
        fig1 = px.bar(
            city_party_votes.sort_values(ascending=True),
            orientation='h',
            title='Total Votes by Party',
            labels={'value': 'Votes', 'index': 'Party'}
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Create custom text array for percentages
        custom_text = []
        for pct in city_party_pct:
            if pct >= 1.5:
                custom_text.append(f'{pct:.1f}%')
            else:
                custom_text.append('')
        
        # Pie chart with custom text display
        fig2 = px.pie(
            values=city_party_pct,
            names=city_party_pct.index,
            title='Vote Share (%)',
            hole=0.3
        )
        
        # Update trace with custom text
        fig2.update_traces(
            textposition='inside',
            textinfo='text',
            text=custom_text
        )
        
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    

def render_polling_station_analysis(city_data, party_cols, selected_city):
    """Render polling station analysis"""
    st.write("### Polling Station Details")
    station_data = city_data.groupby('ballot_code')[party_cols].sum()
    station_total_votes = station_data.sum(axis=1)
    
    fig3 = px.bar(
        station_total_votes,
        title=f'Votes by Polling Station in {selected_city}',
        labels={'value': 'Total Votes', 'index': 'Polling Station'}
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3)
    
    station_data['Total Votes'] = station_data.sum(axis=1)


def render_city_analysis(data, party_cols):
    """Render city-specific analysis"""
    st.write("### City Analysis")
    city_votes = data.groupby('city_name')[party_cols].sum().sum(axis=1)
    top_cities = city_votes.nlargest(20)
        

      # First visualization: Top 20 cities
    fig = px.bar(
        top_cities,
        title='Top 20 Cities by Total Votes',
        labels={'value': 'Total Votes', 'index': 'City'}
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    st.plotly_chart(fig)
    
    # City selection and data preparation
    st.write("### Analysis per City")

    all_cities = sorted(data['city_name'].unique())
    selected_city = st.selectbox("Select a city:", all_cities)
    city_data = data[data['city_name'] == selected_city]

    render_city_metrics(city_data, party_cols)
    render_city_vote_distribution(city_data, party_cols, selected_city)
    render_polling_station_analysis(city_data, party_cols, selected_city)

      
def process_data(data, options):
    """Process data based on sidebar options"""
    # Group and aggregate data
    data_agg = functions.group_and_aggregate_data(data, options['group_column'], options['agg_function'])
    
    # Remove sparse columns
    sparse_agg_data = functions.remove_sparse_columns(data_agg, options['threshold'])
    
    # Reset index for meta columns
    sparse_agg_data = sparse_agg_data.reset_index()
    
    # Perform dimensionality reduction
    reduced_data = functions.dimensionality_reduction(
        sparse_agg_data, 
        options['num_components'], 
        options['meta_columns']
    )
    
    return {
        'aggregated': data_agg,
        'sparse': sparse_agg_data,
        'reduced': reduced_data
    }


def setup_pca_controls(data, viz_type, group_column):
    """Setup PCA-specific controls"""
    if viz_type != "PCA Analysis":
        return {
            'meta_columns': [],
            'num_components': 2
        }
    
    st.write("### PCA Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        # Meta columns selection
        available_columns = [col for col in data.columns if col != group_column]
        meta_columns = st.multiselect(
            "Select metadata columns to preserve:",
            available_columns,
            help="These columns will be preserved in the final output without dimensionality reduction"
        )
            
    with col2:
        # Number of components
        num_components = st.slider("Number of Components", 2, 10, value=2)
    
    return {
        'meta_columns': meta_columns,
        'num_components': num_components,
    }


# Then, modify the main function to handle column exclusion:
def main():
    st.title("Data Analysis and Visualization")
    st.subheader("Interactive Dataset Analysis")

    # File upload and initial data loading
    uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx'])
    if uploaded_file is None:
        return
    data = functions.load_data(uploaded_file.name)
    party_cols = [col for col in data.columns if col.startswith('party_')]

    # Display original data
    st.write("### Original Data:")
    st.write(f"Table shape: {data.shape[0]} rows × {data.shape[1]} columns")
    st.dataframe(data)

    # Visualization selection
    viz_type = st.sidebar.selectbox(
        "Choose visualization type",
        ["PCA Analysis", "Party Distribution", 
         "Party Correlation", "City Analysis"]
    )

    # Basic sidebar setup (common controls)
    sidebar_options = setup_sidebar(data)
    
    # Process and display aggregated data
    # First, exclude selected columns (NEW)
    processed_data = data.drop(columns=sidebar_options['columns_to_exclude'])
    
    data_agg = functions.group_and_aggregate_data(
        processed_data, 
        sidebar_options['group_column'], 
        sidebar_options['agg_function']
    )

    sparse_agg_data = functions.remove_sparse_columns(data_agg, sidebar_options['threshold'])
    
    # Display aggregated data
    st.write(f"### Aggregated Data by {sidebar_options['group_column']}")
    st.write(f"Excluded columns: {', '.join(sidebar_options['columns_to_exclude']) if sidebar_options['columns_to_exclude'] else 'None'}")
    st.write(f"Table shape: {sparse_agg_data.shape[0]} rows × {sparse_agg_data.shape[1]} columns")
    st.dataframe(sparse_agg_data)

    # PCA-specific controls (only shown for PCA visualization)
    pca_options = setup_pca_controls(data, viz_type, sidebar_options['group_column'])
    
    # Combine all options
    options = {**sidebar_options, **pca_options}
    
    # Process data based on selected visualization
    if viz_type == "PCA Analysis":
        processed_data = process_data(data, options)
        render_pca_analysis(processed_data['reduced'], 
                          options['num_components'], 
                          options['meta_columns'])
    else:
        # Other visualizations
        visualization_functions = {
            "Party Distribution": lambda: render_party_distribution(data, party_cols),
            "Party Correlation": lambda: render_party_correlation(data, party_cols),
            "City Analysis": lambda: render_city_analysis(data, party_cols)
        }
        visualization_functions[viz_type]()

if __name__ == "__main__":
    main()
