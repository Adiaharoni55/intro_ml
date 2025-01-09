import streamlit as st
import plotly.express as px
import functions



def process_data(data, options):
    """Process data based on analysis type"""
    party_cols = [col for col in data.columns if col.startswith('party_')]
    
    if options['analysis_type'] == 'city_wise':
        # Original city-wise processing
        processed_data = data.drop(columns=options['columns_to_exclude'])
        data_agg = functions.group_and_aggregate_data(
            processed_data, 
            options['group_column'], 
            options['agg_function']
        )
        return functions.remove_sparse_columns(data_agg, options['threshold'])
    else:
        # Party-wise processing - always aggregate by city_name
        city_data = data.groupby('city_name')[party_cols].sum()  # Always use sum for party-wise
        # Remove excluded columns (cities in this case) before transpose
        if options['columns_to_exclude']:
            city_data = city_data.drop(index=options['columns_to_exclude'])
        data_t = city_data.T  # Transpose the data
        return functions.remove_sparse_columns(data_t, options['threshold'])


def setup_sidebar(data):
    """Setup sidebar options"""
    st.sidebar.title("Options")
    
    # Analysis type selection at the top (hidden) for initialization
    is_party_wise = st.sidebar.checkbox(
        "Switch to party-wise analysis",
        help="When enabled, data will be aggregated by cities and transposed to show party correlations. This provides a different perspective on party relationships across cities.",
        key="party_wise_checkbox"
    )
    
    choose_column = 'city_name'  # Default for party_wise
    agg_function = 'sum'         # Default for party_wise
    columns_to_exclude = []      # Will be populated based on mode
    
    if not is_party_wise:
        # City-wise mode options
        choose_column = st.sidebar.selectbox("Select the column to aggregate by:", data.columns)
        agg_function = st.sidebar.selectbox(
            "Select the aggregation function:",
            ['sum', 'mean', 'median', 'count', 'std', 'min', 'max'],
            help="""Choose how to aggregate the data:
            - sum: Total of all values
            - mean: Average value
            - median: Middle value
            - count: Number of occurrences
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value"""
        )
        available_columns = [col for col in data.columns if col != choose_column]
    else:
        # Party-wise mode - get available columns after transposing
        party_cols = [col for col in data.columns if col.startswith('party_')]
        city_data = data.groupby('city_name')[party_cols].sum()
        available_columns = city_data.index.tolist()  # Cities become columns after transpose
    
    # Show columns to exclude based on current mode's available columns
    columns_to_exclude = st.sidebar.multiselect(
        "Select columns to exclude:",
        available_columns,
        help="These columns will be removed before processing the data"
    )
    
    sparse_threshold = st.sidebar.slider("Select Threshold", 0, 4000, value=750)
    
    # Set analysis type based on checkbox
    analysis_type = 'party_wise' if is_party_wise else 'city_wise'
    
    return {
        'analysis_type': analysis_type,
        'group_column': choose_column,
        'agg_function': agg_function,
        'columns_to_exclude': columns_to_exclude,
        'threshold': sparse_threshold
    }


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


def render_pca_analysis(data_reduce, num_components, meta_columns, with_color):
    """Render PCA visualization section"""
    st.write("### PCA Visualization")
    
    if num_components == 2:
        # Create base scatter plot arguments
        scatter_args = {
            'data_frame': data_reduce,
            'x': "PC1",
            'y': "PC2",
            'title': "2D PCA Visualization",
            'opacity': 0.7,
            'hover_data': meta_columns
        }
        
        # Add color parameter only if with_color is True
        if with_color:
            scatter_args['color'] = data_reduce.index
            
        fig = px.scatter(**scatter_args)
        
        # Customize legend title only if colors are shown
        if with_color:
            fig.update_layout(
                showlegend=True,
                legend_title="City Name"
            )
        st.plotly_chart(fig)
        
    elif num_components == 3:
        # Create base scatter plot arguments
        scatter_args = {
            'data_frame': data_reduce,
            'x': "PC1",
            'y': "PC2",
            'z': "PC3",
            'title': "3D PCA Visualization",
            'opacity': 0.7,
            'hover_data': meta_columns
        }
        
        # Add color parameter only if with_color is True
        if with_color:
            scatter_args['color'] = data_reduce.index
            
        fig = px.scatter_3d(**scatter_args)
        
        # Customize legend title only if colors are shown
        if with_color:
            fig.update_layout(
                showlegend=True,
                legend_title="City Name"
            )
        st.plotly_chart(fig)

    st.write("### Reduced Dimensional Data")
    st.dataframe(data_reduce)


    
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
    
    # Process data based on analysis type
    processed_data = process_data(data, sidebar_options)
    
    # Display aggregated data
    st.write(f"### {'City-wise' if sidebar_options['analysis_type'] == 'city_wise' else 'Party-wise'} Aggregated Data")
    if sidebar_options['columns_to_exclude']:
        st.write(f"Excluded columns: {', '.join(sidebar_options['columns_to_exclude'])}")
    st.write(f"Table shape: {processed_data.shape[0]} rows × {processed_data.shape[1]} columns")
    st.dataframe(processed_data)

    # PCA-specific controls
    if viz_type == "PCA Analysis":
        st.write("### PCA Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Meta columns selection for both modes
            available_columns = [col for col in processed_data.columns]
            meta_columns = st.multiselect(
                "Select metadata columns to preserve:",
                available_columns,
                help="These columns will be preserved in the final output without dimensionality reduction"
            )
                
        with col2:
            # Number of components
            num_components = st.slider("Number of Components", 2, 10, value=2)
        
        with col3:
            with_color = st.checkbox("Color the data by index column")
        
        # Combine options
        pca_options = {
            'meta_columns': meta_columns,
            'num_components': num_components,
            'with_color': with_color
        }
        
        # Combine all options
        options = {**sidebar_options, **pca_options}
        
        # Perform dimensionality reduction
        reduced_data = functions.dimensionality_reduction(
            processed_data, 
            options['num_components'], 
            options['meta_columns']
        )
        
        render_pca_analysis(reduced_data, 
                          options['num_components'], 
                          options['meta_columns'],
                          options['with_color']
                          )
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