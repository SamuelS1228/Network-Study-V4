import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import base64
import math
import random
from copy import deepcopy

# Set page configuration
st.set_page_config(
    page_title="Warehouse Optimizer",
    page_icon="🏭",
    layout="wide"
)

# Title and description
st.title("Warehouse Location Optimizer")
st.markdown("""
This application helps determine optimal warehouse locations based on store locations and their sales volumes.
Upload your store data, select the number of warehouses, and see the optimized locations.
""")

# Function to calculate distance between two points (haversine formula)
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Function to generate sample data
def generate_sample_data(num_stores=100):
    np.random.seed(42)
    
    # Define state boundaries (approximate, for random generation)
    regions = {
        'Northeast': {'lat_range': (40.0, 47.0), 'lon_range': (-80.0, -67.0), 'store_pct': 0.2},
        'Southeast': {'lat_range': (25.0, 39.0), 'lon_range': (-92.0, -75.0), 'store_pct': 0.25},
        'Midwest': {'lat_range': (36.0, 49.0), 'lon_range': (-104.0, -80.0), 'store_pct': 0.25},
        'West': {'lat_range': (32.0, 49.0), 'lon_range': (-124.0, -104.0), 'store_pct': 0.3}
    }
    
    data = []
    for region, bounds in regions.items():
        n_stores = int(num_stores * bounds['store_pct'])
        for _ in range(n_stores):
            lat = np.random.uniform(bounds['lat_range'][0], bounds['lat_range'][1])
            lon = np.random.uniform(bounds['lon_range'][0], bounds['lon_range'][1])
            sales = int(np.random.uniform(10000, 1000000))
            data.append({'Latitude': lat, 'Longitude': lon, 'Sales': sales, 'Region': region})
    
    # Make sure we have exactly num_stores
    if len(data) < num_stores:
        # Add more to the West if needed to reach the target
        for _ in range(num_stores - len(data)):
            region = 'West'
            bounds = regions[region]
            lat = np.random.uniform(bounds['lat_range'][0], bounds['lat_range'][1])
            lon = np.random.uniform(bounds['lon_range'][0], bounds['lon_range'][1])
            sales = int(np.random.uniform(10000, 1000000))
            data.append({'Latitude': lat, 'Longitude': lon, 'Sales': sales, 'Region': region})
    
    return pd.DataFrame(data)

# Custom K-means function with constraints
def constrained_kmeans(X, n_clusters, max_iter=100, sales_weights=None, 
                     min_stores_per_warehouse=0, min_sales_per_warehouse=0,
                     store_sales=None, seed=None):
    """
    Implementation of K-means clustering with constraints on minimum stores and sales
    
    Args:
        X: Array of shape (n_samples, n_features) - in this case, latitude and longitude
        n_clusters: Number of clusters to form
        max_iter: Maximum number of iterations
        sales_weights: Optional weights for each point (based on sales)
        min_stores_per_warehouse: Minimum number of stores per warehouse
        min_sales_per_warehouse: Minimum sales value per warehouse
        store_sales: Array of sales values for each store
        seed: Random seed for reproducibility
        
    Returns:
        centroids: Array of final centroids
        labels: Cluster labels for each point
        objective_value: A measure of solution quality (lower is better)
    """
    n_samples, n_features = X.shape
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # If the constraints are impossible to satisfy, warn and reset
    if min_stores_per_warehouse * n_clusters > n_samples:
        if seed is None:  # Only show warning once
            st.warning(f"Warning: Cannot satisfy minimum of {min_stores_per_warehouse} stores for {n_clusters} warehouses with only {n_samples} total stores. Proceeding with best-effort optimization.")
        min_stores_per_warehouse = 0
    
    # Check if we have sales data for constraints
    has_sales_constraints = (min_sales_per_warehouse > 0) and (store_sales is not None)
    
    # Try up to 5 times to meet the constraints
    for attempt in range(5):
        # Initialize centroids randomly from the data points
        if sales_weights is not None:
            # Normalize weights
            sales_weights = sales_weights / np.sum(sales_weights)
            # Sample based on weights
            idx = np.random.choice(n_samples, size=n_clusters, replace=False, p=sales_weights)
        else:
            idx = np.random.choice(n_samples, size=n_clusters, replace=False)
            
        centroids = X[idx]
        
        # Initialize labels
        labels = np.zeros(n_samples, dtype=int)
        
        for iteration in range(max_iter):
            # Assign points to closest centroid
            distances = np.zeros((n_samples, n_clusters))
            for i in range(n_clusters):
                # Calculate Euclidean distance to each centroid
                distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))
            
            # Get the closest centroid for each point
            new_labels = np.argmin(distances, axis=1)
            
            # Apply constraints if needed
            if min_stores_per_warehouse > 0 or has_sales_constraints:
                # Check store count constraint
                for i in range(n_clusters):
                    stores_in_cluster = np.sum(new_labels == i)
                    if stores_in_cluster < min_stores_per_warehouse:
                        # Need to add more stores to this cluster
                        stores_needed = min_stores_per_warehouse - stores_in_cluster
                        
                        # Find the closest stores not already in this cluster
                        other_stores = np.where(new_labels != i)[0]
                        if len(other_stores) > 0:
                            # Sort by distance to this centroid
                            closest_stores = sorted(other_stores, key=lambda j: distances[j, i])
                            # Take the stores we need
                            stores_to_move = closest_stores[:min(stores_needed, len(closest_stores))]
                            # Assign them to this cluster
                            new_labels[stores_to_move] = i
                
                # Check sales constraint if needed
                if has_sales_constraints:
                    for i in range(n_clusters):
                        sales_in_cluster = np.sum(store_sales[new_labels == i])
                        if sales_in_cluster < min_sales_per_warehouse:
                            # Need to add more sales to this cluster
                            sales_needed = min_sales_per_warehouse - sales_in_cluster
                            
                            # Find stores not in this cluster
                            other_stores = np.where(new_labels != i)[0]
                            if len(other_stores) > 0:
                                # Sort by sales value (descending)
                                high_sales_stores = sorted(other_stores, key=lambda j: store_sales[j], reverse=True)
                                
                                # Add stores until we meet the sales requirement or run out of stores
                                for store_idx in high_sales_stores:
                                    new_labels[store_idx] = i
                                    sales_in_cluster += store_sales[store_idx]
                                    if sales_in_cluster >= min_sales_per_warehouse:
                                        break
            
            # Check for convergence
            if np.array_equal(labels, new_labels):
                break
                
            labels = new_labels
            
            # Update centroids
            for i in range(n_clusters):
                mask = labels == i
                if np.sum(mask) > 0:
                    if sales_weights is not None:
                        # Apply weights for centroid calculation
                        weighted_sum = np.sum(X[mask] * sales_weights[mask].reshape(-1, 1), axis=0)
                        weight_sum = np.sum(sales_weights[mask])
                        centroids[i] = weighted_sum / weight_sum if weight_sum > 0 else weighted_sum
                    else:
                        centroids[i] = np.mean(X[mask], axis=0)
        
        # After optimization, check if constraints are satisfied
        constraints_satisfied = True
        
        # Check store count constraint
        if min_stores_per_warehouse > 0:
            for i in range(n_clusters):
                stores_in_cluster = np.sum(labels == i)
                if stores_in_cluster < min_stores_per_warehouse:
                    constraints_satisfied = False
                    break
        
        # Check sales constraint
        if has_sales_constraints and constraints_satisfied:
            for i in range(n_clusters):
                sales_in_cluster = np.sum(store_sales[labels == i])
                if sales_in_cluster < min_sales_per_warehouse:
                    constraints_satisfied = False
                    break
        
        if constraints_satisfied:
            break
    
    # If we couldn't satisfy all constraints after multiple attempts
    if not constraints_satisfied and seed is None:  # Only show warning once
        st.warning("Could not fully satisfy all warehouse constraints. Using best available solution.")
    
    # Calculate objective value (weighted sum of distances)
    objective_value = 0
    for i in range(n_samples):
        # Get distance to assigned centroid
        cluster_id = labels[i]
        dist = np.sqrt(np.sum((X[i] - centroids[cluster_id])**2))
        
        # Apply sales weight if available
        if sales_weights is not None:
            dist *= sales_weights[i]
        
        objective_value += dist
    
    return centroids, labels, objective_value

# Function to optimize warehouse locations
def optimize_warehouse_locations(data, num_warehouses, sales_weight=0.5, 
                              min_stores=0, min_sales=0, num_runs=10):
    """
    Run the optimization multiple times and select the best solution
    
    Args:
        data: DataFrame with store data
        num_warehouses: Number of warehouses to place
        sales_weight: Weight for sales in the optimization (0-1)
        min_stores: Minimum stores per warehouse
        min_sales: Minimum sales per warehouse
        num_runs: Number of optimization runs to perform
        
    Returns:
        best_store_data: DataFrame with store data and best cluster assignments
        best_warehouse_locations: DataFrame with best warehouse locations
        best_metrics: DataFrame with metrics for the best solution
    """
    # Extract coordinates for clustering
    X = data[['Latitude', 'Longitude']].values
    
    # Prepare sales weights if needed
    if sales_weight != 0.5:
        # Normalize sales to get weights
        sales = data['Sales'].values
        weights = sales / np.max(sales)
        
        # Adjust weights based on the sales_weight parameter
        weights = weights ** (sales_weight * 2)
    else:
        weights = None
    
    # Variables to store the best solution
    best_centroids = None
    best_labels = None
    best_objective = float('inf')
    best_seed = None
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run the optimization multiple times
    for run in range(num_runs):
        # Update progress
        progress = (run + 1) / num_runs
        progress_bar.progress(progress)
        status_text.text(f"Optimization run {run + 1}/{num_runs}...")
        
        # Use a different random seed for each run
        seed = run * 100 + 42
        
        # Run clustering with constraints
        centroids, labels, objective = constrained_kmeans(
            X, 
            num_warehouses, 
            max_iter=100, 
            sales_weights=weights,
            min_stores_per_warehouse=min_stores,
            min_sales_per_warehouse=min_sales,
            store_sales=data['Sales'].values if min_sales > 0 else None,
            seed=seed
        )
        
        # Check if this solution is better than the current best
        if objective < best_objective:
            best_centroids = centroids.copy()
            best_labels = labels.copy()
            best_objective = objective
            best_seed = seed
    
    # Clear the progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Make a copy of the data to avoid modifying the original
    store_data = data.copy()
    
    # Add cluster labels to the data
    store_data['Cluster'] = best_labels
    
    # Create warehouse dataframe
    warehouse_locations = pd.DataFrame(best_centroids, columns=['Latitude', 'Longitude'])
    warehouse_locations['Warehouse_ID'] = warehouse_locations.index
    
    # Calculate distance from each store to its assigned warehouse
    for i, row in store_data.iterrows():
        warehouse = warehouse_locations.iloc[row['Cluster']]
        distance = haversine_distance(
            row['Latitude'], row['Longitude'],
            warehouse['Latitude'], warehouse['Longitude']
        )
        store_data.loc[i, 'Distance_km'] = distance
    
    # Calculate metrics per warehouse
    metrics = []
    for i in range(num_warehouses):
        cluster_stores = store_data[store_data['Cluster'] == i]
        metrics.append({
            'Warehouse_ID': i,
            'Latitude': warehouse_locations.iloc[i]['Latitude'],
            'Longitude': warehouse_locations.iloc[i]['Longitude'],
            'Num_Stores': len(cluster_stores),
            'Total_Sales': cluster_stores['Sales'].sum(),
            'Avg_Distance_km': cluster_stores['Distance_km'].mean() if len(cluster_stores) > 0 else 0,
            'Max_Distance_km': cluster_stores['Distance_km'].max() if len(cluster_stores) > 0 else 0,
            'Min_Distance_km': cluster_stores['Distance_km'].min() if len(cluster_stores) > 0 else 0
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Add optimization info
    st.success(f"Optimization complete! Found best solution on run {best_seed//100 + 1} of {num_runs}.")
    
    return store_data, warehouse_locations, metrics_df

# Function to create a download link for a dataframe
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Sidebar
st.sidebar.header("Configuration")

# Option to use sample data
use_sample = st.sidebar.checkbox("Use sample data", value=True)

if not use_sample:
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file with store data", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            # Check if required columns exist
            required_cols = ['Latitude', 'Longitude', 'Sales']
            if not all(col in data.columns for col in required_cols):
                st.sidebar.error(f"CSV must contain columns: {', '.join(required_cols)}")
                data = None
            else:
                st.sidebar.success(f"Successfully loaded {len(data)} stores.")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            data = None
    else:
        data = None
else:
    # Generate sample data
    data = generate_sample_data(100)
    st.sidebar.success("Using sample data with 100 stores across the US.")

# Basic configuration
st.sidebar.subheader("Basic Configuration")
# Number of warehouses slider
num_warehouses = st.sidebar.slider("Number of Warehouses", min_value=1, max_value=10, value=3)

# Sales weight slider (for optimization balance between sales and distance)
sales_weight = st.sidebar.slider(
    "Optimization Balance", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5,
    step=0.1,
    help="0 = Optimize for distance only, 1 = Heavily favor high-sales stores"
)

# Advanced configuration
st.sidebar.subheader("Advanced Configuration")
show_advanced = st.sidebar.checkbox("Show Advanced Options", value=False)

if show_advanced:
    # Minimum stores per warehouse
    min_stores = st.sidebar.number_input(
        "Minimum Stores per Warehouse",
        min_value=0,
        max_value=50,
        value=0,
        help="Minimum number of stores that must be assigned to each warehouse"
    )
    
    # Minimum sales per warehouse
    min_sales = st.sidebar.number_input(
        "Minimum Sales per Warehouse ($)",
        min_value=0,
        max_value=10000000,
        value=0,
        step=10000,
        help="Minimum total sales volume required for each warehouse"
    )
    
    # Number of optimization runs
    num_runs = st.sidebar.slider(
        "Number of Optimization Runs",
        min_value=1,
        max_value=20,
        value=10,
        help="Run optimization multiple times and select the best result"
    )
else:
    min_stores = 0
    min_sales = 0
    num_runs = 10

# Validate constraints based on data
if data is not None:
    total_stores = len(data)
    total_sales = data['Sales'].sum()
    
    if min_stores * num_warehouses > total_stores:
        st.sidebar.warning(f"⚠️ Warning: Minimum of {min_stores} stores per warehouse × {num_warehouses} warehouses exceeds total of {total_stores} stores.")
    
    if min_sales * num_warehouses > total_sales:
        st.sidebar.warning(f"⚠️ Warning: Minimum sales of ${min_sales:,} per warehouse × {num_warehouses} warehouses exceeds total sales of ${total_sales:,}.")

# Main content
if data is not None:
    # Display raw data
    with st.expander("View Raw Store Data"):
        st.dataframe(data)
    
    # Run optimization on button click
    if st.button("Optimize Warehouse Locations"):
        with st.spinner("Optimizing warehouse locations..."):
            # Run the optimization
            try:
                store_data, warehouse_locations, metrics_df = optimize_warehouse_locations(
                    data.copy(), 
                    num_warehouses,
                    sales_weight,
                    min_stores,
                    min_sales,
                    num_runs
                )
                
                # Show results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Map", "Warehouse Locations", "Metrics", "Download Results"])
                
                with tab1:
                    st.subheader("Store and Warehouse Locations")
                    
                    # Prepare data for PyDeck
                    store_data_for_map = store_data.copy()
                    # Normalize sales for sizing on map
                    max_sales = store_data_for_map['Sales'].max()
                    store_data_for_map['Sales_Normalized'] = store_data_for_map['Sales'] / max_sales
                    
                    # Create tooltip function
                    store_data_for_map['tooltip'] = store_data_for_map.apply(
                        lambda row: f"Store in {row['Region']}<br>Sales: ${int(row['Sales']):,}<br>Distance to Warehouse: {row['Distance_km']:.2f} km", 
                        axis=1
                    )
                    
                    # Create colors for clusters
                    cluster_colors = [
                        [239, 83, 80],  # Red
                        [66, 165, 245], # Blue
                        [76, 175, 80],  # Green
                        [171, 71, 188], # Purple
                        [255, 167, 38], # Orange
                        [77, 208, 225], # Teal
                        [255, 87, 34],  # Deep Orange
                        [141, 110, 99], # Brown
                        [0, 137, 123],  # Dark Teal
                        [63, 81, 181]   # Indigo
                    ]
                    
                    # Create a color column for each store based on its cluster
                    store_data_for_map['color'] = store_data_for_map['Cluster'].apply(
                        lambda x: cluster_colors[x % len(cluster_colors)]
                    )
                    
                    # Create warehouse data for map
                    warehouse_data_for_map = warehouse_locations.copy()
                    warehouse_data_for_map['color'] = warehouse_data_for_map['Warehouse_ID'].apply(
                        lambda x: cluster_colors[x % len(cluster_colors)]
                    )
                    
                    # Add metrics to warehouse data for tooltips
                    for i, row in warehouse_data_for_map.iterrows():
                        metrics_row = metrics_df[metrics_df['Warehouse_ID'] == row['Warehouse_ID']].iloc[0]
                        warehouse_data_for_map.loc[i, 'tooltip'] = (
                            f"Warehouse {int(row['Warehouse_ID'])}<br>"
                            f"Stores: {int(metrics_row['Num_Stores'])}<br>"
                            f"Total Sales: ${int(metrics_row['Total_Sales']):,}<br>"
                            f"Avg Distance: {metrics_row['Avg_Distance_km']:.2f} km"
                        )
                    
                    # Calculate a zoom level and center point
                    # Simple approach: center on mean lat/lon and use a fixed zoom
                    center_lat = store_data_for_map['Latitude'].mean()
                    center_lon = store_data_for_map['Longitude'].mean()
                    
                    # Create layers
                    # Store layer
                    store_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=store_data_for_map,
                        get_position=["Longitude", "Latitude"],
                        get_color="color",
                        get_radius=["Sales_Normalized * 50000", "Sales_Normalized * 50000"],
                        pickable=True,
                        opacity=0.8,
                        stroked=True,
                        filled=True,
                        auto_highlight=True,
                        id="store-layer"
                    )
                    
                    # Warehouse layer
                    warehouse_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=warehouse_data_for_map,
                        get_position=["Longitude", "Latitude"],
                        get_color="color",
                        get_radius=100000,  # Larger radius for warehouses
                        pickable=True,
                        opacity=0.9,
                        stroked=True,
                        filled=True,
                        auto_highlight=True,
                        id="warehouse-layer"
                    )
                    
                    # Create connections between warehouses and stores
                    # This requires creating a dataset of line segments
                    connection_data = []
                    for _, store in store_data_for_map.iterrows():
                        warehouse = warehouse_data_for_map[warehouse_data_for_map['Warehouse_ID'] == store['Cluster']].iloc[0]
                        connection_data.append({
                            'source_lat': store['Latitude'],
                            'source_lon': store['Longitude'],
                            'target_lat': warehouse['Latitude'],
                            'target_lon': warehouse['Longitude'],
                            'color': store['color']
                        })
                    
                    connection_df = pd.DataFrame(connection_data)
                    
                    # Create line layer
                    line_layer = pdk.Layer(
                        "LineLayer",
                        data=connection_df,
                        get_source_position=["source_lon", "source_lat"],
                        get_target_position=["target_lon", "target_lat"],
                        get_color="color",
                        get_width=3,
                        opacity=0.3,
                        pickable=False,
                    )
                    
                    # Create view state
                    view_state = pdk.ViewState(
                        latitude=center_lat,
                        longitude=center_lon,
                        zoom=3.5,
                        pitch=30,
                        bearing=0
                    )
                    
                    # Create tooltips
                    tooltip = {
                        "html": "<b>Info:</b> <br/> {tooltip} <br/>",
                        "style": {
                            "backgroundColor": "white",
                            "color": "black"
                        }
                    }
                    
                    # Create deck
                    r = pdk.Deck(
                        layers=[line_layer, store_layer, warehouse_layer],
                        initial_view_state=view_state,
                        map_style="light",
                        tooltip=tooltip
                    )
                    
                    # Display the map
                    st.pydeck_chart(r)
                    
                    # Legend
                    st.markdown("""
                    **Map Legend:**
                    - Circles: Stores (size indicates sales volume)
                    - Larger Circles: Warehouse locations
                    - Lines: Connection between stores and their assigned warehouse
                    - Colors: Warehouse assignment
                    """)
                
                with tab2:
                    st.subheader("Warehouse Locations")
                    
                    # Create an enhanced warehouse locations table
                    enhanced_metrics = metrics_df.copy()
                    enhanced_metrics = enhanced_metrics.sort_values(by='Warehouse_ID')
                    
                    # Format the metrics for display
                    enhanced_metrics['Formatted_Sales'] = enhanced_metrics['Total_Sales'].apply(lambda x: f"${x:,.2f}")
                    enhanced_metrics['Formatted_Avg_Distance'] = enhanced_metrics['Avg_Distance_km'].apply(lambda x: f"{x:.2f} km")
                    enhanced_metrics['Constraints_Met'] = enhanced_metrics.apply(
                        lambda row: "✅" if (row['Num_Stores'] >= min_stores and row['Total_Sales'] >= min_sales) else "❌",
                        axis=1
                    )
                    
                    # Display the enhanced table
                    st.dataframe(
                        enhanced_metrics[[
                            'Warehouse_ID', 'Latitude', 'Longitude', 
                            'Num_Stores', 'Formatted_Sales', 'Formatted_Avg_Distance',
                            'Constraints_Met'
                        ]].rename(columns={
                            'Warehouse_ID': 'ID',
                            'Formatted_Sales': 'Total Sales',
                            'Formatted_Avg_Distance': 'Avg Distance',
                            'Num_Stores': 'Stores',
                            'Constraints_Met': 'Meets Constraints'
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Display additional constraint information
                    if min_stores > 0 or min_sales > 0:
                        st.markdown("### Constraint Summary")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if min_stores > 0:
                                stores_constraint_met = all(enhanced_metrics['Num_Stores'] >= min_stores)
                                st.metric(
                                    "Min Stores per Warehouse", 
                                    min_stores,
                                    delta="Met" if stores_constraint_met else "Not Met",
                                    delta_color="normal" if stores_constraint_met else "inverse"
                                )
                        
                        with col2:
                            if min_sales > 0:
                                sales_constraint_met = all(enhanced_metrics['Total_Sales'] >= min_sales)
                                st.metric(
                                    "Min Sales per Warehouse", 
                                    f"${min_sales:,}",
                                    delta="Met" if sales_constraint_met else "Not Met",
                                    delta_color="normal" if sales_constraint_met else "inverse"
                                )
                
                with tab3:
                    st.subheader("Warehouse Performance Metrics")
                    
                    # Format metrics for display
                    formatted_metrics = metrics_df.copy()
                    formatted_metrics['Total_Sales'] = formatted_metrics['Total_Sales'].apply(lambda x: f"${x:,.2f}")
                    formatted_metrics['Avg_Distance_km'] = formatted_metrics['Avg_Distance_km'].apply(lambda x: f"{x:.2f}")
                    formatted_metrics['Max_Distance_km'] = formatted_metrics['Max_Distance_km'].apply(lambda x: f"{x:.2f}")
                    formatted_metrics['Min_Distance_km'] = formatted_metrics['Min_Distance_km'].apply(lambda x: f"{x:.2f}")
                    
                    # Create bar charts for metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart for number of stores
                        stores_chart_data = metrics_df.copy()
                        st.bar_chart(
                            stores_chart_data.set_index('Warehouse_ID')['Num_Stores'],
                            use_container_width=True
                        )
                        st.caption("Number of Stores per Warehouse")
                    
                    with col2:
                        # Bar chart for total sales
                        sales_chart_data = metrics_df.copy()
                        st.bar_chart(
                            sales_chart_data.set_index('Warehouse_ID')['Total_Sales'],
                            use_container_width=True
                        )
                        st.caption("Total Sales per Warehouse ($)")
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        # Bar chart for average distance
                        distance_chart_data = metrics_df.copy()
                        st.bar_chart(
                            distance_chart_data.set_index('Warehouse_ID')['Avg_Distance_km'],
                            use_container_width=True
                        )
                        st.caption("Average Distance to Warehouse (km)")
                    
                    with col4:
                        # Create a histogram using numerical bin indices instead of interval objects
                        # Get min and max distance values
                        min_dist = store_data['Distance_km'].min()
                        max_dist = store_data['Distance_km'].max()
                        
                        # Create 10 bins with numerical labels
                        bins = 10
                        bin_width = (max_dist - min_dist) / bins
                        
                        # Create histogram data manually
                        hist_data = {}
                        for i in range(bins):
                            lower = min_dist + i * bin_width
                            upper = min_dist + (i + 1) * bin_width
                            bin_label = f"{lower:.1f}-{upper:.1f}"
                            # Count points in this bin
                            count = len(store_data[(store_data['Distance_km'] >= lower) & 
                                                  (store_data['Distance_km'] < upper)])
                            hist_data[bin_label] = count
                        
                        # Create a dataframe for the histogram
                        hist_df = pd.DataFrame(list(hist_data.items()), 
                                              columns=['Distance Range (km)', 'Count'])
                        
                        # Display as a bar chart
                        st.bar_chart(hist_df.set_index('Distance Range (km)'), use_container_width=True)
                        st.caption("Distribution of Store-to-Warehouse Distances")
                    
                    # Display summary metrics at the bottom
                    st.markdown("### Summary Metrics")
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    
                    with summary_col1:
                        avg_stores = metrics_df['Num_Stores'].mean()
                        st.metric("Avg Stores per Warehouse", f"{avg_stores:.1f}")
                    
                    with summary_col2:
                        avg_sales = metrics_df['Total_Sales'].mean()
                        st.metric("Avg Sales per Warehouse", f"${avg_sales:,.2f}")
                    
                    with summary_col3:
                        avg_distance = store_data['Distance_km'].mean()
                        st.metric("Avg Store Distance", f"{avg_distance:.2f} km")
                    
                    with summary_col4:
                        max_distance = store_data['Distance_km'].max()
                        st.metric("Max Store Distance", f"{max_distance:.2f} km")
                
                with tab4:
                    st.subheader("Download Results")
                    
                    # Provide download links
                    st.markdown(get_csv_download_link(
                        warehouse_locations, 
                        'warehouse_locations.csv',
                        'Download Warehouse Locations as CSV'
                    ), unsafe_allow_html=True)
                    
                    st.markdown(get_csv_download_link(
                        store_data,
                        'store_assignments.csv',
                        'Download Store Assignments as CSV'
                    ), unsafe_allow_html=True)
                    
                    # Also display the warehouse coordinates
                    st.subheader("Warehouse Coordinates")
                    st.dataframe(warehouse_locations[['Warehouse_ID', 'Latitude', 'Longitude']])
                    
                    # Add optimization details
                    st.subheader("Optimization Details")
                    optimization_details = {
                        'Parameter': [
                            'Number of Warehouses',
                            'Sales Weight',
                            'Minimum Stores per Warehouse',
                            'Minimum Sales per Warehouse',
                            'Number of Optimization Runs'
                        ],
                        'Value': [
                            num_warehouses,
                            f"{sales_weight} ({100*sales_weight}% sales / {100*(1-sales_weight)}% distance)",
                            min_stores,
                            f"${min_sales:,}",
                            num_runs
                        ]
                    }
                    st.dataframe(pd.DataFrame(optimization_details), hide_index=True)
                    
                    # Add a summary report that can be downloaded
                    st.subheader("Generate Summary Report")
                    
                    if st.button("Generate Report"):
                        # Create a summary report as HTML
                        report_html = f"""
                        <html>
                        <head>
                            <title>Warehouse Optimization Report</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                h1, h2 {{ color: #2c3e50; }}
                                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                                th {{ background-color: #f2f2f2; }}
                                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                                .constraints {{ margin-top: 20px; margin-bottom: 20px; }}
                            </style>
                        </head>
                        <body>
                            <h1>Warehouse Optimization Report</h1>
                            <div class="summary">
                                <h2>Summary</h2>
                                <p>Number of Warehouses: {num_warehouses}</p>
                                <p>Total Stores: {len(store_data)}</p>
                                <p>Total Sales: ${store_data['Sales'].sum():,.2f}</p>
                                <p>Average Store Distance: {store_data['Distance_km'].mean():.2f} km</p>
                                <p>Optimization Runs: {num_runs}</p>
                            </div>
                            
                            <div class="constraints">
                                <h2>Optimization Parameters</h2>
                                <p>Sales Weight: {sales_weight} ({100*sales_weight}% sales / {100*(1-sales_weight)}% distance)</p>
                                <p>Minimum Stores per Warehouse: {min_stores}</p>
                                <p>Minimum Sales per Warehouse: ${min_sales:,}</p>
                            </div>
                            
                            <h2>Warehouse Locations</h2>
                            <table>
                                <tr>
                                    <th>ID</th>
                                    <th>Latitude</th>
                                    <th>Longitude</th>
                                    <th>Stores</th>
                                    <th>Total Sales</th>
                                    <th>Avg Distance</th>
                                    <th>Meets Constraints</th>
                                </tr>
                        """
                        
                        # Add warehouse data to the report
                        for _, row in metrics_df.iterrows():
                            # Check if constraints are met
                            constraints_met = (row['Num_Stores'] >= min_stores and row['Total_Sales'] >= min_sales)
                            constraints_icon = "✅" if constraints_met else "❌"
                            
                            report_html += f"""
                                <tr>
                                    <td>{int(row['Warehouse_ID'])}</td>
                                    <td>{row['Latitude']:.6f}</td>
                                    <td>{row['Longitude']:.6f}</td>
                                    <td>{int(row['Num_Stores'])}</td>
                                    <td>${row['Total_Sales']:,.2f}</td>
                                    <td>{row['Avg_Distance_km']:.2f} km</td>
                                    <td>{constraints_icon}</td>
                                </tr>
                            """
                        
                        # Complete the HTML
                        report_html += """
                            </table>
                            
                            <h2>Generated on</h2>
                            <p>Report generated by Warehouse Location Optimizer</p>
                        </body>
                        </html>
                        """
                        
                        # Convert HTML to base64 for download
                        b64 = base64.b64encode(report_html.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="warehouse_report.html">Download HTML Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during optimization: {str(e)}")
                st.error("Error details:")
                st.exception(e)
else:
    if not use_sample:
        st.info("Please upload a CSV file with store data or use the sample data option.")
