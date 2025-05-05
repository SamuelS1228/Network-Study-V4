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
