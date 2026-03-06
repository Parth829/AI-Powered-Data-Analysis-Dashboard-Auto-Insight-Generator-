import streamlit as st
import pandas as pd
from components.upload import render_upload_section
from components.dashboard import render_dashboard
from utils.data_cleaning import clean_data
from utils.visualization import plot_numeric_distributions, plot_categorical_distributions, plot_correlation_heatmap, plot_time_series
from utils.insights import generate_insights

st.set_page_config(
    page_title="AI Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 AI-Powered Data Analysis Dashboard")
    st.markdown("Upload your dataset to automatically generate insights, visualizations, and summary statistics.")

    # Sidebar for navigation or settings
    with st.sidebar:
        st.header("Settings")
        st.markdown("Configure your analysis preferences here.")

    # File upload component
    df_raw = render_upload_section()

    if df_raw is not None:
        st.success("File uploaded successfully!")
        
        # Clean data automatically
        with st.spinner("Cleaning and preprocessing data..."):
            df_clean, cleaning_summary = clean_data(df_raw)
            
        st.info(f"**Data Cleaning Summary**: {cleaning_summary}")
        
        # Store in session state so other components can access it
        st.session_state['df'] = df_clean
        
        # Interactive Filters
        st.sidebar.subheader("Filter Data")
        cat_cols = [col for col in df_clean.columns if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category']
        df_filtered = df_clean.copy()
        
        if cat_cols:
            filter_col = st.sidebar.selectbox("Filter by Category:", ["None"] + cat_cols, key='filter_col')
            if filter_col != "None":
                unique_vals = df_clean[filter_col].dropna().unique()
                selected_vals = st.sidebar.multiselect(f"Select {filter_col} values", unique_vals, default=unique_vals, key='filter_vals')
                if selected_vals:
                    df_filtered = df_clean[df_clean[filter_col].isin(selected_vals)]
                    st.sidebar.info(f"Filtered {len(df_clean) - len(df_filtered)} rows out.")
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Dataset Overview", 
            "Auto Visualizations", 
            "Correlation Analysis", 
            "Business Insights",
            "Custom Dashboard"
        ])
        
        with tab1:
            render_dashboard(df_filtered)
            
        with tab2:
            st.header("Automated Visualizations")
            st.markdown("We automatically generated these charts based on your data types.")
            
            st.subheader("Numeric Distributions")
            plot_numeric_distributions(df_filtered)
            
            st.markdown("---")
            st.subheader("Categorical Distributions")
            plot_categorical_distributions(df_filtered)
            
            st.markdown("---")
            st.subheader("Time Series Analysis")
            plot_time_series(df_filtered)
            
        with tab3:
            st.header("Correlation Analysis")
            st.markdown("Discover relationships between numeric variables.")
            plot_correlation_heatmap(df_filtered)

            
        with tab4:
            st.header("💡 Business Insights")
            st.markdown("Automatically generated statistical insights based on your dataset.")
            insights = generate_insights(df_filtered)
            
            if insights:
                for idx, insight in enumerate(insights):
                    st.info(insight)
            else:
                st.warning("Could not generate significant insights for this dataset.")
            
        with tab5:
            st.header("🛠️ Custom Dashboard Builder")
            st.markdown("Build your own custom visualizations.")
            
            col_x, col_y, col_chart = st.columns(3)
            with col_x:
                x_axis = st.selectbox("X-Axis", df_filtered.columns, key='custom_x')
            with col_y:
                y_axis = st.selectbox("Y-Axis", df_filtered.columns, key='custom_y')
            with col_chart:
                chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "BoxPlot"], key='custom_chart')
                
            if st.button("Generate Custom Chart"):
                import plotly.express as px
                
                try:
                    if chart_type == "Bar":
                        # Aggregate if needed to prevent huge bar charts
                        agg_df = df_filtered.groupby(x_axis)[y_axis].sum().reset_index()
                        fig = px.bar(agg_df, x=x_axis, y=y_axis, title=f"Bar Chart: {y_axis} by {x_axis}")
                    elif chart_type == "Line":
                        fig = px.line(df_filtered.sort_values(by=x_axis), x=x_axis, y=y_axis, title=f"Line Chart: {y_axis} over {x_axis}")
                    elif chart_type == "Scatter":
                        fig = px.scatter(df_filtered, x=x_axis, y=y_axis, title=f"Scatter Plot: {y_axis} vs {x_axis}")
                    elif chart_type == "BoxPlot":
                        fig = px.box(df_filtered, x=x_axis, y=y_axis, title=f"Box Plot: {y_axis} grouped by {x_axis}")
                        
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate chart. Error: {str(e)}")

if __name__ == "__main__":
    main()
