"""
LLM Evaluation Dashboard - Main Streamlit Application

A comprehensive dashboard for visualizing and analyzing LLM evaluation results,
including performance metrics, cost analysis, and model comparisons.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from typing import Dict, Any, List

# Add the dashboard directory to Python path
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dashboard_dir)

from data_loader import EvaluationDataLoader, format_model_name, calculate_efficiency_score, calculate_throughput_efficiency_score
from visualizations import EvaluationVisualizer, MetricCalculator

# Page configuration
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_evaluation_data(data_directory: str) -> Dict[str, Any]:
    """Load and cache evaluation data."""
    loader = EvaluationDataLoader(data_directory)
    return loader.load_all_data()

@st.cache_data
def get_summary_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics."""
    performance_df = data.get('performance_metrics', pd.DataFrame())
    cost_df = data.get('cost_metrics', pd.DataFrame())
    comparison_df = data.get('model_comparison', pd.DataFrame())
    
    stats = {
        'total_models': len(comparison_df['model'].unique()) if not comparison_df.empty else 0,
        'total_benchmarks': len(performance_df['benchmark'].unique()) if not performance_df.empty else 0,
        'avg_accuracy': performance_df['accuracy'].mean() if not performance_df.empty else 0,
        'avg_cost_per_1m': cost_df['cost_per_1m_tokens_usd'].mean() if not cost_df.empty else 0
    }
    
    return stats

def main():
    """Main dashboard application."""
    
    # Title and description
    st.title("LLM Evaluation Dashboard")
    st.markdown("""
    **Comprehensive analysis of Large Language Model evaluation results**
    
    This dashboard provides insights into model performance, cost efficiency, and throughput metrics 
    across different benchmarks and evaluation scenarios.
    """)
    
    # Sidebar configuration
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Data directory selection
    default_data_dir = os.path.join(dashboard_dir, "data")
    data_directory = st.sidebar.text_input(
        "Evaluation Data Directory",
        value=default_data_dir,
        help="Path to directory containing evaluation JSON files"
    )
    
    if not os.path.exists(data_directory):
        st.error(f"❌ Data directory not found: {data_directory}")
        st.stop()
    
    # Load data
    try:
        with st.spinner("Loading evaluation data..."):
            data = load_evaluation_data(data_directory)
        
        if not data:
            st.error("❌ No evaluation data found in the specified directory.")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    # Get summary statistics
    stats = get_summary_stats(data)
    
    # Display summary metrics
    st.markdown("## 📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Models Evaluated",
            value=stats['total_models']
        )
        
    with col2:
        st.metric(
            label="Benchmarks",
            value=stats['total_benchmarks']
        )
        
    with col3:
        st.metric(
            label="Avg Accuracy",
            value=f"{stats['avg_accuracy']:.3f}"
        )
        
    with col4:
        st.metric(
            label="Avg Cost/1M Tokens",
            value=f"${stats['avg_cost_per_1m']:.4f}"
        )
    
    st.markdown("---")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Performance & TopN Analysis", "💰 Cost Analysis", "⚡ Throughput Analysis", "🔍 Model Comparison", "📊 Advanced Analytics"
    ])
    
    # Initialize visualizer
    visualizer = EvaluationVisualizer()
    
    # Performance Analysis Tab
    with tab1:
        st.header("🎯 Performance & TopN Analysis")
        
        performance_df = data.get('performance_metrics', pd.DataFrame())
        
        if not performance_df.empty:
            # Performance metric selector
            col1, col2 = st.columns([1, 3])
            
            with col1:
                available_metrics = [col for col in performance_df.columns if col.startswith(('accuracy', 'top'))]
                if available_metrics:
                    selected_metric = st.selectbox(
                        "Select Metric",
                        available_metrics,
                        index=0 if 'accuracy' in available_metrics else 0
                    )
                else:
                    selected_metric = None
                    st.warning("⚠️ No performance metrics available for selection.")
            
            with col2:
                st.empty()  # Spacer
            
            # Performance comparison chart
            st.write("**Performance Comparison**")
            
            if selected_metric is not None:
                fig = visualizer.create_performance_comparison(performance_df, selected_metric)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No metric selected for performance comparison.")
            
            # TopN Analysis Section
            st.subheader("🔢 TopN Analysis")
            
            # TopN comparison
            topn_fig = visualizer.create_topn_comparison(performance_df)
            st.plotly_chart(topn_fig, width='stretch')
            
            # Detailed TopN Analysis
            st.subheader("📈 TopN Performance Progression")
            
            model_options = ['All Models'] + list(performance_df['model'].unique())
            selected_model_topn = st.selectbox(
                "Select Model for TopN Analysis",
                model_options,
                format_func=lambda x: x if x == 'All Models' else format_model_name(x)
            )
            
            if selected_model_topn == 'All Models':
                topn_detailed_fig = visualizer.create_topn_detailed_analysis(performance_df)
            else:
                topn_detailed_fig = visualizer.create_topn_detailed_analysis(performance_df, selected_model_topn)
            
            st.plotly_chart(topn_detailed_fig, width='stretch')
            
            # Performance heatmap
            st.subheader("Performance Heatmap")
            
            heatmap_fig = visualizer.create_benchmark_heatmap(performance_df)
            st.plotly_chart(heatmap_fig, width='stretch')
            
            # Performance data table
            st.subheader("Detailed Performance Data")
            formatted_df = performance_df.copy()
            formatted_df['model'] = formatted_df['model'].apply(format_model_name)
            
            st.dataframe(formatted_df, width='stretch')
            
        else:
            st.warning("⚠️ No performance data available.")
    
    # Cost Analysis Tab
    with tab2:
        st.header("💰 Cost Analysis")
        
        cost_df = data.get('cost_metrics', pd.DataFrame())
        
        if not cost_df.empty:
            # Model selection checkboxes
            st.subheader("🔧 Model Selection")
            available_models = cost_df['model'].unique()
            
            # Create checkboxes for model selection
            selected_models = []
            cols = st.columns(min(len(available_models), 3))  # Max 3 columns
            
            for i, model in enumerate(available_models):
                with cols[i % len(cols)]:
                    if st.checkbox(format_model_name(model), value=True, key=f"cost_model_{i}"):
                        selected_models.append(model)
            
            if not selected_models:
                st.warning("⚠️ Please select at least one model to display cost analysis.")
                return
            
            # Filter data based on selected models
            filtered_cost_df = cost_df[cost_df['model'].isin(selected_models)]
            
            # Main cost analysis chart
            st.subheader("Cost per 1M Tokens")
            
            cost_fig = visualizer.create_cost_analysis(filtered_cost_df)
            st.plotly_chart(cost_fig, width='stretch')
            
            # Cost data table
            st.subheader("Detailed Cost Data")
            formatted_cost_df = filtered_cost_df.copy()
            formatted_cost_df['model'] = formatted_cost_df['model'].apply(format_model_name)
            
            # Select only relevant columns, excluding run cost, execution time, and price per input
            columns_to_show = ['model', 'cost_per_1k_tokens_usd', 'cost_per_1m_tokens_usd', 'mode', 'source']
            # Only include columns that exist in the dataframe
            available_columns = [col for col in columns_to_show if col in formatted_cost_df.columns]
            formatted_cost_df_filtered = formatted_cost_df[available_columns]
            
            st.dataframe(formatted_cost_df_filtered, width='stretch')
            
        else:
            st.warning("⚠️ No cost data available.")
    
    # Throughput Analysis Tab
    with tab3:
        st.header("⚡ Throughput Analysis")
        
        throughput_df = data.get('throughput_metrics', pd.DataFrame())
        
        if not throughput_df.empty:
            # Model selection checkboxes for throughput
            st.subheader("🔧 Model Selection")
            available_models = throughput_df['model'].unique()
            
            # Create checkboxes for model selection
            selected_models_throughput = []
            cols = st.columns(min(len(available_models), 3))  # Max 3 columns
            
            for i, model in enumerate(available_models):
                with cols[i % len(cols)]:
                    if st.checkbox(format_model_name(model), value=True, key=f"throughput_model_{i}"):
                        selected_models_throughput.append(model)
            
            if not selected_models_throughput:
                st.warning("⚠️ Please select at least one model to display throughput analysis.")
            else:
                # Filter data based on selected models
                filtered_throughput_df = throughput_df[throughput_df['model'].isin(selected_models_throughput)]
                
                # Throughput metrics overview
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_tps = filtered_throughput_df['tokens_per_second'].mean()
                    st.metric("Avg Tokens/Sec", f"{avg_tps:.1f}")
                
                with col2:
                    avg_ttft = filtered_throughput_df['estimated_ttft_seconds'].mean()
                    st.metric("Avg Est. TTFT (s)", f"{avg_ttft:.2f}")
                
                with col3:
                    avg_total_time = filtered_throughput_df['elapsed_seconds'].mean()
                    st.metric("Avg Total Time (s)", f"{avg_total_time:.1f}")
                
                st.markdown("---")
                
                # Main throughput analysis chart
                st.subheader("Tokens per Second Analysis")
                throughput_fig = visualizer.create_throughput_analysis(filtered_throughput_df)
                st.plotly_chart(throughput_fig, width='stretch')
                
                # TTFT analysis
                st.subheader("Time to First Token Analysis")
                ttft_fig = visualizer.create_ttft_analysis(filtered_throughput_df)
                st.plotly_chart(ttft_fig, width='stretch')
                
                # Throughput efficiency analysis
                comparison_df = data.get('model_comparison', pd.DataFrame())
                if not comparison_df.empty:
                    filtered_comparison_df = comparison_df[comparison_df['model'].isin(selected_models_throughput)]
                    
                    st.subheader("Throughput Efficiency: Accuracy vs Speed")
                    throughput_efficiency_fig = visualizer.create_throughput_efficiency_scatter(filtered_comparison_df)
                    st.plotly_chart(throughput_efficiency_fig, width='stretch')
                
                # Throughput data table
                st.subheader("Detailed Throughput Data")
                formatted_throughput_df = filtered_throughput_df.copy()
                formatted_throughput_df['model'] = formatted_throughput_df['model'].apply(format_model_name)
                
                # Select relevant columns for display
                columns_to_show = ['model', 'tokens_per_second', 
                                 'estimated_ttft_seconds', 'elapsed_seconds', 'total_tokens', 'mode', 'source']
                available_columns = [col for col in columns_to_show if col in formatted_throughput_df.columns]
                formatted_throughput_df_filtered = formatted_throughput_df[available_columns]
                
                st.dataframe(formatted_throughput_df_filtered, width='stretch')
                
        else:
            st.warning("⚠️ No throughput data available.")
    
    # Model Comparison Tab
    with tab4:
        st.header("🔍 Model Comparison")
        
        comparison_df = data.get('model_comparison', pd.DataFrame())
        
        if not comparison_df.empty:
            # Efficiency scatter plot
            st.subheader("Efficiency Analysis: Accuracy vs Cost")
            
            efficiency_fig = visualizer.create_efficiency_scatter(comparison_df)
            st.plotly_chart(efficiency_fig, width='stretch')
            
            # Model selector for radar chart
            st.subheader("Model Profile Analysis")
            
            model_options = comparison_df['model'].unique()
            selected_model_radar = st.selectbox(
                "Select Model for Profile Analysis",
                model_options,
                format_func=format_model_name,
                key="radar_model_selector"
            )
            
            radar_fig = visualizer.create_model_radar_chart(comparison_df, selected_model_radar)
            st.plotly_chart(radar_fig, width='stretch')
            
            # Model ranking
            st.subheader("Model Rankings")
            
            # Calculate efficiency scores
            comparison_df_copy = comparison_df.copy()
            comparison_df_copy['efficiency_score'] = comparison_df_copy.apply(
                lambda row: calculate_efficiency_score(
                    row['avg_accuracy'],
                    row['cost_per_1m_tokens']
                ), axis=1
            )
            
            # Sort by efficiency score
            ranking_df = comparison_df_copy.sort_values('efficiency_score', ascending=False)
            ranking_df['model'] = ranking_df['model'].apply(format_model_name)
            ranking_df['rank'] = range(1, len(ranking_df) + 1)
            
            # Display ranking table
            ranking_display = ranking_df[['rank', 'model', 'avg_accuracy', 'cost_per_1m_tokens', 
                                        'efficiency_score']].round(4)
            st.dataframe(ranking_display, width='stretch')
            
        else:
            st.warning("⚠️ No model comparison data available.")
    
    # Advanced Analytics Tab
    with tab5:
        st.header("📊 Advanced Analytics")
        
        if not data or all(df.empty for df in data.values() if isinstance(df, pd.DataFrame)):
            st.warning("⚠️ No data available for advanced analytics.")
        else:
            # Custom metric calculator
            st.subheader("Custom Efficiency Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy_weight = st.slider("Accuracy Weight", 0.0, 1.0, 0.5, 0.1)
            with col2:
                cost_weight = st.slider("Cost Weight", 0.0, 1.0, 0.3, 0.1)
            with col3:
                throughput_weight = st.slider("Throughput Weight", 0.0, 1.0, 0.2, 0.1)
            
            # Normalize weights
            total_weight = accuracy_weight + cost_weight + throughput_weight
            if total_weight > 0:
                accuracy_weight /= total_weight
                cost_weight /= total_weight
                throughput_weight /= total_weight
            
            comparison_df = data.get('model_comparison', pd.DataFrame())
            
            if not comparison_df.empty:
                # Calculate custom efficiency scores including throughput
                custom_scores = []
                for _, row in comparison_df.iterrows():
                    # Normalize cost (invert so lower cost = higher score)
                    max_cost = comparison_df['cost_per_1m_tokens'].max()
                    normalized_cost = (max_cost - row['cost_per_1m_tokens']) / max_cost if max_cost > 0 else 0
                    
                    # Normalize throughput
                    max_throughput = comparison_df['tokens_per_second'].max()
                    normalized_throughput = row['tokens_per_second'] / max_throughput if max_throughput > 0 else 0
                    
                    score = (row['avg_accuracy'] * accuracy_weight + 
                            normalized_cost * cost_weight +
                            normalized_throughput * throughput_weight)
                    custom_scores.append(score)
                
                # Create custom ranking
                custom_df = comparison_df.copy()
                custom_df['custom_efficiency'] = custom_scores
                custom_df = custom_df.sort_values('custom_efficiency', ascending=False)
                custom_df['model'] = custom_df['model'].apply(format_model_name)
                
                # Display custom ranking
                st.subheader("Custom Efficiency Ranking")
                display_columns = ['model', 'avg_accuracy', 'cost_per_1m_tokens', 'tokens_per_second', 'custom_efficiency']
                available_display_columns = [col for col in display_columns if col in custom_df.columns]
                custom_display = custom_df[available_display_columns].round(4)
                st.dataframe(custom_display, width='stretch')
                
                # Visualize custom efficiency
                fig_custom = px.bar(
                    custom_df.head(10),  # Top 10 models
                    x='model',
                    y='custom_efficiency',
                    title='Custom Efficiency Scores (Top 10)',
                    labels={'model': 'Model', 'custom_efficiency': 'Custom Efficiency Score'}
                )
                fig_custom.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_custom, width='stretch')
            
            # Data export section
            st.subheader("📥 Data Export")
            
            export_data = st.selectbox(
                "Select data to export",
                ["Performance Metrics", "Cost Metrics", "Model Comparison"]
            )
            
            if st.button("Generate CSV Download"):
                if export_data == "Performance Metrics":
                    csv_data = data.get('performance_metrics', pd.DataFrame())
                elif export_data == "Cost Metrics":
                    csv_data = data.get('cost_metrics', pd.DataFrame())
                else:  # Model Comparison
                    csv_data = data.get('model_comparison', pd.DataFrame())
                
                if not csv_data.empty:
                    csv_string = csv_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_string,
                        file_name=f"{export_data.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No data available for export.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>LLM Evaluation Dashboard | Built with Streamlit & Plotly</p>
        <p>Data source: {}</p>
    </div>
    """.format(data_directory), unsafe_allow_html=True)

if __name__ == "__main__":
    main()