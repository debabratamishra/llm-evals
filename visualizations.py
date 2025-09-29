"""
Visualization components for LLM evaluation dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional
import numpy as np

class EvaluationVisualizer:
    """Class to create visualizations for LLM evaluation data."""
    
    def __init__(self):
        """Initialize the visualizer with color schemes and themes."""
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        self.theme = {
            'background_color': '#ffffff',
            'grid_color': '#f0f0f0',
            'text_color': '#333333'
        }
    
    def create_performance_comparison(self, performance_df: pd.DataFrame, 
                                    metric: str = 'accuracy') -> go.Figure:
        """
        Create a bar chart comparing model performance across benchmarks.
        
        Args:
            performance_df: DataFrame with performance metrics
            metric: Metric to visualize ('accuracy', 'top1', 'top2', etc.)
        
        Returns:
            Plotly figure
        """
        if performance_df.empty:
            return self._create_empty_figure("No performance data available")
        
        fig = px.bar(
            performance_df,
            x='model',
            y=metric,
            color='benchmark',
            barmode='group',
            title=f'Model Performance Comparison - {metric.title()}',
            labels={
                'model': 'Model',
                metric: metric.title(),
                'benchmark': 'Benchmark'
            }
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=True,
            plot_bgcolor=self.theme['background_color']
        )
        
        # Format model names for better readability
        fig.update_xaxes(
            ticktext=[self._format_model_name(name) for name in performance_df['model'].unique()],
            tickvals=performance_df['model'].unique()
        )
        
        return fig
    
    def create_cost_analysis(self, cost_df: pd.DataFrame) -> go.Figure:
        """
        Create a cost analysis visualization focusing on cost per token.
        
        Args:
            cost_df: DataFrame with cost metrics
        
        Returns:
            Plotly figure
        """
        if cost_df.empty:
            return self._create_empty_figure("No cost data available")
        
        # Create single plot for cost per 1M tokens
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=[self._format_model_name(name) for name in cost_df['model']],
                y=cost_df['cost_per_1m_tokens_usd'],
                name='Cost per 1M Tokens ($)',
                marker_color=self.color_palette[0]
            )
        )
        
        fig.update_layout(
            title='Cost per 1M Tokens Comparison',
            xaxis_title='Model',
            yaxis_title='Cost per 1M Tokens ($)',
            height=500,
            showlegend=False,
            plot_bgcolor=self.theme['background_color']
        )
        
        fig.update_xaxes(tickangle=-45)
        
        return fig
    
    def create_topn_comparison(self, performance_df: pd.DataFrame) -> go.Figure:
        """
        Create a comprehensive TopN analysis comparison.
        
        Args:
            performance_df: DataFrame with performance metrics
        
        Returns:
            Plotly figure showing top@1, top@2, top@3, top@5 comparison
        """
        if performance_df.empty:
            return self._create_empty_figure("No performance data available")
        
        # Filter for TopN metrics
        topn_metrics = ['top1', 'top2', 'top5']  # Only use available metrics
        available_metrics = [col for col in topn_metrics if col in performance_df.columns and performance_df[col].notna().any()]
        
        if not available_metrics:
            return self._create_empty_figure("No TopN metrics available")
        
        # Create subplot for each benchmark
        benchmarks = performance_df['benchmark'].unique()
        
        if len(benchmarks) == 1:
            # Single benchmark - show all TopN metrics
            benchmark_data = performance_df[performance_df['benchmark'] == benchmarks[0]]
            
            fig = go.Figure()
            
            for i, metric in enumerate(available_metrics):
                fig.add_trace(go.Bar(
                    name=f'Top@{metric[-1]}',
                    x=[self._format_model_name(name) for name in benchmark_data['model']],
                    y=benchmark_data[metric],
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ))
            
            fig.update_layout(
                title=f'TopN Performance Comparison - {benchmarks[0]}',
                xaxis_title='Model',
                yaxis_title='Score',
                barmode='group',
                height=500,
                plot_bgcolor=self.theme['background_color']
            )
            
        else:
            # Multiple benchmarks - create subplots
            fig = make_subplots(
                rows=1, cols=len(benchmarks),
                subplot_titles=[bench.replace('_', ' ').title() for bench in benchmarks],
                shared_yaxes=True
            )
            
            for i, benchmark in enumerate(benchmarks):
                benchmark_data = performance_df[performance_df['benchmark'] == benchmark]
                
                for j, metric in enumerate(available_metrics):
                    fig.add_trace(
                        go.Bar(
                            name=f'Top@{metric[-1]}' if i == 0 else f'Top@{metric[-1]}',
                            x=[self._format_model_name(name) for name in benchmark_data['model']],
                            y=benchmark_data[metric],
                            marker_color=self.color_palette[j % len(self.color_palette)],
                            showlegend=(i == 0)  # Only show legend for first subplot
                        ),
                        row=1, col=i+1
                    )
            
            fig.update_layout(
                title='TopN Performance Comparison Across Benchmarks',
                height=500,
                plot_bgcolor=self.theme['background_color']
            )
            
            fig.update_xaxes(tickangle=-45)
        
        return fig
    
    def create_topn_detailed_analysis(self, performance_df: pd.DataFrame, 
                                    selected_model: str = None) -> go.Figure:
        """
        Create a detailed TopN analysis for a specific model or all models.
        
        Args:
            performance_df: DataFrame with performance metrics
            selected_model: Optional specific model to focus on
        
        Returns:
            Plotly figure showing detailed TopN breakdown
        """
        if performance_df.empty:
            return self._create_empty_figure("No performance data available")
        
        if selected_model:
            data = performance_df[performance_df['model'] == selected_model]
            if data.empty:
                return self._create_empty_figure(f"No data available for {selected_model}")
        else:
            data = performance_df
        
        # Create a line plot showing TopN progression
        topn_metrics = ['top1', 'top2', 'top5']  # Only use available metrics
        available_metrics = [col for col in topn_metrics if col in data.columns and data[col].notna().any()]
        
        if not available_metrics:
            return self._create_empty_figure("No TopN metrics available")
        
        fig = go.Figure()
        
        # If specific model selected, show progression across benchmarks
        if selected_model:
            for benchmark in data['benchmark'].unique():
                benchmark_data = data[data['benchmark'] == benchmark].iloc[0]
                x_vals = [int(metric[-1]) for metric in available_metrics]  # Extract the N from topN
                y_vals = [benchmark_data[metric] for metric in available_metrics]
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name=benchmark.replace('_', ' ').title(),
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            title = f'TopN Performance Progression - {self._format_model_name(selected_model)}'
            
        else:
            # Show all models, average across benchmarks
            for model in data['model'].unique():
                model_data = data[data['model'] == model]
                avg_metrics = {}
                
                for metric in available_metrics:
                    avg_metrics[metric] = model_data[metric].mean()
                
                x_vals = [int(metric[-1]) for metric in available_metrics]
                y_vals = [avg_metrics[metric] for metric in available_metrics]
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name=self._format_model_name(model),
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            title = 'TopN Performance Progression - All Models'
        
        fig.update_layout(
            title=title,
            xaxis_title='Top-N',
            yaxis_title='Score',
            height=500,
            plot_bgcolor=self.theme['background_color'],
            xaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 5],
                ticktext=['Top@1', 'Top@2', 'Top@5']
            )
        )
        
        return fig
    
    def create_efficiency_scatter(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Create a scatter plot showing efficiency (accuracy vs cost).
        
        Args:
            comparison_df: DataFrame with model comparison data
        
        Returns:
            Plotly figure
        """
        if comparison_df.empty:
            return self._create_empty_figure("No comparison data available")
        
        # Filter out models with zero cost (to avoid log scale issues)
        filtered_df = comparison_df[comparison_df['cost_per_1m_tokens'] > 0].copy()
        
        if filtered_df.empty:
            return self._create_empty_figure("No valid cost data for efficiency analysis")
        
        fig = go.Figure()
        
        for i, row in filtered_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['cost_per_1m_tokens']],
                y=[row['avg_accuracy']],
                mode='markers+text',
                marker=dict(
                    size=25,  # Fixed size since no throughput data
                    color=self.color_palette[i % len(self.color_palette)],
                    opacity=0.7
                ),
                text=[self._format_model_name(row['model'])],
                textposition="top center",
                textfont=dict(color='black', size=12),
                name=self._format_model_name(row['model']),
                showlegend=False
            ))
        
        fig.update_layout(
            title='Model Efficiency: Accuracy vs Cost',
            xaxis_title='Cost per 1M Tokens ($)',
            yaxis_title='Average Accuracy',
            height=500,
            plot_bgcolor=self.theme['background_color'],
            xaxis_type="log" if filtered_df['cost_per_1m_tokens'].max() / filtered_df['cost_per_1m_tokens'].min() > 10 else "linear"
        )
        
        return fig
    
    # Model profile visualization removed as per request
    
    def create_benchmark_heatmap(self, performance_df: pd.DataFrame) -> go.Figure:
        """
        Create a heatmap showing performance across models and benchmarks.
        
        Args:
            performance_df: DataFrame with performance metrics
        
        Returns:
            Plotly figure
        """
        if performance_df.empty:
            return self._create_empty_figure("No performance data available")
        
        # Pivot the data for heatmap
        pivot_df = performance_df.pivot_table(
            index='model',
            columns='benchmark',
            values='accuracy',
            aggfunc='mean'
        )
        
        if pivot_df.empty:
            return self._create_empty_figure("No data available for heatmap")
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=[self._format_model_name(name) for name in pivot_df.index],
            colorscale='RdYlBu_r',
            text=np.round(pivot_df.values, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Accuracy")
        ))
        
        fig.update_layout(
            title='Performance Heatmap Across Models and Benchmarks',
            xaxis_title='Benchmark',
            yaxis_title='Model',
            height=400
        )
        
        return fig
    
    def _format_model_name(self, model_name: str) -> str:
        """Format model name for display."""
        name = model_name.replace('meta-llama/', '').replace('microsoft/', '')
        name = name.replace('Llama-3.2-3B-Instruct', 'Llama-3.2-3B')
        name = name.replace('Phi-4-mini-instruct', 'Phi-4-mini')
        return name
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            height=400,
            plot_bgcolor=self.theme['background_color']
        )
        return fig
    
    def create_throughput_analysis(self, throughput_df: pd.DataFrame) -> go.Figure:
        """
        Create a visualization for throughput metrics (tokens per second).
        
        Args:
            throughput_df: DataFrame with throughput metrics
        
        Returns:
            Plotly figure
        """
        if throughput_df.empty:
            return self._create_empty_figure("No throughput data available")
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=[self._format_model_name(name) for name in throughput_df['model']],
                y=throughput_df['tokens_per_second'],
                name='Total Tokens/Second',
                marker_color=self.color_palette[0],
                yaxis='y'
            )
        )
        
        fig.update_layout(
            title='Throughput Analysis: Tokens per Second',
            xaxis_title='Model',
            yaxis_title='Tokens per Second',
            height=500,
            showlegend=False,
            plot_bgcolor=self.theme['background_color']
        )
        
        fig.update_xaxes(tickangle=-45)
        
        return fig
    
    def create_ttft_analysis(self, throughput_df: pd.DataFrame) -> go.Figure:
        """
        Create a visualization for Time to First Token (TTFT) metrics.
        
        Args:
            throughput_df: DataFrame with throughput metrics
        
        Returns:
            Plotly figure
        """
        if throughput_df.empty:
            return self._create_empty_figure("No TTFT data available")
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=[self._format_model_name(name) for name in throughput_df['model']],
                y=throughput_df['estimated_ttft_seconds'],
                name='Estimated TTFT (seconds)',
                marker_color=self.color_palette[2]
            )
        )
        
        fig.update_layout(
            title='Time to First Token Analysis',
            xaxis_title='Model',
            yaxis_title='Time to First Token (seconds)',
            height=500,
            plot_bgcolor=self.theme['background_color']
        )
        
        fig.update_xaxes(tickangle=-45)
        
        return fig
    
    def create_throughput_efficiency_scatter(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Create a scatter plot showing throughput efficiency (accuracy vs tokens per second).
        
        Args:
            comparison_df: DataFrame with model comparison data
        
        Returns:
            Plotly figure
        """
        if comparison_df.empty:
            return self._create_empty_figure("No comparison data available")
        
        # Filter out models with zero throughput
        filtered_df = comparison_df[comparison_df['tokens_per_second'] > 0].copy()
        
        if filtered_df.empty:
            return self._create_empty_figure("No valid throughput data for efficiency analysis")
        
        fig = go.Figure()
        
        for i, row in filtered_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['tokens_per_second']],
                y=[row['avg_accuracy']],
                mode='markers+text',
                marker=dict(
                    size=max(20, row['cost_per_1m_tokens'] * 10),  # Size based on cost
                    color=self.color_palette[i % len(self.color_palette)],
                    opacity=0.7
                ),
                text=[self._format_model_name(row['model'])],
                textposition="top center",
                textfont=dict(color='black', size=12),
                name=self._format_model_name(row['model']),
                showlegend=False,
                hovertemplate=(
                    f"<b>{self._format_model_name(row['model'])}</b><br>"
                    f"Accuracy: {row['avg_accuracy']:.3f}<br>"
                    f"Tokens/sec: {row['tokens_per_second']:.1f}<br>"
                    f"Cost/1M tokens: ${row['cost_per_1m_tokens']:.3f}<br>"
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title='Throughput Efficiency: Accuracy vs Speed',
            xaxis_title='Tokens per Second',
            yaxis_title='Average Accuracy',
            height=500,
            plot_bgcolor=self.theme['background_color']
        )
        
        return fig

class MetricCalculator:
    """Utility class for calculating derived metrics."""
    
    @staticmethod
    def calculate_cost_effectiveness(accuracy: float, cost_per_1m_tokens: float) -> float:
        """Calculate cost effectiveness score."""
        if cost_per_1m_tokens == 0:
            return 0
        return accuracy / cost_per_1m_tokens
    
    @staticmethod
    def calculate_throughput_efficiency(accuracy: float, tokens_per_second: float) -> float:
        """Calculate throughput efficiency score."""
        if tokens_per_second == 0:
            return 0
        return accuracy * tokens_per_second
    
    @staticmethod
    def calculate_comprehensive_efficiency(accuracy: float, cost_per_1m_tokens: float, tokens_per_second: float) -> float:
        """Calculate comprehensive efficiency combining accuracy, cost, and throughput."""
        if cost_per_1m_tokens == 0 or tokens_per_second == 0:
            return 0
        return (accuracy * tokens_per_second) / cost_per_1m_tokens
    
    @staticmethod
    def normalize_metric(values: pd.Series) -> pd.Series:
        """Normalize a metric to 0-1 scale."""
        min_val = values.min()
        max_val = values.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(values), index=values.index)
        return (values - min_val) / (max_val - min_val)
