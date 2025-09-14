"""
Data loading and processing module for LLM evaluation dashboard.
"""

import json
import pandas as pd
import os
from typing import Dict, List, Any, Optional
import glob

class EvaluationDataLoader:
    """Class to load and process LLM evaluation data from JSON files."""
    
    def __init__(self, data_directory: str):
        """
        Initialize the data loader with the directory containing evaluation files.
        
        Args:
            data_directory: Path to directory containing evaluation JSON files
        """
        self.data_directory = data_directory
        self.data = {}
        self.summary_data = {}
        self.detailed_data = {}
        
    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all evaluation data from the directory.
        
        Returns:
            Dictionary containing all loaded and processed data
        """
        # Load summary files
        self._load_summary_files()
        
        # Load detailed files
        self._load_detailed_files()
        
        # Process and combine data
        self._process_data()
        
        return self.data
    
    def _load_summary_files(self):
        """Load summary evaluation files."""
        summary_files = [
            'advanced_eval_summary.json',
            'gpt4o_eval_summary.json'
        ]
        
        for file in summary_files:
            file_path = os.path.join(self.data_directory, file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.summary_data[file.replace('.json', '')] = data
    
    def _load_detailed_files(self):
        """Load detailed evaluation files."""
        # Find all detailed evaluation files
        pattern = os.path.join(self.data_directory, '*__*.json')
        detailed_files = glob.glob(pattern)
        
        for file_path in detailed_files:
            filename = os.path.basename(file_path)
            if filename.endswith('_details.json') or filename.endswith('_cost_throughput.json'):
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        self.detailed_data[filename.replace('.json', '')] = data
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {filename}")
    
    def _process_data(self):
        """Process and structure the loaded data for visualization."""
        # Process performance metrics
        self.data['performance_metrics'] = self._extract_performance_metrics()
        
        # Process cost metrics
        self.data['cost_metrics'] = self._extract_cost_metrics()
        
        # Process model comparison data
        self.data['model_comparison'] = self._create_model_comparison()
    
    def _extract_performance_metrics(self) -> pd.DataFrame:
        """Extract performance metrics into a structured DataFrame."""
        performance_data = []
        
        for summary_key, summary in self.summary_data.items():
            for model_name, model_data in summary.items():
                for benchmark, metrics in model_data.items():
                    if benchmark in ['mmlu_clinical', 'medmcqa_val']:
                        acc = metrics.get('acc', 0)
                        n = metrics.get('n', 0)
                        
                        row = {
                            'model': model_name,
                            'benchmark': benchmark,
                            'accuracy': acc,
                            'n_samples': n,
                            'top1': metrics.get('top@1', 0),
                            'top2': metrics.get('top@2', 0),
                            'top5': metrics.get('top@5', 0),
                            'source': summary_key
                        }
                        performance_data.append(row)
        
        return pd.DataFrame(performance_data)
    
    def _extract_cost_metrics(self) -> pd.DataFrame:
        """Extract cost metrics into a structured DataFrame."""
        cost_data = []
        
        for summary_key, summary in self.summary_data.items():
            for model_name, model_data in summary.items():
                cost_throughput = model_data.get('cost_throughput', {})
                cost_info = cost_throughput.get('cost', {})
                
                if cost_info:
                    row = {
                        'model': model_name,
                        'run_cost_usd': cost_info.get('run_cost_usd', 0),
                        'cost_per_1k_tokens_usd': cost_info.get('cost_per_1k_tokens_usd', 0),
                        'cost_per_1m_tokens_usd': cost_info.get('cost_per_1m_tokens_usd', 0),
                        'price_per_1k_input_usd': cost_info.get('price_per_1k_input_usd', 0),
                        'price_per_1k_output_usd': cost_info.get('price_per_1k_output_usd', 0),
                        'mode': cost_throughput.get('mode', 'unknown'),
                        'elapsed_seconds': cost_throughput.get('elapsed_seconds', 0),
                        'source': summary_key
                    }
                    cost_data.append(row)
        
        return pd.DataFrame(cost_data)
    
    def _create_model_comparison(self) -> pd.DataFrame:
        """Create a comprehensive model comparison DataFrame."""
        comparison_data = []
        
        for summary_key, summary in self.summary_data.items():
            for model_name, model_data in summary.items():
                # Get average performance across benchmarks
                benchmarks = [k for k in model_data.keys() if k in ['mmlu_clinical', 'medmcqa_val']]
                avg_accuracy = 0
                total_samples = 0
                
                for benchmark in benchmarks:
                    if benchmark in model_data:
                        acc = model_data[benchmark].get('acc', 0)
                        n = model_data[benchmark].get('n', 0)
                        avg_accuracy += acc * n
                        total_samples += n
                
                avg_accuracy = avg_accuracy / total_samples if total_samples > 0 else 0
                
                # Get cost data
                cost_throughput = model_data.get('cost_throughput', {})
                cost_info = cost_throughput.get('cost', {})
                
                row = {
                    'model': model_name,
                    'avg_accuracy': avg_accuracy,
                    'total_samples': total_samples,
                    'cost_per_1m_tokens': cost_info.get('cost_per_1m_tokens_usd', 0),
                    'run_cost': cost_info.get('run_cost_usd', 0),
                    'mode': cost_throughput.get('mode', 'unknown'),
                    'source': summary_key
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_model_names(self) -> List[str]:
        """Get list of all model names in the data."""
        models = set()
        for summary in self.summary_data.values():
            models.update(summary.keys())
        return sorted(list(models))
    
    def get_benchmarks(self) -> List[str]:
        """Get list of all benchmarks in the data."""
        benchmarks = set()
        for summary in self.summary_data.values():
            for model_data in summary.values():
                for key in model_data.keys():
                    if key in ['mmlu_clinical', 'medmcqa_val']:
                        benchmarks.add(key)
        return sorted(list(benchmarks))

# Utility functions for data formatting
def format_model_name(model_name: str) -> str:
    """Format model name for display."""
    # Remove organization prefixes and shorten names
    name = model_name.replace('meta-llama/', '').replace('microsoft/', '')
    name = name.replace('Llama-3.2-3B-Instruct', 'Llama-3.2-3B')
    name = name.replace('Phi-4-mini-instruct', 'Phi-4-mini')
    return name

def calculate_efficiency_score(accuracy: float, cost_per_1m_tokens: float) -> float:
    """
    Calculate a composite efficiency score based on accuracy and cost.
    
    Args:
        accuracy: Model accuracy (0-1)
        cost_per_1m_tokens: Cost per million tokens in USD
    
    Returns:
        Efficiency score (higher is better)
    """
    if cost_per_1m_tokens == 0:
        return accuracy
    
    # Simple efficiency: accuracy per dollar spent
    return accuracy / cost_per_1m_tokens