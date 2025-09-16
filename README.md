# LLM Evaluation Dashboard

A comprehensive dashboard for visualizing and analyzing evaluation results of Large Language Models, including performance metrics, cost analysis, and model comparisons.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.4+-red.svg)
![License](https://img.shields.io/badge/License-Apache2.0-green.svg)

![Home Page of LLM Evals](llmevals.gif)

## 🚀 Features

### 📊 Dashboard Sections

1. **🎯 Performance & TopN Analysis**
   - Model accuracy comparisons across benchmarks
   - TopN performance metrics (Top@1, Top@2, Top@5)
   - Performance heatmaps for model-benchmark combinations
   - Detailed performance progression analysis

2. **💰 Cost Analysis**
   - Cost per token analysis (1K and 1M tokens)
   - Total run cost comparisons
   - Cost efficiency metrics
   - Detailed cost breakdown by model

3. **⚡ Throughput Analysis**
   - Tokens per Second (TPS) measurements
   - Time to First Token (TTFT) analysis  
   - Output token generation speed metrics
   - Throughput efficiency scatter plots (accuracy vs speed)
   - Detailed throughput data tables

4. **🔍 Model Comparison**
   - Efficiency scatter plots (accuracy vs cost)
   - Automated model rankings with efficiency scores

5. **📊 Advanced Analytics**
   - Multi-dimensional efficiency scoring (accuracy + cost + throughput)
   - Data export functionality (CSV download)
   - Interactive visualizations with tooltips
   - Comprehensive metrics tables

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- Conda (recommended) or virtualenv

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/debabratamishra/llm-evals
   cd llm-evals
   ```

2. **Create and activate a conda environment:**
   ```bash
   conda create -n llm_ui python=3.12
   conda activate llm_ui
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Dashboard

#### Option 1: Using the startup script (Recommended)
```bash
chmod +x start_dashboard.sh
./start_dashboard.sh
```

#### Option 2: Manual startup
```bash
conda activate llm_ui
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### 📁 Data Structure

The dashboard automatically loads evaluation data from the `data/` directory. The following file formats are supported:

- `advanced_eval_summary.json` - Summary evaluation files
- `*__*_details.json` - Detailed evaluation results  
- `*__cost_throughput.json` - Cost and throughput metrics

#### Expected Data Format

**Performance Metrics:**
```json
{
  "model_name": {
    "benchmark_name": {
      "n": 100,
      "acc": 0.75,
      "top@1": 0.75,
      "top@2": 0.85,
      "top@5": 0.95
    }
  }
}
```

**Cost & Throughput Metrics:**
```json
{
  "model_name": {
    "cost_throughput": {
      "cost": {
        "run_cost_usd": 0.01,
        "cost_per_1k_tokens_usd": 0.0001,
        "cost_per_1m_tokens_usd": 0.1
      },
      "mode": "api",
      "elapsed_seconds": 45.2,
      "total_tokens": 25000,
      "input_tokens": 15000,
      "output_tokens": 10000
    }
  }
}
    }
  }
}
```

## 🎮 Dashboard Navigation

### Sidebar Controls
- **Data Directory**: Configure the path to evaluation data (defaults to `./data`)
- Real-time data loading with progress indicators

### Main Tabs

1. **🎯 Performance & TopN Analysis**
   - Select performance metrics to visualize
   - Compare TopN accuracy across models
   - View performance heatmaps

2. **💰 Cost Analysis**
   - Analyze cost per token metrics
   - Compare total run costs
   - Identify cost-effective models

3. **⚡ Throughput Analysis**
   - Tokens per Second (TPS) performance metrics
   - Time to First Token (TTFT) measurements
   - Throughput efficiency analysis (accuracy vs speed)
   - Detailed throughput data tables

4. **🔍 Model Comparison**
   - Efficiency analysis (accuracy vs cost)
   - Customizable ranking systems

5. **📊 Advanced Analytics**
   - Multi-dimensional efficiency calculations (accuracy + cost + throughput)
   - Data export functionality
   - Interactive metric visualization

## 📊 Visualization Features

- **Interactive Charts**: Built with Plotly for responsive visualization
- **Custom Tooltips**: Detailed explanations for all metrics
- **Export Options**: Download charts and data as CSV/PNG
- **Responsive Design**: Optimized for different screen sizes
- **Real-time Updates**: Data refreshes automatically when changed

## ⚡ Throughput Metrics Explained

### Key Metrics
- **Tokens per Second (TPS)**: Total throughput including both input and output processing
- **Output Tokens per Second**: Generation speed for output tokens only
- **Time to First Token (TTFT)**: Latency measurement for initial response (estimated)
- **Throughput Efficiency**: Composite metric combining accuracy and speed performance

### Use Cases
- **Latency Optimization**: Use TTFT metrics for real-time applications
- **Throughput Planning**: Use TPS metrics for batch processing scenarios  
- **Balanced Selection**: Use efficiency metrics for optimal accuracy-speed trade-offs
- **Cost-Performance Analysis**: Combined with cost metrics for comprehensive evaluation

## 🔧 Configuration

### Custom Data Directory
You can specify a different data directory using the sidebar input or by modifying the default path in `app.py`:

```python
default_data_dir = os.path.join(dashboard_dir, "your_data_directory")
```

### Adding New Metrics

1. **Extend the data loader** (`data_loader.py`):
   ```python
   def _extract_new_metrics(self) -> pd.DataFrame:
       # Add your metric extraction logic here
       pass
   ```

2. **Create visualizations** (`visualizations.py`):
   ```python
   def create_new_chart(self, data: pd.DataFrame) -> go.Figure:
       # Add your visualization logic here
       pass
   ```

3. **Update the dashboard** (`app.py`):
   ```python
   # Add new tabs or sections
   with st.tab("New Analysis"):
       new_fig = visualizer.create_new_chart(data)
       st.plotly_chart(new_fig)
   ```

## 🏗️ Architecture

### Component Overview

```
llm-evals/
├── app.py                 # Main Streamlit application
├── data_loader.py         # Data loading and processing
├── visualizations.py      # Chart and visualization creation
├── requirements.txt       # Python dependencies
├── start_dashboard.sh     # Quick start script
└── data/                  # Evaluation data directory
    ├── advanced_eval_summary.json
    ├── *__cost_throughput.json
    └── *__*_details.json
```

### Data Flow

1. **Load** → JSON files are loaded from the `data/` directory
2. **Process** → Data is parsed and structured into pandas DataFrames  
3. **Visualize** → Charts are created using Plotly
4. **Interact** → Users can filter, compare, and export data

## 🧪 Testing

To verify the installation and data loading:

```bash
conda activate llm_ui
python -c "
from data_loader import EvaluationDataLoader
loader = EvaluationDataLoader('./data')
data = loader.load_all_data()
print(f'Loaded {len(data)} data sections')
print('Available sections:', list(data.keys()))
"
```

## 🐛 Troubleshooting

### Common Issues

#### Data Not Loading
- **Check file format**: Ensure JSON files are properly formatted
- **Verify data directory**: Confirm the `data/` directory exists and contains files
- **File permissions**: Ensure read permissions on data files

#### Import Errors
```bash
# Reinstall dependencies
conda activate llm_ui
pip install --upgrade -r requirements.txt
```

#### Performance Issues
- **Large datasets**: Consider filtering data for better performance
- **Memory usage**: Monitor system resources with large evaluation datasets
- **Browser cache**: Clear browser cache if visualizations aren't updating

#### Visualization Problems
- **Missing data**: Check console logs for data processing errors
- **Chart rendering**: Ensure browser supports modern JavaScript features
- **Interactive features**: Verify Plotly.js is loading correctly

### Debug Mode

Run the dashboard in debug mode:
```bash
streamlit run app.py --logger.level=debug
```

## 📈 Performance Optimization

- **Data Caching**: Streamlit automatically caches loaded data
- **Efficient Processing**: Use pandas vectorized operations
- **Memory Management**: Process data in chunks for large datasets
- **Visualization**: Limit data points for complex charts

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes** with proper documentation
4. **Test thoroughly** with sample data
5. **Submit a pull request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include type hints where appropriate
- Test with various data formats
- Update documentation for new features

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the web application framework
- **Plotly** for interactive visualizations
- **Pandas** for data processing capabilities

## 📧 Support

For issues, questions, or contributions:
- Create an issue in the repository
- Check existing documentation
- Review troubleshooting section

---

**Built with ❤️ for the LLM evaluation community**
