# LLM Evaluation Dashboard

LLM Evaluation Dashboard is a Python Streamlit web application for visualizing and analyzing Large Language Model evaluation results, including performance metrics, cost analysis, and model comparisons.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

- **Bootstrap, build, and test the repository:**
  - `conda create -n llm_ui python=3.12 -y` -- takes 15 seconds. NEVER CANCEL.
  - `source "$(conda info --base)/etc/profile.d/conda.sh"`
  - `conda activate llm_ui`
  - `pip install --timeout 300 -r requirements.txt` -- takes 30-60 seconds. NEVER CANCEL. May timeout due to network issues.

- **Run the dashboard:**
  - ALWAYS run the bootstrapping steps first.
  - Quick start: `./start_dashboard.sh` -- starts in 10-15 seconds. NEVER CANCEL.
  - Manual start: `conda activate llm_ui && streamlit run app.py`
  - Dashboard will be available at: `http://localhost:8501`

- **Test the application:**
  - Basic imports test: `python -c "from data_loader import EvaluationDataLoader; from visualizations import EvaluationVisualizer; print('✅ All imports successful')"`
  - Syntax check: `python -m py_compile app.py data_loader.py visualizations.py`
  - Dashboard accessibility: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8501` (should return 200)

## Environment Requirements

- **Python**: 3.12 (required for conda environment)
- **Conda**: Required for environment management
- **Dependencies**: All defined in requirements.txt (streamlit, plotly, pandas, numpy, seaborn, matplotlib)
- **Data directory**: `./data/` (optional - user provides evaluation JSON files)

## Validation

- **ALWAYS manually validate any new code** by running through complete user scenarios after making changes.
- **Test complete workflow**: Create sample data → Load data → Start dashboard → Verify HTTP 200 response
- **NEVER CANCEL** any installation or startup commands. Build/install takes 45 seconds total, startup takes 15 seconds.
- **Always test imports** before committing changes: `python -c "from data_loader import EvaluationDataLoader; from visualizations import EvaluationVisualizer"`
- **Dashboard startup must succeed** and return HTTP 200 at localhost:8501

## Key Files and Structure

### Repository root
```
.
├── README.md                    # Complete documentation
├── requirements.txt             # Python dependencies
├── start_dashboard.sh          # Quick start script (executable)
├── app.py                      # Main Streamlit application
├── data_loader.py              # Data loading and processing
├── visualizations.py           # Chart and visualization creation
├── .gitignore                  # Git ignore rules (includes data/)
└── data/                       # User evaluation data (not in repo)
```

### Key Python modules
- **app.py**: Main Streamlit dashboard application entry point
- **data_loader.py**: `EvaluationDataLoader` class for loading JSON evaluation files
- **visualizations.py**: `EvaluationVisualizer` and `MetricCalculator` classes for creating charts

## Data Requirements

The dashboard expects JSON evaluation files in the `data/` directory:
- `advanced_eval_summary.json` - Summary evaluation results
- `*__cost_throughput.json` - Cost and throughput metrics  
- `*__*_details.json` - Detailed evaluation results

**Sample data format for testing:**
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

## Timing Expectations

- **Environment creation**: 15 seconds. NEVER CANCEL. Set timeout to 5+ minutes.
- **Dependency installation**: 30 seconds. NEVER CANCEL. Set timeout to 10+ minutes.
- **Dashboard startup**: 10-15 seconds. NEVER CANCEL. Set timeout to 2+ minutes.
- **Complete bootstrap (first time)**: 45 seconds total. NEVER CANCEL.

## Common Issues and Solutions

- **Conda environment issues**: Always activate with `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate llm_ui`
- **Import errors**: Run `pip install -r requirements.txt` to ensure all dependencies are installed
- **Network timeouts during pip install**: May occur due to firewall limitations. Retry pip install or use `pip install --timeout 300 -r requirements.txt`
- **Dashboard not starting**: Check if port 8501 is available, use different port with `--server.port`
- **No data directory**: Dashboard will show warning but still function. Users can specify data directory in sidebar.

## Development Guidelines

- **No linting tools included**: Repository has no flake8, black, or pytest configured
- **Manual testing required**: Always start dashboard and verify HTTP 200 response after changes
- **Syntax validation**: Use `python -m py_compile` to check syntax before committing
- **Follow existing patterns**: Use pandas for data processing, plotly for visualizations, streamlit for UI

## Testing Scenarios

After making changes, ALWAYS test these scenarios:

1. **Environment setup**: Create fresh conda environment and install dependencies
2. **Basic imports**: Test all module imports work correctly
3. **Dashboard startup**: Start dashboard and verify it serves content
4. **Data loading**: Test with sample evaluation JSON files
5. **HTTP accessibility**: Verify dashboard returns HTTP 200 at localhost:8501

## Emergency Troubleshooting

If environment setup fails:
```bash
# Clean environment and retry with extended timeout
conda env remove -n llm_ui -y
conda create -n llm_ui python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_ui
pip install --timeout 300 --retries 3 -r requirements.txt
```

If dashboard won't start:
```bash
# Check for port conflicts
netstat -tln | grep 8501
# Use different port
streamlit run app.py --server.port 8502
```

## Quick Validation Commands

Use these commands to quickly validate the setup works:

```bash
# Full setup and test (45 seconds total - may take longer with network issues)
conda create -n llm_ui python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_ui
pip install --timeout 300 -r requirements.txt
python -c "from data_loader import EvaluationDataLoader; print('✅ Ready')"
streamlit run app.py --server.headless true &
sleep 15
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501
pkill -f streamlit
```

**Note**: Network timeouts may occur during pip install. If pip fails, retry with extended timeout or check network connectivity.