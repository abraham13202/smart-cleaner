# Smart Cleaner

**AI-powered data cleaning platform with fully automatic preprocessing - runs 100% locally with Ollama!**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Smart Cleaner is a comprehensive Python library that handles **everything** in your data cleaning pipeline automatically using **local AI** (Ollama). No API costs, no internet required, complete privacy.

**One command. Complete preprocessing. Full documentation. Ready for ML.**

```bash
python universal_data_pipeline.py your_data.csv
```

### What You Get:
- `your_data_cleaned.csv` - Cleaned dataset
- `your_data_documentation.Rmd` - Comprehensive technical report for data scientists
- `data_visualizations/` - 9 professional visualizations

## Features

- **100% Local AI** - Uses Ollama (free, private, no API costs)
- **Intelligent Imputation** - AI analyzes each column and recommends optimal strategy
- **Auto Data Type Detection** - Mode for text, mean/median for numeric
- **Comprehensive Documentation** - RMD report with everything data scientists need
- **9 Visualizations** - Box plots, correlations, distributions, and more
- **Healthcare Support** - Built-in validation for health metrics

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows - Download from https://ollama.com
```

### 3. Pull a Model

```bash
ollama pull llama3.2
```

### 4. Run the Pipeline

```bash
python universal_data_pipeline.py your_data.csv
```

That's it! Your cleaned data and documentation will be generated automatically.

## Command Line Usage

```bash
# Basic usage
python universal_data_pipeline.py data.csv

# With target column
python universal_data_pipeline.py data.csv target_column

# Use different Ollama model
python universal_data_pipeline.py data.csv --model mistral

# Custom output file
python universal_data_pipeline.py data.csv -o cleaned_output.csv

# Skip AI (use simple strategies)
python universal_data_pipeline.py data.csv --no-ai

# Skip visualizations
python universal_data_pipeline.py data.csv --no-visualize
```

## Output Files

| File | Description |
|------|-------------|
| `{dataset}_cleaned.csv` | Cleaned dataset ready for ML |
| `{dataset}_documentation.Rmd` | Comprehensive technical report |
| `data_visualizations/*.png` | 9 visualization files |

### Generated Visualizations

1. `01_missing_values.png` - Missing values by column
2. `02_correlation_matrix.png` - Feature correlation heatmap
3. `03_boxplots_outliers.png` - Box plots for outlier detection
4. `04_distributions.png` - Histograms with KDE
5. `05_categorical_distributions.png` - Category frequencies
6. `06_pairplot.png` - Pairwise relationships
7. `07_target_analysis.png` - Target variable analysis
8. `08_data_quality_summary.png` - Data quality overview
9. `09_distribution_stats.png` - Skewness and kurtosis

### RMD Documentation Includes

- Executive Summary (AI-generated)
- Dataset Overview & Data Dictionary
- Data Quality Assessment
- Methodology Explanation
- All Imputation Strategies with Reasoning
- Outlier Treatment Details
- Before/After Comparisons
- Recommendations for Data Scientists
- R and Python Code to Load Data

## Python API Usage

```python
from smart_cleaner import AutoPreprocessor, PipelineConfig
import pandas as pd

# Load your dataset
df = pd.read_csv("data.csv")

# Configure with Ollama (local AI - FREE!)
config = PipelineConfig(
    use_ai_recommendations=True,
    ai_provider="ollama",
    ollama_model="llama3.2",
    target_column="target",
    generate_visualizations=True,
)

# Run complete preprocessing
preprocessor = AutoPreprocessor(config)
cleaned_df, report = preprocessor.process(df)

# Your data is now:
# - Free of duplicates
# - Outliers handled
# - Missing values intelligently imputed
# - Ready for ML
```

## AI Providers

| Provider | Cost | Privacy | Setup |
|----------|------|---------|-------|
| **Ollama** (default) | Free | 100% Local | `ollama pull llama3.2` |
| Gemini | Free tier | Cloud | API key required |
| Claude | Paid | Cloud | API key required |

### Using Different Providers

```python
# Ollama (default - recommended)
config = PipelineConfig(
    ai_provider="ollama",
    ollama_model="llama3.2",
)

# Gemini (free tier available)
config = PipelineConfig(
    ai_provider="gemini",
    gemini_api_key="your-key",
)

# Claude
config = PipelineConfig(
    ai_provider="claude",
    anthropic_api_key="your-key",
)
```

## How It Works

### Imputation Strategy Selection

The AI analyzes each column and selects the best strategy:

| Data Type | Strategy | Why |
|-----------|----------|-----|
| Text/Object | Mode | Most frequent value preserves categorical nature |
| Numeric (normal) | Mean | Best for normally distributed data |
| Numeric (skewed) | Median | Robust to outliers |
| High correlation | KNN/Cohort | Uses related columns for better accuracy |

### Outlier Handling

- **Detection**: IQR method (Q1 - 1.5×IQR to Q3 + 1.5×IQR)
- **Treatment**: Capping (preserves data points, limits extreme values)

## Configuration Options

```python
config = PipelineConfig(
    # AI Settings
    use_ai_recommendations=True,
    ai_provider="ollama",           # "ollama", "gemini", "claude"
    ollama_model="llama3.2",        # Any Ollama model

    # Processing
    remove_duplicates=True,
    handle_outliers=True,
    impute_missing=True,

    # Output
    generate_visualizations=True,
    visualization_output_dir="./data_visualizations",

    # Target (optional)
    target_column="your_target",
)
```

## Project Structure

```
smart_cleaner/
├── core/
│   ├── ai_advisor_ollama.py     # Ollama AI integration
│   ├── ai_advisor_gemini.py     # Gemini AI integration
│   ├── ai_advisor.py            # Claude AI integration
│   ├── auto_pipeline.py         # Full automatic pipeline
│   ├── technical_documenter.py  # RMD documentation generator
│   ├── imputation.py            # Imputation strategies
│   ├── outliers.py              # Outlier detection/handling
│   ├── visualizer.py            # Visualization generation
│   └── eda.py                   # EDA utilities
├── utils/
│   ├── config.py                # Configuration
│   └── validators.py            # Data validators
├── examples/
│   ├── basic_usage.py           # Basic example
│   └── cohort_imputation_demo.py
└── tests/
    └── test_imputation.py
```

## Requirements

- Python 3.8+
- Ollama (for local AI)
- pandas, numpy, scikit-learn
- matplotlib, seaborn (for visualizations)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-cleaner.git
cd smart-cleaner

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull a model
ollama pull llama3.2
```

## Examples

```bash
# Run basic example
python examples/basic_usage.py

# Run cohort imputation demo
python examples/cohort_imputation_demo.py
```

## Environment Variables (Optional)

```bash
export OLLAMA_MODEL="llama3.2"
export OLLAMA_BASE_URL="http://localhost:11434"  # Custom Ollama server
export GEMINI_API_KEY="your-key"  # For Gemini
export ANTHROPIC_API_KEY="your-key"  # For Claude
```

## Testing

```bash
pytest tests/ -v
```

## FAQ

**Q: Do I need an API key?**
A: No! Ollama runs 100% locally and is free. API keys are only needed for cloud providers (Gemini, Claude).

**Q: Which Ollama model should I use?**
A: `llama3.2` is recommended for best results. `mistral` is faster but slightly less accurate.

**Q: Can I use this without AI?**
A: Yes! Use `--no-ai` flag or set `use_ai_recommendations=False` for simple mean/median/mode imputation.

**Q: How do I render the RMD file?**
A: Open in RStudio and click "Knit", or run: `rmarkdown::render("file.Rmd")`

**Q: Is my data sent to the cloud?**
A: With Ollama (default), NO - everything runs locally. With Gemini/Claude, statistics are sent to their APIs.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{smart_cleaner,
  title = {Smart Cleaner: AI-Powered Data Cleaning with Local LLM},
  year = {2024},
  url = {https://github.com/yourusername/smart-cleaner}
}
```

---

**Transform messy data into ML-ready datasets - automatically, locally, and free!**
