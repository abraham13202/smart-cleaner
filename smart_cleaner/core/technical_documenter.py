"""
Technical Documentation Generator using Ollama.

Generates comprehensive RMD documentation for data scientists,
documenting every step of the data cleaning pipeline.
The AI acts as the world's best technical writer.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

try:
    import ollama
except ImportError:
    ollama = None


class TechnicalDocumenter:
    """
    AI-powered technical documentation generator using Ollama.
    Acts as the world's best technical writer to document data cleaning pipelines.
    Generates comprehensive RMD files for data scientists.
    """

    def __init__(self, model: str = "llama3.2", base_url: Optional[str] = None):
        """
        Initialize the technical documenter.

        Args:
            model: Ollama model to use
            base_url: Optional custom Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.documentation_sections = []
        self.start_time = None
        self.dataset_name = "dataset"
        self.initial_state = {}
        self.step_details = []

        if ollama is None:
            print("  Note: ollama package not installed. Using template-based documentation.")
            self.client = None
        else:
            self.client = ollama.Client(host=base_url) if base_url else ollama.Client()

    def start_documentation(self, df: pd.DataFrame, dataset_name: str = "dataset"):
        """Start documenting a new pipeline run."""
        self.start_time = datetime.now()
        self.dataset_name = dataset_name
        self.documentation_sections = []
        self.step_details = []

        # Record initial dataset state
        self._record_initial_state(df)

    def _record_initial_state(self, df: pd.DataFrame):
        """Record the initial state of the dataset."""
        self.initial_state = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "total_missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "numeric_stats": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        }

    def document_step(
        self,
        step_name: str,
        description: str,
        details: Dict[str, Any],
        df_before: Optional[pd.DataFrame] = None,
        df_after: Optional[pd.DataFrame] = None,
    ):
        """Document a pipeline step."""
        step_info = {
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "details": details,
        }

        if df_before is not None and df_after is not None:
            step_info["changes"] = {
                "rows_before": len(df_before),
                "rows_after": len(df_after),
                "rows_removed": len(df_before) - len(df_after),
                "missing_before": int(df_before.isnull().sum().sum()),
                "missing_after": int(df_after.isnull().sum().sum()),
            }

        self.step_details.append(step_info)

    def _get_ai_summary(self, prompt: str, max_length: int = 500) -> str:
        """Get AI-generated summary from Ollama."""
        if self.client is None:
            return ""

        try:
            system_prompt = """You are the world's best technical writer specializing in data science documentation.
Your writing is clear, precise, comprehensive, and professional.
You explain complex data transformations in a way that any data scientist can understand.
You use proper technical terminology and provide actionable insights.
Be thorough but concise. Focus on what matters for reproducibility and understanding.
Do not use markdown headers (no # symbols) in your response - just write paragraphs."""

            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.3}
            )
            return response["message"]["content"]
        except Exception as e:
            return ""

    def generate_rmd(
        self,
        df_final: pd.DataFrame,
        output_path: str,
        report: Dict[str, Any],
    ) -> str:
        """
        Generate comprehensive RMD documentation.

        Args:
            df_final: Final cleaned DataFrame
            output_path: Path to save the RMD file
            report: Pipeline report with all step details

        Returns:
            Path to the generated RMD file
        """
        print("  Generating comprehensive RMD documentation with AI...")

        rmd_content = self._build_rmd_content(df_final, report)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rmd_content)

        print(f"  Documentation saved to: {output_path}")
        return output_path

    def _build_rmd_content(self, df_final: pd.DataFrame, report: Dict[str, Any]) -> str:
        """Build the complete RMD content."""

        print("    Generating executive summary...")
        exec_summary = self._generate_executive_summary(report)

        print("    Generating methodology section...")
        methodology = self._generate_methodology_section(report)

        print("    Generating recommendations...")
        recommendations = self._generate_recommendations(df_final, report)

        print("    Generating data dictionary...")
        data_dictionary = self._generate_data_dictionary(df_final)

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0

        quality_improvement = report.get('data_quality_improvement', {})

        rmd = f'''---
title: "Data Cleaning Pipeline - Technical Report"
subtitle: "Dataset: {self.dataset_name}"
author: "AI Data Analyst (Ollama - {self.model})"
date: "`r format(Sys.time(), '%B %d, %Y at %H:%M')`"
output:
  html_document:
    toc: true
    toc_depth: 4
    toc_float:
      collapsed: false
      smooth_scroll: true
    theme: flatly
    highlight: tango
    code_folding: hide
    df_print: paged
    number_sections: true
---

```{{r setup, include=FALSE}}
knitr::opts_chunk$set(
  echo = TRUE,
  warning = FALSE,
  message = FALSE,
  fig.width = 10,
  fig.height = 6
)
```

<style>
.main-container {{
  max-width: 1400px;
  margin: auto;
}}
h1, h2, h3 {{
  color: #2c3e50;
}}
.alert-info {{
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
  padding: 15px;
  border-radius: 4px;
  margin: 20px 0;
}}
.alert-success {{
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
  padding: 15px;
  border-radius: 4px;
  margin: 20px 0;
}}
table {{
  width: 100%;
  border-collapse: collapse;
}}
th, td {{
  padding: 8px 12px;
  border: 1px solid #ddd;
}}
th {{
  background-color: #f5f5f5;
}}
</style>

---

# Executive Summary {{.tabset}}

<div class="alert-info">
**Pipeline Processing Time:** {duration:.1f} seconds | **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **AI Model:** Ollama ({self.model})
</div>

{exec_summary}

## Key Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total Rows** | {self.initial_state['shape'][0]:,} | {len(df_final):,} | {self.initial_state['shape'][0] - len(df_final):,} removed |
| **Total Columns** | {self.initial_state['shape'][1]} | {len(df_final.columns)} | - |
| **Missing Values** | {self.initial_state['total_missing']:,} | {df_final.isnull().sum().sum():,} | {self.initial_state['total_missing'] - df_final.isnull().sum().sum():,} imputed |
| **Completeness** | {quality_improvement.get('completeness_before', 0):.1f}% | {quality_improvement.get('completeness_after', 0):.1f}% | +{quality_improvement.get('completeness_improvement', 0):.1f}% |
| **Duplicate Rows** | {self.initial_state['duplicates']:,} | 0 | {self.initial_state['duplicates']:,} removed |

---

# Dataset Overview {{.tabset}}

## Original Dataset Characteristics

The original dataset **{self.dataset_name}** contained:

- **{self.initial_state['shape'][0]:,}** rows (observations)
- **{self.initial_state['shape'][1]}** columns (features)
- **{len(self.initial_state['numeric_columns'])}** numeric columns
- **{len(self.initial_state['categorical_columns'])}** categorical columns
- **{self.initial_state['memory_mb']:.2f} MB** memory usage

## Column Types

### Numeric Columns ({len(self.initial_state['numeric_columns'])})

{self._format_column_list(self.initial_state['numeric_columns'])}

### Categorical Columns ({len(self.initial_state['categorical_columns'])})

{self._format_column_list(self.initial_state['categorical_columns'])}

## Data Types Summary

```
{self._format_dtypes_summary()}
```

---

# Data Quality Assessment {{.tabset}}

## Missing Values Analysis

{self._generate_missing_values_analysis()}

## Initial Quality Score

{self._generate_quality_metrics(report)}

---

# Methodology {{.tabset}}

{methodology}

---

# Pipeline Steps - Detailed Documentation {{.tabset}}

{self._generate_detailed_steps(report)}

---

# Imputation Strategies Applied {{.tabset}}

{self._generate_imputation_details(report)}

---

# Outlier Treatment {{.tabset}}

{self._generate_outlier_details(report)}

---

# Final Dataset Summary {{.tabset}}

## Comparison: Before vs After

{self._generate_comparison_table(df_final, report)}

## Final Dataset Statistics

{self._generate_final_statistics(df_final)}

---

# Data Dictionary {{.tabset}}

{data_dictionary}

---

# Visualizations Reference {{.tabset}}

The following visualizations have been generated and saved to `./data_visualizations/`:

| # | File | Description | Use Case for Data Scientist |
|---|------|-------------|----------------------------|
| 1 | `01_missing_values.png` | Missing values by column (color-coded by severity) | Identify data completeness issues before modeling |
| 2 | `02_correlation_matrix.png` | Feature correlation heatmap | Identify multicollinearity, feature selection |
| 3 | `03_boxplots_outliers.png` | Box plots for numeric columns | Outlier detection and distribution shape |
| 4 | `04_distributions.png` | Histograms with KDE density curves | Understand data distributions, normality |
| 5 | `05_categorical_distributions.png` | Category frequencies | Understand categorical variables, class balance |
| 6 | `06_pairplot.png` | Pairwise feature relationships | Feature interactions, clusters |
| 7 | `07_target_analysis.png` | Target variable analysis | Target distribution and feature correlations |
| 8 | `08_data_quality_summary.png` | Data quality overview | Quick quality assessment |
| 9 | `09_distribution_stats.png` | Skewness and kurtosis | Distribution characteristics, transformation needs |

---

# Recommendations for Data Scientists {{.tabset}}

{recommendations}

---

# Reproducibility Information {{.tabset}}

## Environment Details

| Component | Value |
|-----------|-------|
| **Documentation Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| **AI Model Used** | Ollama ({self.model}) |
| **Processing Time** | {duration:.1f} seconds |
| **Original Dataset** | {self.dataset_name} |

## Output Files

| File | Description |
|------|-------------|
| `{self.dataset_name}_cleaned.csv` | Cleaned dataset ready for analysis |
| `data_visualizations/*.png` | All generated visualizations (9 files) |
| `{self.dataset_name}_documentation.Rmd` | This documentation file |

## Transformation Log

```
{self._generate_transformation_log(report)}
```

---

# Appendix A: Complete Column Reference {{.tabset}}

{self._generate_complete_column_reference(df_final)}

---

# Appendix B: Code to Load Data {{.tabset}}

## R Code

```{{r load_data, eval=FALSE}}
# Load the cleaned dataset in R
library(tidyverse)

# Read the cleaned data
df_cleaned <- read_csv("{self.dataset_name}_cleaned.csv")

# Quick overview
glimpse(df_cleaned)

# Summary statistics
summary(df_cleaned)

# Check for any remaining missing values
colSums(is.na(df_cleaned))
```

## Python Code

```python
# Load the cleaned dataset in Python
import pandas as pd
import numpy as np

# Read the cleaned data
df_cleaned = pd.read_csv("{self.dataset_name}_cleaned.csv")

# Quick overview
print(df_cleaned.info())
print("\\n" + "="*50 + "\\n")
print(df_cleaned.describe())

# Check for any remaining missing values
print("\\nMissing values per column:")
print(df_cleaned.isnull().sum())
```

---

# Appendix C: Glossary {{.tabset}}

| Term | Definition |
|------|------------|
| **Imputation** | The process of replacing missing values with substituted values |
| **Mean Imputation** | Replacing missing values with the mean of the non-missing values |
| **Median Imputation** | Replacing missing values with the median (less sensitive to outliers) |
| **Mode Imputation** | Replacing missing values with the most frequent value (for categorical data) |
| **IQR (Interquartile Range)** | The range between the 25th and 75th percentile, used for outlier detection |
| **Outlier** | A data point significantly different from other observations |
| **Capping** | Replacing outlier values with boundary values (Q1-1.5×IQR or Q3+1.5×IQR) |
| **Completeness** | The percentage of non-missing values in the dataset |

---

<div class="alert-success">
**Documentation Complete**

This report was automatically generated by the AI-powered Data Cleaning Pipeline. All transformations are documented for reproducibility and audit purposes. The cleaned dataset is ready for exploratory data analysis and machine learning model development.

For questions about this documentation or the cleaning process, please refer to the methodology section or contact the data engineering team.
</div>

---

*Generated by AI Technical Writer (Ollama {self.model}) | Smart Cleaner Pipeline*
'''
        return rmd

    def _generate_executive_summary(self, report: Dict[str, Any]) -> str:
        """Generate AI-powered executive summary."""
        quality_improvement = report.get('data_quality_improvement', {})

        prompt = f"""Write a concise executive summary (3-4 paragraphs) for a data cleaning technical report.

Dataset: {self.dataset_name}
Initial state:
- {self.initial_state['shape'][0]:,} rows, {self.initial_state['shape'][1]} columns
- {self.initial_state['total_missing']:,} missing values total
- {self.initial_state['duplicates']:,} duplicate rows
- {len(self.initial_state['numeric_columns'])} numeric columns
- {len(self.initial_state['categorical_columns'])} categorical columns

Results:
- Completeness improved from {quality_improvement.get('completeness_before', 0):.1f}% to {quality_improvement.get('completeness_after', 0):.1f}%
- {report.get('rows_removed', 0):,} rows removed
- Missing values imputed: {quality_improvement.get('missing_values_before', 0) - quality_improvement.get('missing_values_after', 0):,}

Write for a data scientist audience. Explain:
1. Initial data quality state
2. Key issues found and how they were addressed
3. Final data quality state
4. Readiness for machine learning

Do not use markdown headers."""

        summary = self._get_ai_summary(prompt)

        if not summary:
            summary = f"""This technical report documents the comprehensive data cleaning pipeline applied to the **{self.dataset_name}** dataset. The original dataset contained {self.initial_state['shape'][0]:,} observations across {self.initial_state['shape'][1]} features, with {self.initial_state['total_missing']:,} missing values representing approximately {(self.initial_state['total_missing']/(self.initial_state['shape'][0]*self.initial_state['shape'][1])*100):.1f}% of all data cells.

The automated AI-powered pipeline identified and systematically addressed multiple data quality issues. Missing values were handled using intelligent imputation strategies selected by the AI based on each column's data type, distribution characteristics, and correlation patterns. Categorical columns received mode imputation while numeric columns were imputed using mean or median based on distribution analysis. Outliers were detected using the IQR method and capped at boundary values to preserve data points while limiting extreme value influence.

Following the cleaning process, data completeness improved from {quality_improvement.get('completeness_before', 0):.1f}% to {quality_improvement.get('completeness_after', 0):.1f}%, representing a significant enhancement in data quality. The final dataset is now suitable for exploratory data analysis and machine learning model development, with all transformations fully documented for reproducibility."""

        return summary

    def _generate_methodology_section(self, report: Dict[str, Any]) -> str:
        """Generate AI-powered methodology description."""
        prompt = f"""Write a detailed methodology section (4-5 paragraphs) explaining the data cleaning approach.

The pipeline uses:
1. AI-powered imputation using local LLM (Ollama) - analyzes each column individually
2. Automatic data type detection - distinguishes numeric vs categorical
3. For text/object columns: mode imputation (most frequent value)
4. For numeric columns: mean or median based on distribution
5. Outlier detection using IQR method with 1.5×IQR threshold
6. Outlier treatment using capping (not removal) to preserve data points
7. Duplicate detection and removal

Explain:
- Why AI-powered decision making improves imputation quality
- How data type affects strategy selection
- Why IQR with capping is preferred over removal
- How this ensures data integrity

Write professionally for a technical audience. Do not use markdown headers."""

        methodology = self._get_ai_summary(prompt)

        if not methodology:
            methodology = """## Automated AI-Powered Approach

The data cleaning pipeline employs a sophisticated multi-stage automated approach that combines rule-based processing with AI-powered decision making. Each column containing missing values is analyzed individually by the local language model (Ollama), which considers the data type, statistical distribution, correlation patterns with other features, and the percentage of missing values to recommend the most appropriate imputation strategy.

## Imputation Strategy Selection

The fundamental principle guiding strategy selection is data type awareness. For **categorical columns** (object, string, category types), mode imputation is applied, replacing missing values with the most frequently occurring value. This approach maintains the categorical nature of the data and ensures imputed values are valid category members. For **numeric columns** (int64, float64), the AI evaluates distribution characteristics including skewness, kurtosis, and the presence of outliers to choose between mean imputation (for approximately normal distributions) and median imputation (for skewed distributions where mean would be influenced by extreme values).

## Outlier Detection and Treatment

Outlier detection employs the Interquartile Range (IQR) method, a robust statistical approach that identifies outliers as values falling below Q1 - 1.5×IQR or above Q3 + 1.5×IQR. Rather than removing outliers (which would result in data loss), the pipeline uses a **capping strategy** that replaces extreme values with the boundary values. This approach preserves all data points while limiting the influence of extreme values on subsequent analyses.

## Quality Assurance

All transformations are logged with complete traceability. The pipeline generates comprehensive visualizations including box plots, distribution histograms, and correlation matrices to enable manual verification of automated decisions. This documentation serves as an audit trail and enables reproducibility of the entire cleaning process."""

        return methodology

    def _generate_recommendations(self, df_final: pd.DataFrame, report: Dict[str, Any]) -> str:
        """Generate AI-powered recommendations."""
        numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_final.select_dtypes(include=["object", "category"]).columns.tolist()

        prompt = f"""Write specific, actionable recommendations for a data scientist who will use this cleaned dataset.

Dataset after cleaning:
- {len(df_final):,} rows, {len(df_final.columns)} columns
- {len(numeric_cols)} numeric columns
- {len(cat_cols)} categorical columns
- Remaining missing values: {df_final.isnull().sum().sum()}

Provide recommendations for:
1. Feature Engineering - specific suggestions
2. Encoding Categorical Variables - which methods to use
3. Feature Scaling - when and how
4. Potential Issues - what to watch for
5. Model Selection - initial suggestions
6. Next Steps - prioritized action items

Be specific and practical. Format as bullet points under each category.
Do not use markdown headers (no # symbols)."""

        recommendations = self._get_ai_summary(prompt)

        if not recommendations:
            recommendations = f"""## Data Preparation Recommendations

**Categorical Encoding ({len(cat_cols)} columns):**

- Apply **one-hot encoding** for categorical features with fewer than 10 unique values to create binary indicator variables
- Use **label encoding** or **target encoding** for high-cardinality categorical features to avoid dimensionality explosion
- Consider **ordinal encoding** for features with natural ordering (e.g., education level, income bracket)

**Feature Scaling ({len(numeric_cols)} numeric columns):**

- Apply **StandardScaler** (z-score normalization) for algorithms sensitive to feature magnitudes (SVM, neural networks, k-NN)
- Use **MinMaxScaler** for algorithms requiring bounded inputs or when preserving zero values is important
- Consider **RobustScaler** for features that may still contain mild outliers after cleaning

## Feature Engineering Opportunities

- Review the correlation matrix visualization to identify highly correlated features (|r| > 0.9) that may require removal or combination
- Create interaction terms between features showing strong correlations with the target variable
- Consider polynomial features for capturing non-linear relationships
- Extract date components (year, month, day, day_of_week) if datetime features are present

## Model Development Suggestions

- Start with baseline models (Logistic Regression for classification, Linear Regression for regression) to establish performance benchmarks
- Progress to ensemble methods (Random Forest, XGBoost, LightGBM) for improved performance
- Use cross-validation (5-fold recommended) to ensure robust model evaluation
- Implement early stopping for gradient boosting models to prevent overfitting

## Potential Issues to Monitor

- Verify class balance in classification targets; apply SMOTE or class weights if imbalanced
- Check for data leakage - ensure no future information is used for prediction
- Monitor feature distributions in production data for drift from training distributions
- Validate that imputed values fall within expected domain ranges

## Prioritized Next Steps

1. Load the cleaned dataset and review the generated visualizations
2. Perform exploratory data analysis focusing on target variable relationships
3. Encode categorical variables using appropriate methods
4. Scale numeric features based on chosen algorithm requirements
5. Split data into train/validation/test sets with stratification
6. Train baseline model and establish performance benchmarks
7. Iterate with feature engineering and advanced models"""

        return recommendations

    def _generate_data_dictionary(self, df: pd.DataFrame) -> str:
        """Generate comprehensive data dictionary."""
        lines = ["## Column Definitions\n"]
        lines.append("| # | Column Name | Data Type | Non-Null Count | Unique Values | Description |")
        lines.append("|---|-------------|-----------|----------------|---------------|-------------|")

        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            unique = df[col].nunique()

            # Generate description based on column name and type
            desc = self._infer_column_description(col, df[col])

            lines.append(f"| {i} | `{col}` | {dtype} | {non_null:,} | {unique:,} | {desc} |")

        return "\n".join(lines)

    def _infer_column_description(self, col_name: str, series: pd.Series) -> str:
        """Infer column description from name and data."""
        name_lower = col_name.lower()

        # Common patterns
        if 'id' in name_lower:
            return "Unique identifier"
        elif 'date' in name_lower or 'time' in name_lower:
            return "Temporal feature"
        elif 'price' in name_lower or 'cost' in name_lower or 'revenue' in name_lower:
            return "Monetary value"
        elif 'age' in name_lower:
            return "Age in years"
        elif 'name' in name_lower:
            return "Name/label field"
        elif 'count' in name_lower or 'num' in name_lower:
            return "Count/quantity"
        elif 'rate' in name_lower or 'ratio' in name_lower or 'pct' in name_lower:
            return "Rate/percentage"
        elif 'score' in name_lower or 'rating' in name_lower:
            return "Score/rating value"
        elif pd.api.types.is_numeric_dtype(series):
            return "Numeric feature"
        else:
            return "Categorical feature"

    def _format_column_list(self, columns: List[str]) -> str:
        """Format column list as markdown."""
        if not columns:
            return "*None*"
        formatted = ", ".join([f"`{col}`" for col in columns[:15]])
        if len(columns) > 15:
            formatted += f" ... and {len(columns)-15} more"
        return formatted

    def _format_dtypes_summary(self) -> str:
        """Format data types summary."""
        dtype_counts = {}
        for dtype in self.initial_state['dtypes'].values():
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        lines = []
        for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
            lines.append(f"{dtype}: {count} columns")
        return "\n".join(lines)

    def _generate_missing_values_analysis(self) -> str:
        """Generate missing values analysis section."""
        missing_cols = {k: v for k, v in self.initial_state['missing_values'].items() if v > 0}

        if not missing_cols:
            return "**No missing values were found in the original dataset.**"

        lines = ["| Column | Missing Count | Missing % | Severity |"]
        lines.append("|--------|---------------|-----------|----------|")

        for col, count in sorted(missing_cols.items(), key=lambda x: -x[1])[:25]:
            pct = count / self.initial_state['shape'][0] * 100
            severity = "High" if pct > 30 else "Medium" if pct > 10 else "Low"
            lines.append(f"| `{col}` | {count:,} | {pct:.1f}% | {severity} |")

        if len(missing_cols) > 25:
            lines.append(f"| *... {len(missing_cols)-25} more columns* | | | |")

        return "\n".join(lines)

    def _generate_quality_metrics(self, report: Dict[str, Any]) -> str:
        """Generate quality metrics section."""
        quality = report.get('data_quality_improvement', {})
        total_cells = self.initial_state['shape'][0] * self.initial_state['shape'][1]
        completeness = (total_cells - self.initial_state['total_missing']) / total_cells * 100

        return f"""
| Metric | Value |
|--------|-------|
| **Initial Completeness** | {completeness:.1f}% |
| **Total Missing Values** | {self.initial_state['total_missing']:,} |
| **Duplicate Rows** | {self.initial_state['duplicates']:,} |
| **Data Quality Level** | {'Good' if completeness > 90 else 'Fair' if completeness > 70 else 'Poor'} |
"""

    def _generate_detailed_steps(self, report: Dict[str, Any]) -> str:
        """Generate detailed step documentation."""
        steps = report.get('steps', [])
        if not steps:
            return "*No steps recorded*"

        lines = []
        for i, step in enumerate(steps, 1):
            status = "✓ Completed" if step.get('status') == 'completed' else "○ Skipped"
            lines.append(f"## Step {i}: {step.get('name', 'Unknown').replace('_', ' ').title()}")
            lines.append(f"\n**Status:** {status}\n")

            step_report = step.get('report', {})
            if isinstance(step_report, dict) and step_report:
                lines.append("**Details:**\n")
                for key, value in list(step_report.items())[:10]:
                    if isinstance(value, (int, float)):
                        lines.append(f"- {key.replace('_', ' ').title()}: {value:,}" if isinstance(value, int) else f"- {key.replace('_', ' ').title()}: {value:.2f}")
                    elif isinstance(value, str):
                        lines.append(f"- {key.replace('_', ' ').title()}: {value}")
            lines.append("\n")

        return "\n".join(lines)

    def _generate_imputation_details(self, report: Dict[str, Any]) -> str:
        """Generate imputation details section."""
        steps = report.get('steps', [])
        impute_step = next((s for s in steps if s.get('name') == 'impute_missing'), None)

        if not impute_step:
            return "*Imputation step not found in report*"

        imputations = impute_step.get('report', {}).get('imputations', [])
        if not imputations:
            return "*No columns required imputation*"

        lines = ["## Imputation Summary\n"]
        lines.append("| Column | Strategy | Confidence | Reasoning |")
        lines.append("|--------|----------|------------|-----------|")

        for imp in imputations[:35]:
            col = imp.get('column', 'Unknown')
            strategy = imp.get('strategy', 'Unknown')
            confidence = imp.get('confidence', 'N/A')
            reasoning = imp.get('reasoning', 'N/A')
            if len(reasoning) > 50:
                reasoning = reasoning[:50] + "..."
            lines.append(f"| `{col}` | {strategy} | {confidence} | {reasoning} |")

        if len(imputations) > 35:
            lines.append(f"| *... {len(imputations)-35} more columns* | | | |")

        # Add summary statistics
        strategy_counts = {}
        for imp in imputations:
            s = imp.get('strategy', 'unknown')
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        lines.append("\n## Strategy Distribution\n")
        lines.append("| Strategy | Count |")
        lines.append("|----------|-------|")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {strategy} | {count} |")

        return "\n".join(lines)

    def _generate_outlier_details(self, report: Dict[str, Any]) -> str:
        """Generate outlier details section."""
        steps = report.get('steps', [])
        outlier_step = next((s for s in steps if s.get('name') == 'handle_outliers'), None)

        if not outlier_step:
            return "*Outlier handling step was skipped or not found*"

        outliers = outlier_step.get('report', {}).get('outliers', [])
        if not outliers:
            return "**No significant outliers were detected in the numeric columns.**"

        lines = ["## Outliers Detected and Treated\n"]
        lines.append("| Column | Outliers Found | Percentage | Treatment |")
        lines.append("|--------|----------------|------------|-----------|")

        total_outliers = 0
        for out in outliers[:20]:
            col = out.get('column', 'Unknown')
            count = out.get('outliers_detected', 0)
            pct = out.get('percentage', 0)
            total_outliers += count
            lines.append(f"| `{col}` | {count:,} | {pct:.1f}% | Capped at IQR bounds |")

        if len(outliers) > 20:
            lines.append(f"| *... {len(outliers)-20} more columns* | | | |")

        lines.append(f"\n**Total outliers treated:** {total_outliers:,}")

        return "\n".join(lines)

    def _generate_comparison_table(self, df_final: pd.DataFrame, report: Dict[str, Any]) -> str:
        """Generate before/after comparison table."""
        quality = report.get('data_quality_improvement', {})

        return f"""
| Metric | Before Cleaning | After Cleaning | Change |
|--------|-----------------|----------------|--------|
| **Rows** | {self.initial_state['shape'][0]:,} | {len(df_final):,} | {len(df_final) - self.initial_state['shape'][0]:+,} |
| **Columns** | {self.initial_state['shape'][1]} | {len(df_final.columns)} | {len(df_final.columns) - self.initial_state['shape'][1]:+} |
| **Missing Values** | {self.initial_state['total_missing']:,} | {df_final.isnull().sum().sum():,} | {df_final.isnull().sum().sum() - self.initial_state['total_missing']:+,} |
| **Completeness** | {quality.get('completeness_before', 0):.1f}% | {quality.get('completeness_after', 0):.1f}% | +{quality.get('completeness_improvement', 0):.1f}% |
| **Memory (MB)** | {self.initial_state['memory_mb']:.2f} | {df_final.memory_usage(deep=True).sum()/1024/1024:.2f} | - |
"""

    def _generate_final_statistics(self, df: pd.DataFrame) -> str:
        """Generate final statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:12]

        if len(numeric_cols) == 0:
            return "*No numeric columns in final dataset*"

        lines = ["| Column | Mean | Std | Min | 25% | 50% | 75% | Max |"]
        lines.append("|--------|------|-----|-----|-----|-----|-----|-----|")

        for col in numeric_cols:
            stats = df[col].describe()
            lines.append(f"| `{col}` | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['25%']:.2f} | {stats['50%']:.2f} | {stats['75%']:.2f} | {stats['max']:.2f} |")

        if len(df.select_dtypes(include=[np.number]).columns) > 12:
            lines.append(f"| *... {len(df.select_dtypes(include=[np.number]).columns)-12} more columns* | | | | | | | |")

        return "\n".join(lines)

    def _generate_transformation_log(self, report: Dict[str, Any]) -> str:
        """Generate transformation log."""
        lines = []
        lines.append(f"Pipeline Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'}")
        lines.append(f"Dataset: {self.dataset_name}")
        lines.append(f"Initial Shape: {self.initial_state['shape']}")
        lines.append("")

        for i, step in enumerate(report.get('steps', []), 1):
            status = "COMPLETED" if step.get('status') == 'completed' else "SKIPPED"
            lines.append(f"[{i}] {step.get('name', 'Unknown').upper()}: {status}")

        lines.append("")
        lines.append(f"Pipeline End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    def _generate_complete_column_reference(self, df: pd.DataFrame) -> str:
        """Generate complete column reference."""
        lines = ["| # | Column | Type | Non-Null | Unique | Sample Values |"]
        lines.append("|---|--------|------|----------|--------|---------------|")

        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)[:12]
            non_null = df[col].notna().sum()
            unique = df[col].nunique()
            samples = str(df[col].dropna().head(2).tolist())[:25] + "..."
            lines.append(f"| {i} | `{col}` | {dtype} | {non_null:,} | {unique:,} | {samples} |")

        return "\n".join(lines)
