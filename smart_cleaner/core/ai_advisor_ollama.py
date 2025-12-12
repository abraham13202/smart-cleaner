"""
AI-powered data cleaning advisor using Ollama (LOCAL - FREE!).
Provides intelligent recommendations for data imputation and cleaning strategies.
Runs entirely on your local machine with no API costs.

Features CONTEXT-AWARE IMPUTATION:
- Detects demographic/categorical columns (age, gender, category, etc.)
- Recommends cohort-based imputation when appropriate
- Example: Missing BMI for 34-year-old female → uses mean BMI of females aged 30-40
"""

import json
import re
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

try:
    import ollama
except ImportError:
    ollama = None

from ..utils.validators import (
    get_column_statistics,
    detect_correlations,
    get_missing_value_summary,
)


class OllamaAdvisor:
    """
    AI-powered advisor using Ollama for local LLM inference.
    Analyzes data and provides intelligent, CONTEXT-AWARE recommendations.

    Key Feature: Cohort-based imputation
    - Detects grouping columns (age, gender, category, region, etc.)
    - Recommends imputing based on similar groups
    - Example: BMI missing for 34F → mean BMI of females aged 30-40
    """

    # Common demographic/grouping column patterns
    COHORT_PATTERNS = {
        'age': r'(?i)^(age|edad|alter|anni)s?$|age_|_age',
        'gender': r'(?i)^(gender|sex|male|female|geschlecht)$|gender_|_gender|_sex',
        'category': r'(?i)(category|type|class|group|segment|tier|level|grade)',
        'region': r'(?i)(region|country|state|city|location|area|zone|district)',
        'department': r'(?i)(department|dept|division|unit|team)',
        'status': r'(?i)(status|state|condition)',
        'year': r'(?i)^year$|_year$|^yr$',
        'month': r'(?i)^month$|_month$',
    }

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: Optional[str] = None,
    ):
        """
        Initialize Ollama Advisor.

        Args:
            model: Ollama model to use (e.g., 'llama3.2', 'mistral', 'codellama')
            base_url: Optional custom Ollama server URL (default: http://localhost:11434)
        """
        if ollama is None:
            raise ImportError(
                "ollama package not installed. "
                "Install it with: pip install ollama\n"
                "Also ensure Ollama is running: https://ollama.ai"
            )

        self.model = model
        self.base_url = base_url
        self._cohort_columns_cache = None

        # Configure custom host if provided
        if base_url:
            self.client = ollama.Client(host=base_url)
        else:
            self.client = ollama.Client()

    def analyze_missing_values(
        self, df: pd.DataFrame, column: str
    ) -> Dict[str, Any]:
        """
        Analyze missing values in a column and get AI recommendations.
        Uses CONTEXT-AWARE analysis to recommend cohort-based imputation.

        Args:
            df: DataFrame to analyze
            column: Column name with missing values

        Returns:
            Dictionary with recommendations and reasoning
        """
        # Detect potential cohort columns once per dataframe
        if self._cohort_columns_cache is None:
            self._cohort_columns_cache = self._detect_cohort_columns(df)

        # Gather context about the data
        context = self._build_context(df, column)

        # Get AI recommendation
        recommendation = self._get_ai_recommendation(context)

        return recommendation

    def analyze_all_missing(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze all columns with missing values and get recommendations.

        Args:
            df: DataFrame to analyze

        Returns:
            List of recommendations for each column with missing values
        """
        # Detect cohort columns once for the entire dataframe
        self._cohort_columns_cache = self._detect_cohort_columns(df)

        missing_summary = get_missing_value_summary(df)
        recommendations = []

        for column in missing_summary.keys():
            try:
                rec = self.analyze_missing_values(df, column)
                recommendations.append(rec)
            except Exception as e:
                recommendations.append({
                    "column": column,
                    "error": str(e),
                    "strategy": "mode" if df[column].dtype == 'object' else "mean",
                })

        return recommendations

    def _detect_cohort_columns(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Detect potential cohort/grouping columns in the dataframe.

        Returns dict with:
        - 'categorical': columns good for direct grouping (gender, category)
        - 'numeric_binnable': numeric columns good for binning (age, year)
        """
        cohort_columns = {
            'categorical': [],
            'numeric_binnable': [],
        }

        for col in df.columns:
            col_lower = col.lower()
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            non_null_count = df[col].notna().sum()

            # Skip columns with too many missing values
            if non_null_count < len(df) * 0.5:
                continue

            # Check for categorical cohort columns
            if dtype == 'object' or unique_count <= 20:
                # Check if it matches known patterns
                cohort_type = None
                for pattern_name, pattern in self.COHORT_PATTERNS.items():
                    if re.search(pattern, col):
                        cohort_type = pattern_name
                        break

                if cohort_type or (unique_count >= 2 and unique_count <= 15):
                    cohort_columns['categorical'].append({
                        'column': col,
                        'type': cohort_type or 'categorical',
                        'unique_values': unique_count,
                        'sample_values': df[col].dropna().unique()[:5].tolist(),
                    })

            # Check for numeric columns good for binning (like age)
            elif dtype in ['int64', 'float64', 'int32', 'float32']:
                for pattern_name, pattern in self.COHORT_PATTERNS.items():
                    if re.search(pattern, col):
                        col_min = df[col].min()
                        col_max = df[col].max()
                        cohort_columns['numeric_binnable'].append({
                            'column': col,
                            'type': pattern_name,
                            'min': float(col_min) if pd.notna(col_min) else 0,
                            'max': float(col_max) if pd.notna(col_max) else 100,
                            'suggested_bins': self._suggest_bins(df[col], pattern_name),
                        })
                        break

        return cohort_columns

    def _suggest_bins(self, series: pd.Series, col_type: str) -> List[int]:
        """Suggest appropriate bins for a numeric column."""
        col_min = series.min()
        col_max = series.max()

        if col_type == 'age':
            # Age bins: 0-18, 18-30, 30-40, 40-50, 50-60, 60-70, 70+
            return [0, 18, 30, 40, 50, 60, 70, 100]
        elif col_type == 'year':
            # Decade bins
            start = int(col_min // 10 * 10)
            end = int(col_max // 10 * 10) + 10
            return list(range(start, end + 10, 10))
        else:
            # Quartile-based bins
            try:
                quantiles = series.quantile([0, 0.25, 0.5, 0.75, 1.0]).tolist()
                return [int(q) for q in quantiles]
            except:
                return [int(col_min), int(col_max)]

    def _build_context(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Build comprehensive context about the DataFrame and target column."""
        # Get column statistics
        col_stats = get_column_statistics(df, column)

        # Get correlations with other numeric columns
        correlations = detect_correlations(df, threshold=0.2)
        relevant_correlations = [
            c for c in correlations
            if c["column1"] == column or c["column2"] == column
        ]

        # Get sample values (non-missing)
        sample_values = df[column].dropna().head(10).tolist()

        # Get related columns info
        related_columns = {}
        for corr in relevant_correlations[:3]:  # Top 3 correlations
            related_col = (
                corr["column2"] if corr["column1"] == column else corr["column1"]
            )
            related_columns[related_col] = get_column_statistics(df, related_col)

        context = {
            "target_column": column,
            "statistics": col_stats,
            "sample_values": sample_values,
            "correlations": relevant_correlations,
            "related_columns": related_columns,
            "total_rows": len(df),
            "missing_percentage": col_stats["missing_count"] / len(df) * 100,
            "cohort_columns": self._cohort_columns_cache,
            "all_columns": list(df.columns),
        }

        return context

    def _get_ai_recommendation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered recommendation from Ollama."""
        prompt = self._build_prompt(context)

        try:
            system_prompt = """You are the greatest data analyst in the world, specializing in CONTEXT-AWARE data imputation.

YOUR SUPERPOWER: You don't just use simple mean/median - you use COHORT-BASED IMPUTATION for smarter results.

EXAMPLE OF SMART IMPUTATION:
- Column: BMI with missing values
- Available cohort columns: gender (M/F), age (numeric)
- SMART approach: For a 34-year-old female with missing BMI, use mean BMI of females aged 30-40
- This is MUCH better than using overall mean BMI!

STRATEGY SELECTION RULES:
1. For TEXT/OBJECT columns: ALWAYS use "mode"
2. For NUMERIC columns with good cohort columns available: Use "cohort_mean"
3. For NUMERIC columns without cohort columns: Use "mean" or "median"
4. If strong correlations exist: Consider "knn"

WHEN TO USE COHORT_MEAN:
- When categorical columns exist (gender, category, region, type, etc.)
- When numeric columns can be binned (age → age groups, year → decades)
- When it makes domain sense (BMI varies by age/gender, salary varies by department/level)

Always respond with valid JSON only, no other text."""

            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.3}
            )

            response_text = response["message"]["content"]
            recommendation = self._parse_recommendation(response_text, context)
            return recommendation

        except Exception as e:
            # Fallback to simple strategy on error
            dtype = context["statistics"]["dtype"]
            is_numeric = dtype in ["int64", "float64", "int32", "float32"]
            return {
                "column": context["target_column"],
                "strategy": "mean" if is_numeric else "mode",
                "reasoning": f"Error getting AI recommendation: {str(e)}. Using fallback strategy.",
                "confidence": "low",
                "parameters": {},
            }

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Ollama with cohort information."""
        column = context["target_column"]
        stats = context["statistics"]
        correlations = context["correlations"]
        cohort_info = context.get("cohort_columns", {})
        dtype = stats['dtype']

        is_text = dtype in ['object', 'string', 'category']
        is_numeric = dtype in ['int64', 'float64', 'int32', 'float32']

        # Build cohort columns description
        cohort_description = ""
        if cohort_info:
            cat_cohorts = cohort_info.get('categorical', [])
            num_cohorts = cohort_info.get('numeric_binnable', [])

            if cat_cohorts or num_cohorts:
                cohort_description = "\n\nAVAILABLE COHORT COLUMNS FOR CONTEXT-AWARE IMPUTATION:\n"

                if cat_cohorts:
                    cohort_description += "\nCategorical (can group directly):\n"
                    for c in cat_cohorts[:5]:
                        cohort_description += f"  - {c['column']}: {c['unique_values']} unique values, e.g., {c['sample_values'][:3]}\n"

                if num_cohorts:
                    cohort_description += "\nNumeric (can bin into groups):\n"
                    for c in num_cohorts[:3]:
                        cohort_description += f"  - {c['column']}: range {c['min']}-{c['max']}, suggested bins: {c['suggested_bins']}\n"

        # Data type instruction
        if is_text:
            dtype_instruction = f"""
IMPORTANT: This column has dtype '{dtype}' which is TEXT/CATEGORICAL.
You MUST use "mode". DO NOT use mean/median/cohort_mean for text!"""
        else:
            dtype_instruction = f"""
This column has dtype '{dtype}' which is NUMERIC.
Consider using "cohort_mean" if relevant cohort columns are available.
Otherwise use "mean", "median", or "knn"."""

        prompt = f"""Analyze this column and recommend the BEST imputation strategy.

TARGET COLUMN: {column}
Data Type: {dtype}
Missing Values: {stats['missing_count']} ({context['missing_percentage']:.2f}%)
Total Rows: {context['total_rows']}
{dtype_instruction}

COLUMN STATISTICS:
{json.dumps(stats, indent=2)}

SAMPLE VALUES: {context['sample_values'][:5]}

CORRELATIONS WITH OTHER COLUMNS:
{json.dumps(correlations[:5], indent=2) if correlations else "None significant"}
{cohort_description}

THINK ABOUT:
1. Is this column likely to vary by demographic groups? (e.g., BMI by age/gender, salary by department)
2. Are there good cohort columns available for context-aware imputation?
3. Would simple mean/median lose important context?

Respond with JSON only:
{{
  "strategy": "mode|mean|median|cohort_mean|knn",
  "reasoning": "Explain your choice, especially if using cohort_mean",
  "confidence": "high|medium|low",
  "parameters": {{
    "cohort_columns": ["col1", "col2"],  // if using cohort_mean
    "cohort_bins": {{"age": [0,30,50,70,100]}}  // if binning numeric cohorts
  }}
}}"""

        return prompt

    def _parse_recommendation(
        self, response_text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse AI response into structured recommendation."""
        valid_strategies = ["mean", "median", "mode", "cohort_mean", "knn", "drop", "forward_fill", "backward_fill"]
        numeric_only_strategies = ["mean", "median", "knn", "cohort_mean"]

        dtype = context["statistics"]["dtype"]
        is_text = dtype in ["object", "string", "category"]
        is_numeric = dtype in ["int64", "float64", "int32", "float32"]

        try:
            # Extract JSON from response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                recommendation = json.loads(json_str)
                recommendation["column"] = context["target_column"]

                # Fix: LLM sometimes returns "mean|median" - take the first valid one
                strategy = recommendation.get("strategy", "")
                if "|" in strategy:
                    options = [s.strip() for s in strategy.split("|")]
                    for opt in options:
                        if opt in valid_strategies:
                            recommendation["strategy"] = opt
                            break
                    else:
                        recommendation["strategy"] = "mean" if is_numeric else "mode"

                # CRITICAL: Force mode for text columns
                if is_text and recommendation.get("strategy") in numeric_only_strategies:
                    recommendation["strategy"] = "mode"
                    recommendation["reasoning"] = f"Forced to 'mode' because column is text/object type. {recommendation.get('reasoning', '')}"

                # Validate strategy
                if recommendation.get("strategy") not in valid_strategies:
                    recommendation["strategy"] = "mean" if is_numeric else "mode"

                # Validate cohort_mean has parameters
                if recommendation.get("strategy") == "cohort_mean":
                    params = recommendation.get("parameters", {})
                    cohort_cols = params.get("cohort_columns", [])

                    # Verify cohort columns exist
                    all_cols = context.get("all_columns", [])
                    valid_cohort_cols = [c for c in cohort_cols if c in all_cols]

                    if not valid_cohort_cols:
                        # Try to auto-detect from our cache
                        cohort_info = context.get("cohort_columns", {})
                        cat_cohorts = [c['column'] for c in cohort_info.get('categorical', [])]
                        num_cohorts = [c['column'] for c in cohort_info.get('numeric_binnable', [])]

                        if cat_cohorts:
                            valid_cohort_cols = cat_cohorts[:2]
                            recommendation["parameters"]["cohort_columns"] = valid_cohort_cols
                        elif num_cohorts:
                            valid_cohort_cols = num_cohorts[:1]
                            recommendation["parameters"]["cohort_columns"] = valid_cohort_cols
                            # Add suggested bins
                            for nc in cohort_info.get('numeric_binnable', []):
                                if nc['column'] in valid_cohort_cols:
                                    if "cohort_bins" not in recommendation["parameters"]:
                                        recommendation["parameters"]["cohort_bins"] = {}
                                    recommendation["parameters"]["cohort_bins"][nc['column']] = nc['suggested_bins']
                        else:
                            # No valid cohort columns, fall back to mean
                            recommendation["strategy"] = "mean"
                            recommendation["reasoning"] += " (Fallback: no valid cohort columns found)"

                return recommendation
            else:
                raise ValueError("No JSON found in response")

        except (json.JSONDecodeError, ValueError) as e:
            # Smart fallback based on available cohort columns
            cohort_info = context.get("cohort_columns", {})
            cat_cohorts = cohort_info.get('categorical', [])

            if is_numeric and cat_cohorts:
                # We have cohort columns - use cohort_mean
                return {
                    "column": context["target_column"],
                    "strategy": "cohort_mean",
                    "reasoning": f"Parse error, but using cohort_mean with detected cohort columns.",
                    "confidence": "medium",
                    "parameters": {
                        "cohort_columns": [cat_cohorts[0]['column']],
                    },
                }
            else:
                return {
                    "column": context["target_column"],
                    "strategy": "mean" if is_numeric else "mode",
                    "reasoning": f"Could not parse AI response: {str(e)}. Using simple strategy.",
                    "confidence": "low",
                    "parameters": {},
                }
