"""
AI-powered data cleaning advisor using Ollama (LOCAL - FREE!).
Provides intelligent recommendations for data imputation and cleaning strategies.
Runs entirely on your local machine with no API costs.
"""

import json
from typing import Dict, List, Any, Optional
import pandas as pd

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
    Analyzes data and provides intelligent recommendations for cleaning.
    Runs completely locally - no API costs or internet required.
    """

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

        Args:
            df: DataFrame to analyze
            column: Column name with missing values

        Returns:
            Dictionary with recommendations and reasoning
        """
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
                    "strategy": "drop",  # Fallback strategy
                })

        return recommendations

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
        }

        return context

    def _get_ai_recommendation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered recommendation from Ollama."""
        prompt = self._build_prompt(context)

        try:
            system_prompt = """You are the greatest data analyst in the world, specializing in data imputation and cleaning.
Your role is to analyze datasets and recommend the most accurate imputation strategies before handing the data over to data scientists.

CRITICAL RULES FOR IMPUTATION STRATEGY:
- For TEXT/STRING/OBJECT columns (dtype: object): ALWAYS use "mode" - NEVER use "mean" or "median"
- For NUMERIC columns (dtype: int64, float64): Use "mean", "median", "knn", or "cohort_mean"
- For CATEGORICAL columns with few unique values: Use "mode"
- Mean and median ONLY work on numbers, not text!

You excel at:
- Identifying the best imputation method based on DATA TYPE first, then distribution
- Using MODE for any text, string, categorical, or object type columns
- Using MEAN/MEDIAN only for numeric columns
- Detecting correlations that can improve imputation accuracy
- Choosing context-aware strategies (cohort-based, KNN) when appropriate

Always respond with valid JSON only, no other text or explanation outside the JSON."""

            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.3,  # Lower temperature for more consistent JSON
                }
            )

            response_text = response["message"]["content"]
            recommendation = self._parse_recommendation(response_text, context)
            return recommendation

        except Exception as e:
            # Fallback to simple strategy on error
            return {
                "column": context["target_column"],
                "strategy": "mean" if context["statistics"]["dtype"] in ["int64", "float64"] else "mode",
                "reasoning": f"Error getting AI recommendation: {str(e)}. Using fallback strategy.",
                "confidence": "low",
                "parameters": {},
            }

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Ollama."""
        column = context["target_column"]
        stats = context["statistics"]
        correlations = context["correlations"]
        dtype = stats['dtype']

        # Determine if column is text/object or numeric
        is_text = dtype == 'object' or dtype == 'string' or dtype == 'category'
        dtype_instruction = ""
        if is_text:
            dtype_instruction = f"""
IMPORTANT: This column has dtype '{dtype}' which is TEXT/CATEGORICAL.
You MUST use "mode" for text columns. DO NOT use "mean" or "median" - they only work on numbers!"""
        else:
            dtype_instruction = f"""
This column has dtype '{dtype}' which is NUMERIC.
You can use "mean", "median", "knn", or "cohort_mean"."""

        prompt = f"""Analyze this column with missing values and recommend the best imputation strategy.

Column: {column}
Data Type: {dtype}
{dtype_instruction}

Missing Values: {stats['missing_count']} ({context['missing_percentage']:.2f}%)
Total Rows: {context['total_rows']}

Statistics:
{json.dumps(stats, indent=2)}

Sample Values:
{context['sample_values']}

Correlations with other columns:
{json.dumps(correlations, indent=2)}

STRATEGY RULES:
- dtype "object" or text columns -> MUST use "mode"
- dtype "int64" or "float64" -> can use "mean", "median", "knn", "cohort_mean"

Respond ONLY with this JSON format (no other text):
{{
  "strategy": "{'mode' if is_text else 'mean|median|knn|cohort_mean'}",
  "reasoning": "Brief explanation",
  "confidence": "high|medium|low",
  "parameters": {{}},
  "alternative_strategies": []
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
            # Try to extract JSON from response
            # LLMs might add explanation before/after JSON
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

                # CRITICAL: Force mode for text columns even if LLM says otherwise
                if is_text and recommendation.get("strategy") in numeric_only_strategies:
                    recommendation["strategy"] = "mode"
                    recommendation["reasoning"] = f"Forced to 'mode' because column is text/object type. {recommendation.get('reasoning', '')}"

                # Validate strategy is known
                if recommendation.get("strategy") not in valid_strategies:
                    recommendation["strategy"] = "mean" if is_numeric else "mode"

                return recommendation
            else:
                raise ValueError("No JSON found in response")

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback if parsing fails
            return {
                "column": context["target_column"],
                "strategy": "mean" if is_numeric else "mode",
                "reasoning": f"Could not parse AI response: {str(e)}. Using simple strategy.",
                "confidence": "low",
                "parameters": {},
            }
