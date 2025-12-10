"""
AI-powered data cleaning advisor using Google Gemini.
Provides intelligent recommendations for data imputation and cleaning strategies.
"""

import json
from typing import Dict, List, Any, Optional
import pandas as pd

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from ..utils.config import Config
from ..utils.validators import (
    get_column_statistics,
    detect_correlations,
    get_missing_value_summary,
)


class AIAdvisor:
    """
    AI-powered advisor that analyzes data and provides intelligent
    recommendations for cleaning and imputation strategies.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize AI Advisor.

        Args:
            config: Configuration object with API credentials
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install it with: pip install google-generativeai"
            )

        self.config = config or Config()
        self.config.validate()
        genai.configure(api_key=self.config.gemini_api_key)
        self.model = genai.GenerativeModel(self.config.model)

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
        """
        Build comprehensive context about the DataFrame and target column.

        Args:
            df: DataFrame to analyze
            column: Target column name

        Returns:
            Context dictionary with all relevant information
        """
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
        """
        Get AI-powered recommendation from Gemini.

        Args:
            context: Data context dictionary

        Returns:
            Recommendation dictionary with strategy and parameters
        """
        prompt = self._build_prompt(context)

        try:
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            response_text = response.text
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
        """Build prompt for Gemini API."""
        column = context["target_column"]
        stats = context["statistics"]
        correlations = context["correlations"]

        prompt = f"""You are a data science expert specializing in data cleaning and imputation strategies.

Analyze the following column with missing values and recommend the best imputation strategy:

Column: {column}
Data Type: {stats['dtype']}
Missing Values: {stats['missing_count']} ({context['missing_percentage']:.2f}%)
Total Rows: {context['total_rows']}

Statistics:
{json.dumps(stats, indent=2)}

Sample Values:
{context['sample_values']}

Correlations with other columns:
{json.dumps(correlations, indent=2)}

Related Columns Information:
{json.dumps(context['related_columns'], indent=2)}

Based on this analysis, recommend the best imputation strategy. Consider:
1. The data type and distribution
2. Correlations with other columns (for context-aware imputation)
3. The percentage of missing values
4. Domain-specific considerations (e.g., for healthcare data like BMI, age-based cohorts might be relevant)

Respond in the following JSON format:
{{
  "strategy": "mean|median|mode|cohort_mean|knn|drop|forward_fill|backward_fill|custom",
  "reasoning": "Detailed explanation of why this strategy is recommended",
  "confidence": "high|medium|low",
  "parameters": {{
    "cohort_column": "column_name (if using cohort-based strategy)",
    "cohort_bins": [[min, max], ...] (if using age/value cohorts),
    "k_neighbors": 5 (if using KNN),
    "custom_logic": "description of custom logic if needed"
  }},
  "alternative_strategies": ["strategy1", "strategy2"]
}}

Provide only the JSON response, no additional text."""

        return prompt

    def _parse_recommendation(
        self, response_text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse AI response into structured recommendation.

        Args:
            response_text: Raw response from Claude
            context: Original context for fallback

        Returns:
            Parsed recommendation dictionary
        """
        try:
            # Try to extract JSON from response
            # Sometimes Claude might add explanation before/after JSON
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                recommendation = json.loads(json_str)
                recommendation["column"] = context["target_column"]
                return recommendation
            else:
                raise ValueError("No JSON found in response")

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback if parsing fails
            return {
                "column": context["target_column"],
                "strategy": "mean" if context["statistics"]["dtype"] in ["int64", "float64"] else "mode",
                "reasoning": f"Could not parse AI response: {str(e)}. Using simple strategy.",
                "confidence": "low",
                "parameters": {},
            }
