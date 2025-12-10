"""
Main DataCleaner class for orchestrating data cleaning operations.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import warnings

from ..utils.config import Config
from ..utils.validators import validate_dataframe, get_missing_value_summary
from .ai_advisor import AIAdvisor
from .imputation import ImputationEngine


class DataCleaner:
    """
    Main class for intelligent data cleaning and imputation.

    This class orchestrates the entire data cleaning workflow:
    1. Analyze data for missing values and quality issues
    2. Get AI-powered recommendations for cleaning strategies
    3. Apply recommended or custom imputation strategies
    4. Track and log all transformations
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Config] = None):
        """
        Initialize DataCleaner.

        Args:
            api_key: Anthropic API key (optional if set in environment)
            config: Custom configuration object
        """
        if config is None:
            config = Config(anthropic_api_key=api_key)

        self.config = config
        self.ai_advisor = AIAdvisor(config)
        self.transformation_log: List[Dict[str, Any]] = []

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze DataFrame and get AI-powered recommendations.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing:
                - missing_summary: Overview of missing values
                - recommendations: AI recommendations for each column
                - data_quality_score: Overall quality assessment
        """
        validate_dataframe(df)

        # Get missing value summary
        missing_summary = get_missing_value_summary(df)

        if not missing_summary:
            return {
                "missing_summary": {},
                "recommendations": [],
                "data_quality_score": 100.0,
                "message": "No missing values found. Data is complete!",
            }

        # Get AI recommendations for each column with missing values
        print(f"Analyzing {len(missing_summary)} columns with missing values...")
        recommendations = self.ai_advisor.analyze_all_missing(df)

        # Calculate data quality score
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = sum(info["count"] for info in missing_summary.values())
        quality_score = ((total_cells - missing_cells) / total_cells) * 100

        return {
            "missing_summary": missing_summary,
            "recommendations": recommendations,
            "data_quality_score": round(quality_score, 2),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns_with_missing": len(missing_summary),
        }

    def clean(
        self,
        df: pd.DataFrame,
        recommendations: Optional[List[Dict[str, Any]]] = None,
        auto_apply: bool = False,
        custom_strategies: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Clean DataFrame by applying imputation strategies.

        Args:
            df: DataFrame to clean
            recommendations: List of recommendations from analyze() method.
                           If None, will generate recommendations automatically.
            auto_apply: If True, automatically apply all AI recommendations.
                       If False, will use custom_strategies or skip.
            custom_strategies: Dictionary mapping column names to strategy names.
                             Overrides AI recommendations for specified columns.

        Returns:
            Cleaned DataFrame with imputed values

        Example:
            # Auto-apply AI recommendations
            cleaned_df = cleaner.clean(df, auto_apply=True)

            # Use custom strategies
            cleaned_df = cleaner.clean(df, custom_strategies={
                'age': 'median',
                'bmi': 'cohort_mean'
            })
        """
        validate_dataframe(df)
        df_cleaned = df.copy()

        # Get recommendations if not provided
        if recommendations is None and auto_apply:
            analysis = self.analyze(df)
            recommendations = analysis["recommendations"]
        elif recommendations is None and not auto_apply and custom_strategies is None:
            warnings.warn(
                "No recommendations or custom strategies provided. "
                "Set auto_apply=True or provide custom_strategies."
            )
            return df_cleaned

        # Apply imputation strategies
        if auto_apply and recommendations:
            print(f"Applying AI recommendations to {len(recommendations)} columns...")
            for rec in recommendations:
                if "error" in rec:
                    print(f"Skipping {rec['column']} due to error: {rec['error']}")
                    continue

                column = rec["column"]
                print(f"Imputing {column} using {rec['strategy']}...")

                try:
                    original_series = df_cleaned[column].copy()
                    imputed_series = ImputationEngine.apply_recommendation(
                        df_cleaned, rec
                    )
                    df_cleaned[column] = imputed_series

                    # Log transformation
                    report = ImputationEngine.get_imputation_report(
                        original_series, imputed_series
                    )
                    self._log_transformation(column, rec, report)

                except Exception as e:
                    print(f"Error imputing {column}: {str(e)}")
                    continue

        # Apply custom strategies
        if custom_strategies:
            print(f"Applying custom strategies to {len(custom_strategies)} columns...")
            for column, strategy in custom_strategies.items():
                if column not in df_cleaned.columns:
                    print(f"Column {column} not found in DataFrame. Skipping.")
                    continue

                print(f"Imputing {column} using {strategy}...")

                try:
                    original_series = df_cleaned[column].copy()
                    imputed_series = ImputationEngine.impute(
                        df_cleaned, column, strategy
                    )
                    df_cleaned[column] = imputed_series

                    # Log transformation
                    report = ImputationEngine.get_imputation_report(
                        original_series, imputed_series
                    )
                    self._log_transformation(
                        column,
                        {"strategy": strategy, "reasoning": "Custom user strategy"},
                        report,
                    )

                except Exception as e:
                    print(f"Error imputing {column}: {str(e)}")
                    continue

        return df_cleaned

    def get_recommendation(
        self, df: pd.DataFrame, column: str
    ) -> Dict[str, Any]:
        """
        Get AI recommendation for a specific column.

        Args:
            df: DataFrame
            column: Column name

        Returns:
            Recommendation dictionary
        """
        validate_dataframe(df)

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        return self.ai_advisor.analyze_missing_values(df, column)

    def apply_strategy(
        self,
        df: pd.DataFrame,
        column: str,
        strategy: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Apply a specific imputation strategy to a column.

        Args:
            df: DataFrame
            column: Column name
            strategy: Strategy name
            parameters: Optional strategy parameters

        Returns:
            DataFrame with imputed column
        """
        validate_dataframe(df)
        df_result = df.copy()

        original_series = df_result[column].copy()
        imputed_series = ImputationEngine.impute(
            df_result, column, strategy, parameters
        )
        df_result[column] = imputed_series

        # Log transformation
        report = ImputationEngine.get_imputation_report(
            original_series, imputed_series
        )
        self._log_transformation(
            column,
            {"strategy": strategy, "parameters": parameters or {}},
            report,
        )

        return df_result

    def get_transformation_log(self) -> List[Dict[str, Any]]:
        """
        Get log of all transformations applied.

        Returns:
            List of transformation records
        """
        return self.transformation_log

    def _log_transformation(
        self, column: str, recommendation: Dict[str, Any], report: Dict[str, Any]
    ) -> None:
        """Log a transformation for audit trail."""
        log_entry = {
            "column": column,
            "strategy": recommendation.get("strategy"),
            "reasoning": recommendation.get("reasoning", ""),
            "parameters": recommendation.get("parameters", {}),
            "report": report,
        }
        self.transformation_log.append(log_entry)

    def summary_report(self) -> str:
        """
        Generate a human-readable summary report of all transformations.

        Returns:
            Formatted string report
        """
        if not self.transformation_log:
            return "No transformations applied yet."

        report_lines = ["=" * 60, "Data Cleaning Summary Report", "=" * 60, ""]

        for i, entry in enumerate(self.transformation_log, 1):
            report_lines.append(f"{i}. Column: {entry['column']}")
            report_lines.append(f"   Strategy: {entry['strategy']}")
            report_lines.append(f"   Values Imputed: {entry['report']['values_imputed']}")

            if entry['reasoning']:
                report_lines.append(f"   Reasoning: {entry['reasoning']}")

            if entry['parameters']:
                report_lines.append(f"   Parameters: {entry['parameters']}")

            report_lines.append("")

        report_lines.append("=" * 60)
        return "\n".join(report_lines)
