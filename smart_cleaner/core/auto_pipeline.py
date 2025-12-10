"""
Comprehensive automatic data preprocessing pipeline.
Orchestrates all cleaning, validation, and analysis steps.
"""

import os
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass, field

from .duplicates import DuplicateHandler
from .health_validator import HealthValidator
from .outliers import OutlierHandler
from .visualizer import DataVisualizer
from .ai_advisor import AIAdvisor
from .ai_advisor_gemini import GeminiAdvisor
from .ai_advisor_ollama import OllamaAdvisor
from .imputation import ImputationEngine
from ..utils.config import Config


@dataclass
class PipelineConfig:
    """Configuration for auto preprocessing pipeline."""

    # Duplicate handling
    remove_duplicates: bool = True
    duplicate_exclude_columns: Optional[List[str]] = None

    # Health validation
    validate_health_metrics: bool = True
    health_fix_strategy: str = "cap"  # 'cap', 'remove', 'set_nan'

    # Outlier handling
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # 'iqr', 'zscore', 'isolation_forest'
    outlier_strategy: str = "cap"  # 'cap', 'remove', 'impute_median', 'impute_mean'
    outlier_exclude_columns: Optional[List[str]] = None

    # Missing value imputation
    impute_missing: bool = True
    use_ai_recommendations: bool = False  # Set to True for AI-powered imputation
    ai_provider: str = "ollama"  # 'ollama' (LOCAL - DEFAULT), 'gemini', or 'claude'

    # Ollama settings (local LLM - no API key needed!)
    ollama_model: str = "llama3.2"  # e.g., 'llama3.2', 'mistral', 'codellama'
    ollama_base_url: Optional[str] = None  # Default: http://localhost:11434

    # Visualization
    generate_visualizations: bool = True
    visualization_output_dir: str = "./data_visualizations"

    # Target analysis
    target_column: Optional[str] = None

    # API keys for AI recommendations
    anthropic_api_key: Optional[str] = None  # For Claude
    gemini_api_key: Optional[str] = None  # For Gemini (FREE!)

    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.ollama_base_url:
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        env_ollama_model = os.getenv("OLLAMA_MODEL")
        if env_ollama_model:
            self.ollama_model = env_ollama_model


class AutoPreprocessor:
    """
    Comprehensive automatic data preprocessing pipeline.
    Handles everything from duplicates to missing values to visualizations.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize auto preprocessor.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.report: Dict[str, Any] = {
            "steps_completed": [],
            "steps_skipped": [],
            "errors": [],
        }

    def process(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete preprocessing pipeline.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (processed_df, comprehensive_report)
        """
        print("=" * 80)
        print("AUTOMATIC DATA PREPROCESSING PIPELINE")
        print("=" * 80)
        print()

        df_processed = df.copy()
        self.report = {
            "original_shape": df.shape,
            "original_memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "steps": [],
        }

        # Step 1: Initial data profile
        print("Step 1: Generating initial data profile...")
        self._add_step("initial_profile", self._step_initial_profile(df_processed))

        # Step 2: Remove duplicates
        if self.config.remove_duplicates:
            print("Step 2: Removing duplicate rows...")
            df_processed, step_report = self._step_remove_duplicates(df_processed)
            self._add_step("remove_duplicates", step_report)
        else:
            print("Step 2: Skipping duplicate removal (disabled)")
            self._skip_step("remove_duplicates")

        # Step 3: Validate health metrics
        if self.config.validate_health_metrics:
            print("Step 3: Validating health metrics...")
            df_processed, step_report = self._step_validate_health(df_processed)
            self._add_step("validate_health", step_report)
        else:
            print("Step 3: Skipping health validation (disabled)")
            self._skip_step("validate_health")

        # Step 4: Handle outliers
        if self.config.handle_outliers:
            print("Step 4: Detecting and handling outliers...")
            df_processed, step_report = self._step_handle_outliers(df_processed)
            self._add_step("handle_outliers", step_report)
        else:
            print("Step 4: Skipping outlier handling (disabled)")
            self._skip_step("handle_outliers")

        # Step 5: Impute missing values
        if self.config.impute_missing:
            print("Step 5: Imputing missing values...")
            df_processed, step_report = self._step_impute_missing(df_processed)
            self._add_step("impute_missing", step_report)
        else:
            print("Step 5: Skipping missing value imputation (disabled)")
            self._skip_step("impute_missing")

        # Step 6: Target analysis (if target specified)
        if self.config.target_column:
            print(f"Step 6: Analyzing target variable '{self.config.target_column}'...")
            step_report = self._step_analyze_target(df_processed)
            self._add_step("target_analysis", step_report)

        # Step 7: Generate visualizations
        if self.config.generate_visualizations:
            print("Step 7: Generating visualizations...")
            step_report = self._step_visualize(df_processed)
            self._add_step("visualizations", step_report)
        else:
            print("Step 7: Skipping visualizations (disabled)")
            self._skip_step("visualizations")

        # Final summary
        self.report["final_shape"] = df_processed.shape
        self.report["final_memory_mb"] = df_processed.memory_usage(deep=True).sum() / 1024 / 1024
        self.report["rows_removed"] = df.shape[0] - df_processed.shape[0]
        self.report["data_quality_improvement"] = self._calculate_quality_improvement(df, df_processed)

        print()
        print("=" * 80)
        print("PIPELINE COMPLETED")
        print("=" * 80)
        print(f"Original shape: {df.shape}")
        print(f"Final shape: {df_processed.shape}")
        print(f"Rows removed: {self.report['rows_removed']}")
        print()

        return df_processed, self.report

    def _step_initial_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate initial data profile."""
        profile_text = DataVisualizer.generate_data_profile(df)
        print(profile_text)

        return {
            "profile": profile_text,
            "total_rows": len(df),
            "total_columns": len(df.columns),
        }

    def _step_remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove duplicate rows."""
        df_cleaned, report = DuplicateHandler.auto_remove_duplicates(
            df,
            exclude_columns=self.config.duplicate_exclude_columns,
        )

        print(f"  Duplicates found: {report['duplicate_count']}")
        print(f"  Rows removed: {report['rows_removed']}")

        return df_cleaned, report

    def _step_validate_health(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate health metrics."""
        df_cleaned, validation_report = HealthValidator.auto_validate_all(
            df,
            fix_strategy=self.config.health_fix_strategy,
        )

        if validation_report:
            print(f"  Health metrics validated: {len(validation_report)} columns")
            for item in validation_report:
                if "invalid_count" in item:
                    print(f"    {item['column']}: {item['invalid_count']} invalid values fixed")

        return df_cleaned, {"validations": validation_report}

    def _step_handle_outliers(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outliers."""
        df_cleaned, outlier_report = OutlierHandler.auto_handle_all_outliers(
            df,
            exclude_columns=self.config.outlier_exclude_columns,
            method=self.config.outlier_method,
            strategy=self.config.outlier_strategy,
        )

        if outlier_report:
            print(f"  Outliers detected in {len(outlier_report)} columns")
            for item in outlier_report:
                if "outliers_detected" in item:
                    print(f"    {item['column']}: {item['outliers_detected']} outliers ({item['percentage']:.1f}%)")

        return df_cleaned, {"outliers": outlier_report}

    def _step_impute_missing(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute missing values."""
        missing_summary = df.isnull().sum()
        cols_with_missing = missing_summary[missing_summary > 0].index.tolist()

        if len(cols_with_missing) == 0:
            print("  No missing values to impute")
            return df, {"status": "no_missing_values"}

        print(f"  Columns with missing values: {len(cols_with_missing)}")

        df_cleaned = df.copy()
        imputation_report = []

        if self.config.use_ai_recommendations:
            # Determine which AI provider to use
            api_key = None
            provider = self.config.ai_provider.lower()

            if provider == "ollama":
                # Ollama is local - no API key needed!
                print(f"  Using AI-powered recommendations (Ollama - LOCAL!)...")
            elif provider == "gemini" and self.config.gemini_api_key:
                api_key = self.config.gemini_api_key
                print(f"  Using AI-powered recommendations (Gemini - FREE!)...")
            elif provider == "claude" and self.config.anthropic_api_key:
                api_key = self.config.anthropic_api_key
                print(f"  Using AI-powered recommendations (Claude)...")
            else:
                print(f"  AI enabled but no API key for {provider}. Using simple strategies...")
                self.config.use_ai_recommendations = False

            if self.config.use_ai_recommendations:
                try:
                    # Initialize the appropriate advisor
                    if provider == "ollama":
                        advisor = OllamaAdvisor(
                            model=self.config.ollama_model,
                            base_url=self.config.ollama_base_url,
                        )
                    elif provider == "gemini":
                        advisor = GeminiAdvisor(api_key=api_key)
                    else:  # claude
                        config = Config(anthropic_api_key=api_key)
                        advisor = AIAdvisor(config)

                    for column in cols_with_missing:
                        try:
                            print(f"    Analyzing {column}...", end=" ")
                            rec = advisor.analyze_missing_values(df_cleaned, column)

                            # Debug: Show what AI recommended
                            strategy = rec.get('strategy', 'unknown')
                            confidence = rec.get('confidence', 'N/A')
                            reasoning = rec.get('reasoning', 'No reasoning provided')[:80]

                            print(f"{strategy} (confidence: {confidence})")
                            print(f"       Reasoning: {reasoning}...")

                            original_series = df_cleaned[column].copy()
                            imputed_series = ImputationEngine.apply_recommendation(df_cleaned, rec)
                            df_cleaned[column] = imputed_series

                            imputation_report.append({
                                "column": column,
                                "strategy": strategy,
                                "reasoning": reasoning,
                                "confidence": confidence,
                            })

                        except Exception as e:
                            print(f"    {column}: Error - {str(e)}, using mean/mode")
                            imputation_report.append({
                                "column": column,
                                "error": str(e),
                            })

                except Exception as e:
                    print(f"  AI recommendations failed: {str(e)}")
                    print("  Falling back to simple strategies...")
                    self.config.use_ai_recommendations = False

        if not self.config.use_ai_recommendations:
            print("  Using simple imputation strategies...")
            for column in cols_with_missing:
                if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                    df_cleaned[column] = ImputationEngine.impute(df_cleaned, column, "mean")
                    imputation_report.append({"column": column, "strategy": "mean"})
                    print(f"    {column}: mean")
                else:
                    df_cleaned[column] = ImputationEngine.impute(df_cleaned, column, "mode")
                    imputation_report.append({"column": column, "strategy": "mode"})
                    print(f"    {column}: mode")

        return df_cleaned, {"imputations": imputation_report}

    def _step_analyze_target(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze target variable."""
        if self.config.target_column not in df.columns:
            error = f"Target column '{self.config.target_column}' not found"
            print(f"  Error: {error}")
            return {"error": error}

        analysis_text = DataVisualizer.analyze_target_relationships(
            df,
            self.config.target_column,
        )

        print(analysis_text)

        return {
            "target_column": self.config.target_column,
            "analysis": analysis_text,
        }

    def _step_visualize(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate visualizations."""
        viz_result = DataVisualizer.create_visualizations(
            df,
            target_column=self.config.target_column,
            output_dir=self.config.visualization_output_dir,
        )

        if viz_result["status"] == "success":
            print(f"  Visualizations saved to: {viz_result['output_directory']}")
            print(f"  Plots created: {len(viz_result['plots_created'])}")
            for plot in viz_result['plots_created']:
                print(f"    - {plot}")
        else:
            print(f"  {viz_result['reason']}")
            if "recommendation" in viz_result:
                print(f"  {viz_result['recommendation']}")

        return viz_result

    def _add_step(self, step_name: str, step_report: Dict[str, Any]) -> None:
        """Add step to report."""
        self.report["steps"].append({
            "name": step_name,
            "status": "completed",
            "report": step_report,
        })

    def _skip_step(self, step_name: str) -> None:
        """Mark step as skipped."""
        self.report["steps"].append({
            "name": step_name,
            "status": "skipped",
        })

    def _calculate_quality_improvement(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall data quality improvement."""
        # Completeness
        before_missing = df_before.isnull().sum().sum()
        after_missing = df_after.isnull().sum().sum()

        before_cells = df_before.shape[0] * df_before.shape[1]
        after_cells = df_after.shape[0] * df_after.shape[1]

        before_completeness = ((before_cells - before_missing) / before_cells * 100) if before_cells > 0 else 0
        after_completeness = ((after_cells - after_missing) / after_cells * 100) if after_cells > 0 else 0

        return {
            "completeness_before": round(before_completeness, 2),
            "completeness_after": round(after_completeness, 2),
            "completeness_improvement": round(after_completeness - before_completeness, 2),
            "missing_values_before": int(before_missing),
            "missing_values_after": int(after_missing),
        }

    def get_summary_report(self) -> str:
        """Get human-readable summary report."""
        lines = []
        lines.append("=" * 80)
        lines.append("DATA PREPROCESSING SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"Original shape: {self.report.get('original_shape')}")
        lines.append(f"Final shape: {self.report.get('final_shape')}")
        lines.append(f"Rows removed: {self.report.get('rows_removed', 0)}")
        lines.append("")

        quality = self.report.get("data_quality_improvement", {})
        lines.append("Data Quality:")
        lines.append(f"  Completeness before: {quality.get('completeness_before', 0):.2f}%")
        lines.append(f"  Completeness after: {quality.get('completeness_after', 0):.2f}%")
        lines.append(f"  Improvement: +{quality.get('completeness_improvement', 0):.2f}%")
        lines.append("")

        lines.append("Steps Completed:")
        for step in self.report.get("steps", []):
            status_icon = "✓" if step["status"] == "completed" else "○"
            lines.append(f"  {status_icon} {step['name']}")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)
