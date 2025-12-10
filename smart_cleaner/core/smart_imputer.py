"""
Smart Cohort Imputation System.
Implements context-aware, cohort-based imputation strategies.

Example: For a 35-year-old female missing BMI, impute using mean BMI
of females aged 30-40, NOT the overall mean.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class SmartImputer:
    """
    Intelligent imputation system that uses cohort-based strategies.

    Key principle: Use context (other columns) to create cohorts,
    then impute within those cohorts for more accurate values.
    """

    def __init__(self):
        """Initialize the smart imputer."""
        self.imputation_log = []
        self.cohort_statistics = {}

    def impute_with_strategy(
        self,
        df: pd.DataFrame,
        column: str,
        strategy: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute a column using the specified strategy.

        Args:
            df: DataFrame with missing values
            column: Column to impute
            strategy: Strategy dictionary from AI advisor

        Returns:
            Tuple of (imputed DataFrame, imputation report)
        """
        df_result = df.copy()
        strategy_type = strategy.get('strategy', 'mean')

        report = {
            "column": column,
            "strategy": strategy_type,
            "before_missing": int(df[column].isnull().sum()),
            "sample_imputations": [],
        }

        if strategy_type in ['cohort_mean', 'cohort_median']:
            df_result, cohort_report = self._cohort_impute(
                df_result, column, strategy
            )
            report.update(cohort_report)

        elif strategy_type == 'mean':
            fill_value = df[column].mean()
            df_result[column] = df_result[column].fillna(fill_value)
            report["fill_value"] = float(fill_value)

        elif strategy_type == 'median':
            fill_value = df[column].median()
            df_result[column] = df_result[column].fillna(fill_value)
            report["fill_value"] = float(fill_value)

        elif strategy_type == 'mode':
            fill_value = df[column].mode().iloc[0] if len(df[column].mode()) > 0 else None
            if fill_value is not None:
                df_result[column] = df_result[column].fillna(fill_value)
            report["fill_value"] = str(fill_value)

        elif strategy_type == 'knn':
            df_result = self._knn_impute(df_result, column, strategy)

        elif strategy_type == 'forward_fill':
            df_result[column] = df_result[column].fillna(method='ffill')

        elif strategy_type == 'backward_fill':
            df_result[column] = df_result[column].fillna(method='bfill')

        report["after_missing"] = int(df_result[column].isnull().sum())
        report["values_imputed"] = report["before_missing"] - report["after_missing"]

        self.imputation_log.append(report)
        return df_result, report

    def _cohort_impute(
        self,
        df: pd.DataFrame,
        column: str,
        strategy: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform cohort-based imputation.

        This is the KEY feature - impute using statistics from similar records.

        Args:
            df: DataFrame
            column: Column to impute
            strategy: Strategy with cohort information

        Returns:
            Tuple of (imputed DataFrame, cohort report)
        """
        strategy_type = strategy.get('strategy', 'cohort_mean')
        cohort_columns = strategy.get('cohort_columns', [])
        cohort_bins = strategy.get('cohort_bins', {})
        fallback = strategy.get('fallback_strategy', 'median')

        report = {
            "cohort_columns": cohort_columns,
            "cohort_statistics": {},
            "sample_imputations": [],
        }

        if not cohort_columns:
            # No cohort columns specified, fall back to simple strategy
            if strategy_type == 'cohort_mean':
                fill_value = df[column].mean()
            else:
                fill_value = df[column].median()
            df[column] = df[column].fillna(fill_value)
            report["note"] = "No cohort columns, used overall statistic"
            return df, report

        # Create cohort column
        df_work = df.copy()
        cohort_col = self._create_cohort_column(df_work, cohort_columns, cohort_bins)
        df_work['_cohort_'] = cohort_col

        # Calculate statistics per cohort
        if strategy_type == 'cohort_mean':
            cohort_stats = df_work.groupby('_cohort_')[column].mean()
        else:
            cohort_stats = df_work.groupby('_cohort_')[column].median()

        # Store cohort statistics for reporting
        cohort_counts = df_work.groupby('_cohort_')[column].count()
        for cohort in cohort_stats.index:
            report["cohort_statistics"][str(cohort)] = {
                "statistic": float(cohort_stats[cohort]) if pd.notna(cohort_stats[cohort]) else None,
                "count": int(cohort_counts[cohort])
            }

        self.cohort_statistics[column] = report["cohort_statistics"]

        # Track sample imputations
        missing_mask = df_work[column].isnull()
        sample_indices = df_work[missing_mask].head(5).index

        # Impute missing values using cohort statistics
        for idx in df_work[missing_mask].index:
            cohort = df_work.loc[idx, '_cohort_']

            if pd.notna(cohort) and cohort in cohort_stats.index and pd.notna(cohort_stats[cohort]):
                fill_value = cohort_stats[cohort]
            else:
                # Fallback if cohort has no data
                if fallback == 'mean':
                    fill_value = df[column].mean()
                else:
                    fill_value = df[column].median()

            df_work.loc[idx, column] = fill_value

            # Record sample imputations
            if idx in sample_indices:
                report["sample_imputations"].append({
                    "row": int(idx),
                    "cohort": str(cohort),
                    "imputed_value": float(fill_value),
                    "cohort_info": self._get_cohort_description(
                        df_work.loc[idx], cohort_columns, cohort_bins
                    )
                })

        # Remove temp column and update original
        df[column] = df_work[column]

        return df, report

    def _create_cohort_column(
        self,
        df: pd.DataFrame,
        cohort_columns: List[str],
        cohort_bins: Dict[str, List]
    ) -> pd.Series:
        """
        Create a cohort identifier from multiple columns.

        Args:
            df: DataFrame
            cohort_columns: Columns to use for cohort
            cohort_bins: Bins for numeric columns

        Returns:
            Series of cohort identifiers
        """
        cohort_parts = []

        for col in cohort_columns:
            if col not in df.columns:
                continue

            if col in cohort_bins:
                # Bin numeric column
                bins = cohort_bins[col]
                labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
                binned = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
                cohort_parts.append(binned.astype(str))
            else:
                # Use as-is (categorical)
                cohort_parts.append(df[col].astype(str))

        if not cohort_parts:
            return pd.Series(['all'] * len(df), index=df.index)

        # Combine into single cohort identifier
        cohort = cohort_parts[0]
        for part in cohort_parts[1:]:
            cohort = cohort + '_' + part

        return cohort

    def _get_cohort_description(
        self,
        row: pd.Series,
        cohort_columns: List[str],
        cohort_bins: Dict[str, List]
    ) -> str:
        """Get human-readable cohort description for a row."""
        parts = []
        for col in cohort_columns:
            if col not in row.index:
                continue

            value = row[col]
            if col in cohort_bins:
                bins = cohort_bins[col]
                for i in range(len(bins) - 1):
                    if bins[i] <= value < bins[i+1]:
                        parts.append(f"{col}={bins[i]}-{bins[i+1]}")
                        break
            else:
                parts.append(f"{col}={value}")

        return ", ".join(parts)

    def _knn_impute(
        self,
        df: pd.DataFrame,
        column: str,
        strategy: Dict[str, Any]
    ) -> pd.DataFrame:
        """K-Nearest Neighbors imputation."""
        try:
            from sklearn.impute import KNNImputer

            k = strategy.get('parameters', {}).get('k_neighbors', 5)

            # Get numeric columns for KNN
            numeric_df = df.select_dtypes(include=[np.number])

            imputer = KNNImputer(n_neighbors=k)
            imputed = imputer.fit_transform(numeric_df)

            # Update only the target column
            col_idx = list(numeric_df.columns).index(column)
            df[column] = imputed[:, col_idx]

        except ImportError:
            # Fallback to median
            df[column] = df[column].fillna(df[column].median())

        return df

    def auto_impute(
        self,
        df: pd.DataFrame,
        strategies: Dict[str, Dict],
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Automatically impute all columns with missing values.

        Args:
            df: DataFrame with missing values
            strategies: Dictionary mapping columns to strategies
            target_column: Target column (skip imputation)

        Returns:
            Tuple of (imputed DataFrame, list of imputation reports)
        """
        df_result = df.copy()
        reports = []

        # Find columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if target_column and target_column in missing_cols:
            missing_cols.remove(target_column)

        for col in missing_cols:
            if col in strategies:
                strategy = strategies[col]
            else:
                # Default strategy
                if pd.api.types.is_numeric_dtype(df[col]):
                    strategy = {"strategy": "median"}
                else:
                    strategy = {"strategy": "mode"}

            df_result, report = self.impute_with_strategy(df_result, col, strategy)
            reports.append(report)

        return df_result, reports

    def get_imputation_summary(self) -> str:
        """Get summary of all imputations performed."""
        lines = []
        lines.append("=" * 60)
        lines.append("IMPUTATION SUMMARY")
        lines.append("=" * 60)

        for log in self.imputation_log:
            lines.append(f"\nColumn: {log['column']}")
            lines.append(f"  Strategy: {log['strategy']}")
            lines.append(f"  Values imputed: {log['values_imputed']}")

            if 'cohort_columns' in log:
                lines.append(f"  Cohort columns: {log['cohort_columns']}")

            if 'sample_imputations' in log and log['sample_imputations']:
                lines.append("  Sample imputations:")
                for sample in log['sample_imputations'][:3]:
                    lines.append(f"    - Row {sample['row']}: {sample.get('cohort_info', 'N/A')} -> {sample['imputed_value']:.4f}")

        return "\n".join(lines)

    @classmethod
    def create_default_strategies(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Create default cohort-based strategies.

        Automatically identifies good cohort columns and creates strategies.

        Args:
            df: DataFrame to analyze
            target_column: Target column (excluded)

        Returns:
            Dictionary of strategies for each column with missing values
        """
        strategies = {}

        # Find potential cohort columns (categorical with few values, complete data)
        cohort_candidates = []
        for col in df.columns:
            if col == target_column:
                continue

            missing_pct = df[col].isnull().sum() / len(df)
            n_unique = df[col].nunique()

            if missing_pct < 0.05 and n_unique <= 10:
                cohort_candidates.append({
                    "column": col,
                    "type": "categorical",
                    "n_unique": n_unique
                })
            elif missing_pct < 0.05 and pd.api.types.is_numeric_dtype(df[col]):
                cohort_candidates.append({
                    "column": col,
                    "type": "numeric",
                    "range": [float(df[col].min()), float(df[col].max())]
                })

        # Find age-like columns for binning
        age_cols = [c for c in df.columns if 'age' in c.lower()]
        gender_cols = [c for c in df.columns if any(g in c.lower() for g in ['gender', 'sex'])]

        # Create strategies for columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if target_column and target_column in missing_cols:
            missing_cols.remove(target_column)

        for col in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Build cohort strategy for numeric columns
                cohort_columns = []
                cohort_bins = {}

                # Add gender if available
                if gender_cols:
                    cohort_columns.extend(gender_cols)

                # Add age bins if available
                if age_cols:
                    cohort_columns.extend(age_cols)
                    for age_col in age_cols:
                        cohort_bins[age_col] = [0, 20, 30, 40, 50, 60, 70, 80, 100, 150]

                if cohort_columns:
                    strategies[col] = {
                        "strategy": "cohort_mean",
                        "cohort_columns": cohort_columns[:2],  # Max 2 cohort columns
                        "cohort_bins": cohort_bins,
                        "fallback_strategy": "median",
                        "reasoning": f"Cohort-based mean using {cohort_columns[:2]}"
                    }
                else:
                    strategies[col] = {
                        "strategy": "median",
                        "reasoning": "Median imputation (no good cohort columns found)"
                    }
            else:
                # Categorical column
                strategies[col] = {
                    "strategy": "mode",
                    "reasoning": "Mode imputation for categorical"
                }

        return strategies
