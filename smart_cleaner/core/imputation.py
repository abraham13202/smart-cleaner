"""
Imputation strategies for missing values.
Includes both traditional and AI-powered intelligent imputation methods.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class ImputationStrategy(ABC):
    """Base class for imputation strategies."""

    @abstractmethod
    def impute(self, df: pd.DataFrame, column: str, **kwargs) -> pd.Series:
        """
        Impute missing values in a column.

        Args:
            df: DataFrame containing the column
            column: Name of column to impute
            **kwargs: Additional parameters for the strategy

        Returns:
            Series with imputed values
        """
        pass


class MeanImputation(ImputationStrategy):
    """Impute missing values with column mean."""

    def impute(self, df: pd.DataFrame, column: str, **kwargs) -> pd.Series:
        """Impute with mean value."""
        mean_value = df[column].mean()
        return df[column].fillna(mean_value)


class MedianImputation(ImputationStrategy):
    """Impute missing values with column median."""

    def impute(self, df: pd.DataFrame, column: str, **kwargs) -> pd.Series:
        """Impute with median value."""
        median_value = df[column].median()
        return df[column].fillna(median_value)


class ModeImputation(ImputationStrategy):
    """Impute missing values with column mode."""

    def impute(self, df: pd.DataFrame, column: str, **kwargs) -> pd.Series:
        """Impute with mode value."""
        mode_value = df[column].mode()
        if len(mode_value) > 0:
            return df[column].fillna(mode_value[0])
        return df[column]


class ForwardFillImputation(ImputationStrategy):
    """Forward fill missing values."""

    def impute(self, df: pd.DataFrame, column: str, **kwargs) -> pd.Series:
        """Impute with forward fill."""
        return df[column].fillna(method='ffill')


class BackwardFillImputation(ImputationStrategy):
    """Backward fill missing values."""

    def impute(self, df: pd.DataFrame, column: str, **kwargs) -> pd.Series:
        """Impute with backward fill."""
        return df[column].fillna(method='bfill')


class CohortMeanImputation(ImputationStrategy):
    """
    Context-aware imputation using cohort means.

    SMART IMPUTATION: Instead of using overall mean, uses mean of similar groups.

    Examples:
    - BMI missing for 34-year-old female → mean BMI of females aged 30-40
    - Salary missing for Senior Engineer in NYC → mean salary of Senior Engineers in NYC
    - Price missing for 2BR apartment in Manhattan → mean price of 2BR in Manhattan

    Supports:
    - Single cohort column (e.g., gender)
    - Multiple cohort columns (e.g., gender + age_group)
    - Numeric columns with binning (e.g., age → age groups)
    """

    def impute(self, df: pd.DataFrame, column: str, **kwargs) -> pd.Series:
        """
        Impute with cohort-based mean (CONTEXT-AWARE).

        Args:
            df: DataFrame
            column: Column to impute
            cohort_columns: List of columns to use for grouping, e.g., ['gender', 'age']
            cohort_column: Single column (legacy support)
            cohort_bins: Dict mapping column names to bin edges
                         e.g., {'age': [0, 30, 50, 70, 100]}

        Returns:
            Series with imputed values
        """
        # Support both single column (legacy) and multiple columns
        cohort_columns = kwargs.get('cohort_columns', [])
        if not cohort_columns:
            single_col = kwargs.get('cohort_column')
            if single_col:
                cohort_columns = [single_col]

        cohort_bins = kwargs.get('cohort_bins', {})

        # Validate cohort columns exist
        valid_cohort_cols = [c for c in cohort_columns if c in df.columns]

        if not valid_cohort_cols:
            # Fallback to simple mean
            return MeanImputation().impute(df, column)

        # Create a working copy to avoid modifying original
        df_work = df.copy()
        result = df[column].copy()
        temp_cols = []

        # Process each cohort column
        groupby_cols = []
        for cohort_col in valid_cohort_cols:
            if cohort_col in cohort_bins:
                # Bin the numeric column
                bins = cohort_bins[cohort_col]
                temp_col = f'_cohort_{cohort_col}'
                temp_cols.append(temp_col)

                labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
                df_work[temp_col] = pd.cut(
                    df_work[cohort_col],
                    bins=bins,
                    labels=labels,
                    include_lowest=True
                )
                groupby_cols.append(temp_col)
            else:
                # Use column directly (categorical)
                groupby_cols.append(cohort_col)

        # Calculate mean for each cohort combination
        try:
            cohort_means = df_work.groupby(groupby_cols, observed=True)[column].mean()

            # Impute based on cohort
            for idx, row in df_work[df[column].isna()].iterrows():
                try:
                    # Get the cohort key for this row
                    if len(groupby_cols) == 1:
                        cohort_key = row[groupby_cols[0]]
                    else:
                        cohort_key = tuple(row[col] for col in groupby_cols)

                    # Look up the cohort mean
                    if cohort_key in cohort_means.index:
                        result.loc[idx] = cohort_means[cohort_key]
                    elif len(groupby_cols) == 1 and pd.notna(cohort_key):
                        # Try direct lookup for single column
                        if cohort_key in cohort_means.index:
                            result.loc[idx] = cohort_means[cohort_key]
                except (KeyError, TypeError):
                    # This cohort combination doesn't exist, will use fallback
                    pass

        except Exception as e:
            # If groupby fails, fall back to simple imputation
            print(f"    Cohort grouping failed: {e}. Using simple mean.")
            return MeanImputation().impute(df, column)

        # Fill any remaining NaN with overall mean (edge cases)
        if result.isna().any():
            overall_mean = df[column].mean()
            remaining = result.isna().sum()
            result.fillna(overall_mean, inplace=True)
            if remaining > 0:
                print(f"    {remaining} values filled with overall mean (no matching cohort)")

        return result


class KNNImputation(ImputationStrategy):
    """
    Impute missing values using K-Nearest Neighbors.
    Uses sklearn's KNNImputer if available.
    """

    def impute(self, df: pd.DataFrame, column: str, **kwargs) -> pd.Series:
        """
        Impute with KNN.

        Args:
            df: DataFrame
            column: Column to impute
            k_neighbors: Number of neighbors to use (default: 5)

        Returns:
            Series with imputed values
        """
        try:
            from sklearn.impute import KNNImputer
        except ImportError:
            # Fallback if sklearn not available
            print("sklearn not installed. Falling back to mean imputation.")
            return MeanImputation().impute(df, column)

        k_neighbors = kwargs.get('k_neighbors', 5)

        # Select only numeric columns for KNN
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if column not in numeric_cols:
            return ModeImputation().impute(df, column)

        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=k_neighbors)
        imputed_data = imputer.fit_transform(df[numeric_cols])

        # Get the column index
        col_idx = numeric_cols.index(column)

        return pd.Series(imputed_data[:, col_idx], index=df.index)


class ImputationEngine:
    """
    Main engine for applying imputation strategies.
    Maps strategy names to implementations.
    """

    STRATEGIES = {
        "mean": MeanImputation(),
        "median": MedianImputation(),
        "mode": ModeImputation(),
        "forward_fill": ForwardFillImputation(),
        "backward_fill": BackwardFillImputation(),
        "cohort_mean": CohortMeanImputation(),
        "knn": KNNImputation(),
        "drop": ModeImputation(),  # Fallback to mode instead of actually dropping
    }

    @classmethod
    def impute(
        cls,
        df: pd.DataFrame,
        column: str,
        strategy: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> pd.Series:
        """
        Apply imputation strategy to a column.

        Args:
            df: DataFrame containing the column
            column: Column name to impute
            strategy: Strategy name ('mean', 'median', 'cohort_mean', etc.)
            parameters: Additional parameters for the strategy

        Returns:
            Series with imputed values

        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy not in cls.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available strategies: {list(cls.STRATEGIES.keys())}"
            )

        strategy_impl = cls.STRATEGIES[strategy]
        params = parameters or {}

        return strategy_impl.impute(df, column, **params)

    @classmethod
    def apply_recommendation(
        cls, df: pd.DataFrame, recommendation: Dict[str, Any]
    ) -> pd.Series:
        """
        Apply an AI-recommended imputation strategy.

        Args:
            df: DataFrame
            recommendation: Recommendation dictionary from AIAdvisor

        Returns:
            Series with imputed values
        """
        column = recommendation["column"]
        strategy = recommendation["strategy"]
        parameters = recommendation.get("parameters", {})

        return cls.impute(df, column, strategy, parameters)

    @classmethod
    def get_imputation_report(
        cls, original: pd.Series, imputed: pd.Series
    ) -> Dict[str, Any]:
        """
        Generate a report comparing original and imputed data.

        Args:
            original: Original series with missing values
            imputed: Imputed series

        Returns:
            Dictionary with comparison statistics
        """
        num_imputed = original.isna().sum()

        report = {
            "total_values": len(original),
            "missing_before": int(num_imputed),
            "missing_after": int(imputed.isna().sum()),
            "values_imputed": int(num_imputed - imputed.isna().sum()),
        }

        # Add statistics if numeric
        if pd.api.types.is_numeric_dtype(imputed):
            report.update({
                "mean_before": float(original.mean()) if not original.isna().all() else None,
                "mean_after": float(imputed.mean()) if not imputed.isna().all() else None,
                "std_before": float(original.std()) if not original.isna().all() else None,
                "std_after": float(imputed.std()) if not imputed.isna().all() else None,
            })

        return report
