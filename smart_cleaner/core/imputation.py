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
    Impute missing values based on cohort means.

    Example: For BMI missing values, use mean BMI of people in same age group.
    """

    def impute(self, df: pd.DataFrame, column: str, **kwargs) -> pd.Series:
        """
        Impute with cohort-based mean.

        Args:
            df: DataFrame
            column: Column to impute
            cohort_column: Column to use for cohort grouping (e.g., 'age')
            cohort_bins: List of bin edges for cohort groups
                         e.g., [0, 20, 30, 40, 50, 100] for age groups

        Returns:
            Series with imputed values
        """
        cohort_column = kwargs.get('cohort_column')
        cohort_bins = kwargs.get('cohort_bins')

        if not cohort_column or cohort_column not in df.columns:
            # Fallback to simple mean
            return MeanImputation().impute(df, column)

        result = df[column].copy()

        if cohort_bins:
            # Create cohort groups based on bins
            cohort_labels = [f"{cohort_bins[i]}-{cohort_bins[i+1]}"
                           for i in range(len(cohort_bins) - 1)]
            df['_temp_cohort'] = pd.cut(
                df[cohort_column],
                bins=cohort_bins,
                labels=cohort_labels,
                include_lowest=True
            )

            # Calculate mean for each cohort
            cohort_means = df.groupby('_temp_cohort')[column].mean()

            # Impute based on cohort
            for cohort_value in cohort_means.index:
                mask = (df['_temp_cohort'] == cohort_value) & (df[column].isna())
                result.loc[mask] = cohort_means[cohort_value]

            # Clean up temporary column
            df.drop('_temp_cohort', axis=1, inplace=True)
        else:
            # Use cohort column values directly (for categorical cohorts)
            cohort_means = df.groupby(cohort_column)[column].mean()

            for cohort_value in cohort_means.index:
                mask = (df[cohort_column] == cohort_value) & (df[column].isna())
                result.loc[mask] = cohort_means[cohort_value]

        # Fill any remaining NaN with overall mean (edge cases)
        if result.isna().any():
            result.fillna(df[column].mean(), inplace=True)

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
