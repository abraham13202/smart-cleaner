"""
Outlier detection and automatic handling.
"""

from typing import Dict, List, Any, Optional, Tuple, Literal
import pandas as pd
import numpy as np


class OutlierHandler:
    """
    Automatic outlier detection and handling.
    Supports multiple detection methods and handling strategies.
    """

    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        column: str,
        method: Literal["iqr", "zscore", "isolation_forest"] = "iqr",
        threshold: Optional[float] = None,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Detect outliers in a numeric column.

        Args:
            df: DataFrame
            column: Column name
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Custom threshold (default: 1.5 for IQR, 3 for z-score)

        Returns:
            Tuple of (outlier_mask, detection_info)
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' is not numeric")

        series = df[column].dropna()

        if method == "iqr":
            threshold = threshold or 1.5
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)

            info = {
                "method": "iqr",
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "threshold": threshold,
            }

        elif method == "zscore":
            threshold = threshold or 3.0
            mean = series.mean()
            std = series.std()

            if std == 0:
                outlier_mask = pd.Series([False] * len(df), index=df.index)
            else:
                z_scores = np.abs((df[column] - mean) / std)
                outlier_mask = z_scores > threshold

            info = {
                "method": "zscore",
                "mean": float(mean),
                "std": float(std),
                "threshold": threshold,
            }

        elif method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest
            except ImportError:
                raise ImportError(
                    "scikit-learn required for isolation_forest method. "
                    "Install with: pip install scikit-learn"
                )

            # Remove NaN for fitting
            valid_data = df[[column]].dropna()

            iso_forest = IsolationForest(contamination=threshold or 0.1, random_state=42)
            predictions = iso_forest.fit_predict(valid_data)

            # Create full mask
            outlier_mask = pd.Series([False] * len(df), index=df.index)
            outlier_mask.loc[valid_data.index] = predictions == -1

            info = {
                "method": "isolation_forest",
                "contamination": threshold or 0.1,
            }

        else:
            raise ValueError(f"Unknown method: {method}")

        info["outlier_count"] = int(outlier_mask.sum())
        info["outlier_percentage"] = float(outlier_mask.sum() / len(df) * 100)

        return outlier_mask, info

    @staticmethod
    def handle_outliers(
        df: pd.DataFrame,
        column: str,
        outlier_mask: pd.Series,
        strategy: Literal["remove", "cap", "impute_median", "impute_mean"] = "cap",
        info: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Handle detected outliers.

        Args:
            df: DataFrame
            column: Column name
            outlier_mask: Boolean mask indicating outliers
            strategy: How to handle outliers
            info: Detection info from detect_outliers (needed for 'cap' strategy)

        Returns:
            DataFrame with outliers handled
        """
        df_result = df.copy()

        if strategy == "remove":
            # Remove rows with outliers
            df_result = df_result[~outlier_mask]

        elif strategy == "cap":
            # Cap outliers at bounds
            if info is None or "lower_bound" not in info:
                raise ValueError("'cap' strategy requires detection info with bounds")

            lower_bound = info["lower_bound"]
            upper_bound = info["upper_bound"]

            df_result.loc[outlier_mask & (df_result[column] < lower_bound), column] = lower_bound
            df_result.loc[outlier_mask & (df_result[column] > upper_bound), column] = upper_bound

        elif strategy == "impute_median":
            # Replace outliers with median
            median_value = df[column].median()
            df_result.loc[outlier_mask, column] = median_value

        elif strategy == "impute_mean":
            # Replace outliers with mean
            mean_value = df[column].mean()
            df_result.loc[outlier_mask, column] = mean_value

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return df_result

    @staticmethod
    def auto_handle_all_outliers(
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
        method: str = "iqr",
        strategy: str = "cap",
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Automatically detect and handle outliers in all numeric columns.

        Args:
            df: DataFrame
            exclude_columns: Columns to skip
            method: Detection method
            strategy: Handling strategy

        Returns:
            Tuple of (cleaned_df, outlier_report)
        """
        exclude_columns = exclude_columns or []
        df_result = df.copy()
        outlier_report = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_process = [col for col in numeric_cols if col not in exclude_columns]

        for column in cols_to_process:
            try:
                # Detect outliers
                outlier_mask, info = OutlierHandler.detect_outliers(
                    df_result, column, method=method
                )

                if info["outlier_count"] > 0:
                    # Handle outliers
                    df_result = OutlierHandler.handle_outliers(
                        df_result, column, outlier_mask, strategy=strategy, info=info
                    )

                    outlier_report.append({
                        "column": column,
                        "outliers_detected": info["outlier_count"],
                        "percentage": info["outlier_percentage"],
                        "method": method,
                        "strategy": strategy,
                        "bounds": {
                            "lower": info.get("lower_bound"),
                            "upper": info.get("upper_bound"),
                        },
                    })

            except Exception as e:
                outlier_report.append({
                    "column": column,
                    "error": str(e),
                })

        return df_result, outlier_report
