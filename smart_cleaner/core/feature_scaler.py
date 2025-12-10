"""
Feature Scaling module for numerical variables.
Handles standardization, normalization, log transforms, and more.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from scipy import stats


class FeatureScaler:
    """
    Automated feature scaling for numerical variables.
    Chooses appropriate scaling based on distribution analysis.
    """

    def __init__(self):
        """Initialize scaler with storage for fitted parameters."""
        self.scaling_params = {}
        self.scaling_report = {}

    def auto_scale(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        method: str = 'auto',
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Automatically scale numerical columns.

        Strategy for 'auto' method:
        - Highly skewed data (|skewness| > 1): Log transform first, then standardize
        - Normal-ish data: StandardScaler (z-score)
        - Bounded data (0-1 or known range): MinMax scaling

        Args:
            df: Input DataFrame
            columns: Columns to scale (default: all numeric)
            target_column: Target variable (excluded from scaling)
            method: 'auto', 'standard', 'minmax', 'robust', 'log'

        Returns:
            Tuple of (scaled DataFrame, scaling report)
        """
        df_result = df.copy()
        report = {
            "standardized": [],
            "minmax_scaled": [],
            "robust_scaled": [],
            "log_transformed": [],
            "skipped": [],
            "scaling_params": {},
        }

        # Get numeric columns
        if columns is None:
            columns = df_result.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target column
        if target_column and target_column in columns:
            columns.remove(target_column)

        for col in columns:
            # Skip if too many nulls
            if df_result[col].isnull().sum() / len(df_result) > 0.5:
                report["skipped"].append({
                    "column": col,
                    "reason": "more than 50% null values"
                })
                continue

            # Analyze distribution
            col_data = df_result[col].dropna()

            if len(col_data) < 10:
                report["skipped"].append({
                    "column": col,
                    "reason": "insufficient data points"
                })
                continue

            skewness = col_data.skew()
            min_val = col_data.min()
            max_val = col_data.max()

            # Choose method
            if method == 'auto':
                chosen_method = self._choose_scaling_method(
                    col_data, skewness, min_val, max_val
                )
            else:
                chosen_method = method

            # Apply scaling
            if chosen_method == 'log':
                df_result, params = self._log_transform(df_result, col)
                report["log_transformed"].append({
                    "column": col,
                    "original_skewness": skewness,
                    "params": params,
                })

            elif chosen_method == 'standard':
                df_result, params = self._standardize(df_result, col)
                report["standardized"].append({
                    "column": col,
                    "mean": params["mean"],
                    "std": params["std"],
                })

            elif chosen_method == 'minmax':
                df_result, params = self._minmax_scale(df_result, col)
                report["minmax_scaled"].append({
                    "column": col,
                    "min": params["min"],
                    "max": params["max"],
                })

            elif chosen_method == 'robust':
                df_result, params = self._robust_scale(df_result, col)
                report["robust_scaled"].append({
                    "column": col,
                    "median": params["median"],
                    "iqr": params["iqr"],
                })

            report["scaling_params"][col] = {
                "method": chosen_method,
                "params": params,
            }

        self.scaling_params = report["scaling_params"]
        self.scaling_report = report
        return df_result, report

    def _choose_scaling_method(
        self,
        data: pd.Series,
        skewness: float,
        min_val: float,
        max_val: float
    ) -> str:
        """Choose the best scaling method based on data characteristics."""
        # Check for highly skewed data with positive values (good for log)
        if abs(skewness) > 1.5 and min_val > 0:
            return 'log'

        # Check for bounded data (likely percentages or ratios)
        if min_val >= 0 and max_val <= 1:
            return 'minmax'

        # Check for many outliers (use robust scaling)
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_count = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
        if outlier_count / len(data) > 0.1:
            return 'robust'

        # Default to standardization
        return 'standard'

    def _standardize(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Standardize column (z-score normalization)."""
        mean = df[column].mean()
        std = df[column].std()

        new_col = f"{column}_scaled"
        df[new_col] = (df[column] - mean) / (std + 1e-8)

        params = {"mean": mean, "std": std}
        self.scaling_params[column] = {"method": "standard", "params": params}
        return df, params

    def _minmax_scale(
        self,
        df: pd.DataFrame,
        column: str,
        feature_range: Tuple[float, float] = (0, 1)
    ) -> Tuple[pd.DataFrame, Dict]:
        """Min-Max scale column to specified range."""
        min_val = df[column].min()
        max_val = df[column].max()

        new_col = f"{column}_scaled"
        df[new_col] = (df[column] - min_val) / (max_val - min_val + 1e-8)

        # Scale to feature range
        range_min, range_max = feature_range
        df[new_col] = df[new_col] * (range_max - range_min) + range_min

        params = {"min": min_val, "max": max_val, "feature_range": feature_range}
        self.scaling_params[column] = {"method": "minmax", "params": params}
        return df, params

    def _robust_scale(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Robust scaling using median and IQR."""
        median = df[column].median()
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        new_col = f"{column}_scaled"
        df[new_col] = (df[column] - median) / (iqr + 1e-8)

        params = {"median": median, "q1": q1, "q3": q3, "iqr": iqr}
        self.scaling_params[column] = {"method": "robust", "params": params}
        return df, params

    def _log_transform(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Log transform column (handles zeros with log1p)."""
        min_val = df[column].min()

        new_col = f"{column}_log"

        # Handle negative values by shifting
        if min_val <= 0:
            shift = abs(min_val) + 1
            df[new_col] = np.log1p(df[column] + shift)
        else:
            shift = 0
            df[new_col] = np.log1p(df[column])

        # Also create standardized version of log-transformed data
        mean = df[new_col].mean()
        std = df[new_col].std()
        df[f"{column}_log_scaled"] = (df[new_col] - mean) / (std + 1e-8)

        params = {"shift": shift, "log_mean": mean, "log_std": std}
        self.scaling_params[column] = {"method": "log", "params": params}
        return df, params

    @classmethod
    def box_cox_transform(
        cls,
        df: pd.DataFrame,
        column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Box-Cox transformation for normalizing skewed data.
        Only works with positive data.
        """
        col_data = df[column].dropna()

        # Ensure positive values
        min_val = col_data.min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            col_data = col_data + shift
        else:
            shift = 0

        # Fit Box-Cox
        transformed_data, lambda_param = stats.boxcox(col_data)

        new_col = f"{column}_boxcox"
        df[new_col] = np.nan
        df.loc[df[column].notna(), new_col] = transformed_data

        params = {"lambda": lambda_param, "shift": shift}
        return df, params

    @classmethod
    def yeo_johnson_transform(
        cls,
        df: pd.DataFrame,
        column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Yeo-Johnson transformation for normalizing skewed data.
        Works with positive and negative data.
        """
        col_data = df[column].dropna()

        # Fit Yeo-Johnson
        transformed_data, lambda_param = stats.yeojohnson(col_data)

        new_col = f"{column}_yeojohnson"
        df[new_col] = np.nan
        df.loc[df[column].notna(), new_col] = transformed_data

        params = {"lambda": lambda_param}
        return df, params

    def transform_new_data(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply saved scaling parameters to new data.

        Args:
            df: New DataFrame to transform
            columns: Columns to transform (default: all with saved params)

        Returns:
            Transformed DataFrame
        """
        df_result = df.copy()

        if columns is None:
            columns = list(self.scaling_params.keys())

        for col in columns:
            if col not in self.scaling_params:
                continue

            params = self.scaling_params[col]
            method = params["method"]
            p = params["params"]

            if method == "standard":
                new_col = f"{col}_scaled"
                df_result[new_col] = (df_result[col] - p["mean"]) / (p["std"] + 1e-8)

            elif method == "minmax":
                new_col = f"{col}_scaled"
                df_result[new_col] = (df_result[col] - p["min"]) / (p["max"] - p["min"] + 1e-8)

            elif method == "robust":
                new_col = f"{col}_scaled"
                df_result[new_col] = (df_result[col] - p["median"]) / (p["iqr"] + 1e-8)

            elif method == "log":
                new_col = f"{col}_log"
                if p["shift"] > 0:
                    df_result[new_col] = np.log1p(df_result[col] + p["shift"])
                else:
                    df_result[new_col] = np.log1p(df_result[col])

                df_result[f"{col}_log_scaled"] = (
                    (df_result[new_col] - p["log_mean"]) / (p["log_std"] + 1e-8)
                )

        return df_result

    def get_scaling_summary(self) -> str:
        """Get human-readable scaling summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("FEATURE SCALING SUMMARY")
        lines.append("=" * 60)

        report = self.scaling_report

        lines.append(f"\nStandardized (z-score): {len(report.get('standardized', []))}")
        for item in report.get('standardized', []):
            lines.append(f"  - {item['column']}: mean={item['mean']:.3f}, std={item['std']:.3f}")

        lines.append(f"\nMin-Max Scaled: {len(report.get('minmax_scaled', []))}")
        for item in report.get('minmax_scaled', []):
            lines.append(f"  - {item['column']}: range [{item['min']:.3f}, {item['max']:.3f}]")

        lines.append(f"\nRobust Scaled: {len(report.get('robust_scaled', []))}")
        for item in report.get('robust_scaled', []):
            lines.append(f"  - {item['column']}: median={item['median']:.3f}, IQR={item['iqr']:.3f}")

        lines.append(f"\nLog Transformed: {len(report.get('log_transformed', []))}")
        for item in report.get('log_transformed', []):
            lines.append(f"  - {item['column']}: original_skewness={item['original_skewness']:.3f}")

        lines.append(f"\nSkipped: {len(report.get('skipped', []))}")
        for item in report.get('skipped', []):
            lines.append(f"  - {item['column']}: {item['reason']}")

        return "\n".join(lines)

    def save_params(self, filepath: str) -> None:
        """Save scaling parameters to JSON file."""
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_params = convert_to_serializable(self.scaling_params)

        with open(filepath, 'w') as f:
            json.dump(serializable_params, f, indent=2)

    def load_params(self, filepath: str) -> None:
        """Load scaling parameters from JSON file."""
        with open(filepath, 'r') as f:
            self.scaling_params = json.load(f)
