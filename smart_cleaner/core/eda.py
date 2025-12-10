"""
Exploratory Data Analysis (EDA) utilities.
Provides tools for understanding data distributions, outliers, and patterns.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


class EDAAnalyzer:
    """
    Exploratory Data Analysis toolkit for understanding datasets.
    """

    @staticmethod
    def summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "shape": df.shape,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing_values": {
                col: int(df[col].isna().sum())
                for col in df.columns
                if df[col].isna().sum() > 0
            },
            "duplicate_rows": int(df.duplicated().sum()),
        }

        return summary

    @staticmethod
    def column_profile(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Generate detailed profile for a single column.

        Args:
            df: DataFrame
            column: Column name to profile

        Returns:
            Dictionary with column profile
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        profile = {
            "name": column,
            "dtype": str(df[column].dtype),
            "total_count": len(df),
            "missing_count": int(df[column].isna().sum()),
            "missing_percentage": float(df[column].isna().sum() / len(df) * 100),
            "unique_count": int(df[column].nunique()),
            "unique_percentage": float(df[column].nunique() / len(df) * 100),
        }

        # Numeric column statistics
        if pd.api.types.is_numeric_dtype(df[column]):
            non_null = df[column].dropna()
            if len(non_null) > 0:
                profile.update({
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "std": float(non_null.std()),
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "q25": float(non_null.quantile(0.25)),
                    "q75": float(non_null.quantile(0.75)),
                    "skewness": float(non_null.skew()),
                    "kurtosis": float(non_null.kurtosis()),
                })

                # Detect outliers using IQR method
                q1, q3 = non_null.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]

                profile.update({
                    "outlier_count": len(outliers),
                    "outlier_percentage": float(len(outliers) / len(non_null) * 100),
                })

        # Categorical column statistics
        else:
            value_counts = df[column].value_counts()
            profile.update({
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "top_5_values": value_counts.head(5).to_dict(),
            })

        return profile

    @staticmethod
    def detect_outliers(
        df: pd.DataFrame, column: str, method: str = "iqr"
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Detect outliers in a numeric column.

        Args:
            df: DataFrame
            column: Column name
            method: Detection method ('iqr' or 'zscore')

        Returns:
            Tuple of (outlier_mask, bounds_dict)
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' is not numeric")

        series = df[column].dropna()

        if method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            bounds = {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }

        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            z_scores = np.abs((df[column] - mean) / std)
            outlier_mask = z_scores > 3

            bounds = {
                "mean": float(mean),
                "std": float(std),
                "z_threshold": 3.0,
            }

        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")

        return outlier_mask, bounds

    @staticmethod
    def correlation_analysis(
        df: pd.DataFrame, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Analyze correlations between numeric columns.

        Args:
            df: DataFrame
            threshold: Minimum correlation to report

        Returns:
            Dictionary with correlation analysis
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {
                "message": "Not enough numeric columns for correlation analysis",
                "correlations": [],
            }

        corr_matrix = df[numeric_cols].corr()

        # Extract high correlations
        high_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) >= threshold:
                    high_correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": float(corr_value),
                        "strength": EDAAnalyzer._correlation_strength(abs(corr_value)),
                    })

        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "total_numeric_columns": len(numeric_cols),
            "high_correlations_count": len(high_correlations),
            "correlations": high_correlations,
            "correlation_matrix": corr_matrix.to_dict(),
        }

    @staticmethod
    def _correlation_strength(corr: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"

    @staticmethod
    def data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with quality metrics
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()

        # Calculate completeness score
        completeness = ((total_cells - missing_cells) / total_cells) * 100

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        uniqueness = ((len(df) - duplicates) / len(df)) * 100

        # Calculate overall quality score (weighted average)
        quality_score = (completeness * 0.6 + uniqueness * 0.4)

        report = {
            "overall_quality_score": round(quality_score, 2),
            "completeness_score": round(completeness, 2),
            "uniqueness_score": round(uniqueness, 2),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_cells": total_cells,
            "missing_cells": int(missing_cells),
            "duplicate_rows": int(duplicates),
            "columns_with_missing": len([col for col in df.columns if df[col].isna().any()]),
            "quality_level": EDAAnalyzer._quality_level(quality_score),
        }

        return report

    @staticmethod
    def _quality_level(score: float) -> str:
        """Classify data quality level."""
        if score >= 95:
            return "excellent"
        elif score >= 85:
            return "good"
        elif score >= 70:
            return "fair"
        elif score >= 50:
            return "poor"
        else:
            return "critical"

    @staticmethod
    def distribution_analysis(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Analyze distribution of a numeric column.

        Args:
            df: DataFrame
            column: Column name

        Returns:
            Dictionary with distribution characteristics
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' is not numeric")

        series = df[column].dropna()

        if len(series) == 0:
            return {"error": "No non-null values in column"}

        # Calculate distribution statistics
        skewness = series.skew()
        kurtosis = series.kurtosis()

        # Classify distribution shape
        if abs(skewness) < 0.5:
            skew_type = "symmetric"
        elif skewness > 0:
            skew_type = "right_skewed"
        else:
            skew_type = "left_skewed"

        if abs(kurtosis) < 3:
            kurt_type = "mesokurtic"
        elif kurtosis > 3:
            kurt_type = "leptokurtic"
        else:
            kurt_type = "platykurtic"

        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "mode": float(series.mode()[0]) if len(series.mode()) > 0 else None,
            "std": float(series.std()),
            "variance": float(series.var()),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "skew_type": skew_type,
            "kurtosis_type": kurt_type,
            "percentiles": {
                "p10": float(series.quantile(0.1)),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.5)),
                "p75": float(series.quantile(0.75)),
                "p90": float(series.quantile(0.9)),
            },
        }
