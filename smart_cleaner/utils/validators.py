"""
Data validation utilities for Smart Cleaner.
"""

import pandas as pd
from typing import List, Dict, Any


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate that input is a proper DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")
    return True


def get_missing_value_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive summary of missing values in DataFrame.

    Returns:
        Dictionary with missing value statistics per column.
    """
    missing_info = {}

    for column in df.columns:
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            missing_info[column] = {
                "count": int(missing_count),
                "percentage": float(missing_count / len(df) * 100),
                "dtype": str(df[column].dtype),
            }

    return missing_info


def get_column_statistics(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Get statistical information about a specific column.

    Returns:
        Dictionary with column statistics.
    """
    stats = {
        "name": column,
        "dtype": str(df[column].dtype),
        "total_count": len(df),
        "missing_count": int(df[column].isna().sum()),
        "unique_count": int(df[column].nunique()),
    }

    # Add numeric statistics if applicable
    if pd.api.types.is_numeric_dtype(df[column]):
        non_null_values = df[column].dropna()
        if len(non_null_values) > 0:
            stats.update({
                "mean": float(non_null_values.mean()),
                "median": float(non_null_values.median()),
                "std": float(non_null_values.std()),
                "min": float(non_null_values.min()),
                "max": float(non_null_values.max()),
            })

    return stats


def detect_correlations(df: pd.DataFrame, threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Detect correlations between numeric columns.

    Args:
        df: Input DataFrame
        threshold: Minimum correlation coefficient to report

    Returns:
        List of correlation pairs above threshold.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) < 2:
        return []

    correlations = []
    corr_matrix = df[numeric_cols].corr()

    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1 :]:
            corr_value = corr_matrix.loc[col1, col2]
            if abs(corr_value) >= threshold:
                correlations.append({
                    "column1": col1,
                    "column2": col2,
                    "correlation": float(corr_value),
                })

    return sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)
