"""
Healthcare-specific data validation and cleaning.
Validates common health metrics and fixes invalid values.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


class HealthValidator:
    """
    Healthcare-specific data validation.
    Defines valid ranges for common health metrics.
    """

    # Standard health metric ranges
    HEALTH_RANGES = {
        "age": (0, 120),
        "bmi": (10, 60),
        "weight": (2, 300),  # kg
        "height": (40, 250),  # cm
        "blood_pressure_systolic": (70, 200),  # mmHg
        "blood_pressure_diastolic": (40, 130),  # mmHg
        "blood_pressure": (70, 200),  # Assuming systolic if single value
        "heart_rate": (30, 220),  # bpm
        "temperature": (35, 42),  # Celsius
        "glucose": (40, 600),  # mg/dL
        "cholesterol": (100, 400),  # mg/dL
        "oxygen_saturation": (70, 100),  # percentage
        "respiratory_rate": (8, 60),  # breaths per minute
    }

    @staticmethod
    def detect_column_type(column_name: str) -> Optional[str]:
        """
        Auto-detect health metric type from column name.

        Args:
            column_name: Name of the column

        Returns:
            Detected metric type or None
        """
        column_lower = column_name.lower().replace("_", "").replace(" ", "")

        # Check for exact matches or contains
        for metric_type in HealthValidator.HEALTH_RANGES.keys():
            metric_normalized = metric_type.replace("_", "")
            if metric_normalized in column_lower or column_lower in metric_normalized:
                return metric_type

        # Special cases
        if "bp" in column_lower and "sys" in column_lower:
            return "blood_pressure_systolic"
        if "bp" in column_lower and ("dia" in column_lower or "dys" in column_lower):
            return "blood_pressure_diastolic"
        if "hr" in column_lower or "pulse" in column_lower:
            return "heart_rate"
        if "temp" in column_lower:
            return "temperature"
        if "spo2" in column_lower or "o2" in column_lower:
            return "oxygen_saturation"
        if "rr" in column_lower or "respiration" in column_lower:
            return "respiratory_rate"

        return None

    @staticmethod
    def validate_column(
        df: pd.DataFrame,
        column: str,
        metric_type: Optional[str] = None,
        custom_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a health metric column.

        Args:
            df: DataFrame
            column: Column name
            metric_type: Type of health metric (auto-detected if None)
            custom_range: Custom valid range (min, max)

        Returns:
            Validation report
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                "column": column,
                "valid": False,
                "reason": "Not a numeric column",
            }

        # Auto-detect metric type
        if metric_type is None:
            metric_type = HealthValidator.detect_column_type(column)

        # Get valid range
        if custom_range:
            min_val, max_val = custom_range
        elif metric_type and metric_type in HealthValidator.HEALTH_RANGES:
            min_val, max_val = HealthValidator.HEALTH_RANGES[metric_type]
        else:
            # No validation range available
            return {
                "column": column,
                "metric_type": "unknown",
                "validation_applied": False,
                "reason": "No validation range available",
            }

        # Check for invalid values
        series = df[column].dropna()
        invalid_mask = (series < min_val) | (series > max_val)
        invalid_count = invalid_mask.sum()

        return {
            "column": column,
            "metric_type": metric_type or "custom",
            "valid_range": (float(min_val), float(max_val)),
            "invalid_count": int(invalid_count),
            "invalid_percentage": float(invalid_count / len(series) * 100) if len(series) > 0 else 0,
            "validation_applied": True,
            "invalid_values": series[invalid_mask].tolist()[:10],  # First 10 examples
        }

    @staticmethod
    def fix_invalid_values(
        df: pd.DataFrame,
        column: str,
        metric_type: Optional[str] = None,
        custom_range: Optional[Tuple[float, float]] = None,
        strategy: str = "cap",
    ) -> pd.DataFrame:
        """
        Fix invalid values in a health metric column.

        Args:
            df: DataFrame
            column: Column name
            metric_type: Type of health metric
            custom_range: Custom valid range
            strategy: How to fix ('cap', 'remove', 'set_nan')

        Returns:
            DataFrame with fixed values
        """
        df_result = df.copy()

        # Get validation info
        validation = HealthValidator.validate_column(
            df_result, column, metric_type, custom_range
        )

        if not validation.get("validation_applied") or validation["invalid_count"] == 0:
            return df_result

        min_val, max_val = validation["valid_range"]
        series = df_result[column]

        if strategy == "cap":
            # Cap values at bounds
            df_result.loc[series < min_val, column] = min_val
            df_result.loc[series > max_val, column] = max_val

        elif strategy == "remove":
            # Remove rows with invalid values
            valid_mask = (series >= min_val) & (series <= max_val) | series.isna()
            df_result = df_result[valid_mask]

        elif strategy == "set_nan":
            # Set invalid values to NaN
            invalid_mask = (series < min_val) | (series > max_val)
            df_result.loc[invalid_mask, column] = np.nan

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return df_result

    @staticmethod
    def auto_validate_all(
        df: pd.DataFrame,
        fix_strategy: str = "cap",
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Automatically validate and fix all health metrics in DataFrame.

        Args:
            df: DataFrame
            fix_strategy: How to fix invalid values

        Returns:
            Tuple of (cleaned_df, validation_report)
        """
        df_result = df.copy()
        validation_report = []

        for column in df_result.columns:
            # Try to detect health metric
            metric_type = HealthValidator.detect_column_type(column)

            if metric_type:
                try:
                    # Validate
                    validation = HealthValidator.validate_column(
                        df_result, column, metric_type
                    )

                    if validation.get("invalid_count", 0) > 0:
                        # Fix invalid values
                        df_result = HealthValidator.fix_invalid_values(
                            df_result, column, metric_type, strategy=fix_strategy
                        )

                        validation["strategy_applied"] = fix_strategy
                        validation_report.append(validation)

                except Exception as e:
                    validation_report.append({
                        "column": column,
                        "error": str(e),
                    })

        return df_result, validation_report

    @staticmethod
    def get_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get recommendations for health data validation.

        Args:
            df: DataFrame

        Returns:
            List of recommendations
        """
        recommendations = []

        for column in df.columns:
            metric_type = HealthValidator.detect_column_type(column)

            if metric_type:
                validation = HealthValidator.validate_column(df, column, metric_type)

                if validation.get("invalid_count", 0) > 0:
                    recommendations.append({
                        "column": column,
                        "metric_type": metric_type,
                        "issue": f"Found {validation['invalid_count']} invalid values",
                        "valid_range": validation["valid_range"],
                        "recommendation": f"Cap values to range {validation['valid_range']} or set to NaN",
                    })

        return recommendations
