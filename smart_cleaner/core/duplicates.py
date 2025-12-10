"""
Duplicate detection and removal.
"""

from typing import Dict, List, Any, Optional
import pandas as pd


class DuplicateHandler:
    """Handle duplicate rows in DataFrames."""

    @staticmethod
    def detect_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> Dict[str, Any]:
        """
        Detect duplicate rows.

        Args:
            df: DataFrame
            subset: Columns to consider for duplicates (None = all columns)
            keep: Which duplicates to mark ('first', 'last', False)

        Returns:
            Dictionary with duplicate information
        """
        duplicate_mask = df.duplicated(subset=subset, keep=keep)
        duplicate_count = duplicate_mask.sum()

        return {
            "total_rows": len(df),
            "duplicate_count": int(duplicate_count),
            "duplicate_percentage": float(duplicate_count / len(df) * 100),
            "unique_rows": len(df) - duplicate_count,
            "subset_columns": subset or "all",
        }

    @staticmethod
    def remove_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            df: DataFrame
            subset: Columns to consider
            keep: Which duplicates to keep

        Returns:
            DataFrame with duplicates removed
        """
        return df.drop_duplicates(subset=subset, keep=keep)

    @staticmethod
    def auto_remove_duplicates(
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Automatically detect and remove duplicates.

        Args:
            df: DataFrame
            exclude_columns: Columns to exclude from duplicate detection
                           (e.g., IDs, timestamps)

        Returns:
            Tuple of (cleaned_df, report)
        """
        # Detect duplicates first
        detection = DuplicateHandler.detect_duplicates(df)

        if detection["duplicate_count"] == 0:
            # Add missing keys for consistency
            detection["rows_removed"] = 0
            detection["final_row_count"] = len(df)
            detection["excluded_columns"] = exclude_columns or []
            return df, detection

        # Determine columns for duplicate check
        if exclude_columns:
            check_columns = [col for col in df.columns if col not in exclude_columns]
        else:
            check_columns = None

        # Remove duplicates
        df_cleaned = DuplicateHandler.remove_duplicates(df, subset=check_columns)

        report = {
            **detection,
            "rows_removed": len(df) - len(df_cleaned),
            "final_row_count": len(df_cleaned),
            "excluded_columns": exclude_columns or [],
        }

        return df_cleaned, report
