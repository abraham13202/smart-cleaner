"""
Feature Encoding module for categorical variables.
Handles one-hot encoding, label encoding, ordinal encoding, and more.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json


class FeatureEncoder:
    """
    Automated feature encoding for categorical variables.
    Chooses appropriate encoding based on cardinality and data type.
    """

    # Ordinal mappings for common health-related categories
    ORDINAL_MAPPINGS = {
        'education': {
            'no_education': 0, 'none': 0,
            'elementary': 1, 'primary': 1,
            'some_high_school': 2, 'some high school': 2,
            'high_school': 3, 'high school': 3, 'high school graduate': 3,
            'some_college': 4, 'some college': 4,
            'college': 5, 'college graduate': 5, 'bachelors': 5,
            'graduate': 6, 'masters': 6, 'phd': 7, 'doctorate': 7,
        },
        'income': {
            'low': 0, 'lower': 0,
            'lower_middle': 1, 'lower middle': 1,
            'middle': 2,
            'upper_middle': 3, 'upper middle': 3,
            'high': 4, 'upper': 4,
        },
        'general_health': {
            'poor': 0,
            'fair': 1,
            'good': 2,
            'very_good': 3, 'very good': 3,
            'excellent': 4,
        },
        'frequency': {
            'never': 0,
            'rarely': 1,
            'sometimes': 2,
            'often': 3,
            'always': 4,
        },
        'agreement': {
            'strongly_disagree': 0, 'strongly disagree': 0,
            'disagree': 1,
            'neutral': 2,
            'agree': 3,
            'strongly_agree': 4, 'strongly agree': 4,
        },
    }

    # Binary mappings
    BINARY_MAPPINGS = {
        'yes': 1, 'no': 0,
        'true': 1, 'false': 0,
        'y': 1, 'n': 0,
        't': 1, 'f': 0,
        '1': 1, '0': 0,
        'male': 1, 'female': 0,
        'm': 1, 'f': 0,
        'positive': 1, 'negative': 0,
        'present': 1, 'absent': 0,
    }

    def __init__(self):
        """Initialize encoder with storage for fitted encodings."""
        self.encodings = {}
        self.encoding_report = {}

    def auto_encode(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        max_onehot_cardinality: int = 10,
        drop_original: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Automatically encode all categorical columns.

        Strategy:
        - Binary columns (2 unique values): Binary encoding (0/1)
        - Low cardinality (≤ max_onehot_cardinality): One-hot encoding
        - High cardinality: Label encoding with frequency mapping
        - Ordinal columns (detected by name): Ordinal encoding

        Args:
            df: Input DataFrame
            target_column: Target variable (excluded from encoding)
            max_onehot_cardinality: Max unique values for one-hot encoding
            drop_original: Whether to drop original categorical columns

        Returns:
            Tuple of (encoded DataFrame, encoding report)
        """
        df_result = df.copy()
        report = {
            "binary_encoded": [],
            "onehot_encoded": [],
            "label_encoded": [],
            "ordinal_encoded": [],
            "skipped": [],
            "encoding_mappings": {},
        }

        # Get categorical columns
        categorical_cols = df_result.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        # Exclude target column
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)

        for col in categorical_cols:
            unique_values = df_result[col].dropna().unique()
            n_unique = len(unique_values)

            # Skip if all null
            if n_unique == 0:
                report["skipped"].append({"column": col, "reason": "all null values"})
                continue

            # Check if it's an ordinal column
            ordinal_type = self._detect_ordinal_type(col, unique_values)

            if ordinal_type:
                # Ordinal encoding
                df_result, mapping = self._ordinal_encode(df_result, col, ordinal_type)
                report["ordinal_encoded"].append({
                    "column": col,
                    "ordinal_type": ordinal_type,
                    "mapping": mapping,
                })
                report["encoding_mappings"][col] = {
                    "type": "ordinal",
                    "mapping": mapping,
                }

            elif n_unique == 2:
                # Binary encoding
                df_result, mapping = self._binary_encode(df_result, col)
                report["binary_encoded"].append({
                    "column": col,
                    "mapping": mapping,
                })
                report["encoding_mappings"][col] = {
                    "type": "binary",
                    "mapping": mapping,
                }

            elif n_unique <= max_onehot_cardinality:
                # One-hot encoding
                df_result, new_cols = self._onehot_encode(df_result, col, drop_original)
                report["onehot_encoded"].append({
                    "column": col,
                    "new_columns": new_cols,
                    "cardinality": n_unique,
                })
                report["encoding_mappings"][col] = {
                    "type": "onehot",
                    "new_columns": new_cols,
                }

            else:
                # Label encoding with frequency for high cardinality
                df_result, mapping = self._label_encode(df_result, col)
                report["label_encoded"].append({
                    "column": col,
                    "cardinality": n_unique,
                    "mapping_sample": dict(list(mapping.items())[:10]),
                })
                report["encoding_mappings"][col] = {
                    "type": "label",
                    "mapping": mapping,
                }

            # Drop original if requested and one-hot was used
            if drop_original and col in df_result.columns:
                if any(col == item["column"] for item in report["onehot_encoded"]):
                    df_result = df_result.drop(columns=[col])

        self.encoding_report = report
        return df_result, report

    def _detect_ordinal_type(
        self,
        column_name: str,
        unique_values: np.ndarray
    ) -> Optional[str]:
        """Detect if a column should use ordinal encoding."""
        col_lower = column_name.lower().replace('_', ' ').replace('-', ' ')

        # Check column name patterns
        for ordinal_type, mapping in self.ORDINAL_MAPPINGS.items():
            if ordinal_type in col_lower:
                return ordinal_type

        # Check if values match known ordinal patterns
        values_lower = set(str(v).lower().strip() for v in unique_values if pd.notna(v))

        for ordinal_type, mapping in self.ORDINAL_MAPPINGS.items():
            mapping_keys = set(mapping.keys())
            # If significant overlap, it's likely ordinal
            overlap = len(values_lower & mapping_keys)
            if overlap >= len(values_lower) * 0.5 and overlap >= 2:
                return ordinal_type

        return None

    def _binary_encode(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Encode binary categorical column to 0/1."""
        unique_values = df[column].dropna().unique()
        mapping = {}

        for val in unique_values:
            val_lower = str(val).lower().strip()
            if val_lower in self.BINARY_MAPPINGS:
                mapping[val] = self.BINARY_MAPPINGS[val_lower]

        # If no match found, assign 0 and 1 arbitrarily
        if len(mapping) < 2:
            for i, val in enumerate(unique_values):
                if val not in mapping:
                    mapping[val] = i

        new_col = f"{column}_encoded"
        df[new_col] = df[column].map(mapping)

        self.encodings[column] = {"type": "binary", "mapping": mapping}
        return df, mapping

    def _onehot_encode(
        self,
        df: pd.DataFrame,
        column: str,
        drop_original: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        """One-hot encode a categorical column."""
        # Create dummies
        dummies = pd.get_dummies(
            df[column],
            prefix=column,
            prefix_sep='_',
            dummy_na=False,
            dtype=int
        )

        # Clean column names
        new_cols = []
        for col in dummies.columns:
            clean_col = col.replace(' ', '_').replace('-', '_')
            new_cols.append(clean_col)

        dummies.columns = new_cols

        # Add to dataframe
        df = pd.concat([df, dummies], axis=1)

        self.encodings[column] = {"type": "onehot", "columns": new_cols}
        return df, new_cols

    def _ordinal_encode(
        self,
        df: pd.DataFrame,
        column: str,
        ordinal_type: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Encode ordinal categorical column."""
        mapping = self.ORDINAL_MAPPINGS.get(ordinal_type, {})

        def map_value(val):
            if pd.isna(val):
                return np.nan
            val_lower = str(val).lower().strip()
            return mapping.get(val_lower, np.nan)

        new_col = f"{column}_encoded"
        df[new_col] = df[column].apply(map_value)

        # For unmapped values, use median of mapped values
        median_val = df[new_col].median()
        df[new_col] = df[new_col].fillna(median_val)

        self.encodings[column] = {"type": "ordinal", "mapping": mapping}
        return df, mapping

    def _label_encode(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """Label encode with frequency-based ordering (most frequent = 0)."""
        # Get value counts
        value_counts = df[column].value_counts()

        # Create mapping (most frequent gets lowest number)
        mapping = {val: idx for idx, val in enumerate(value_counts.index)}

        new_col = f"{column}_encoded"
        df[new_col] = df[column].map(mapping)

        self.encodings[column] = {"type": "label", "mapping": mapping}
        return df, mapping

    @classmethod
    def target_encode(
        cls,
        df: pd.DataFrame,
        column: str,
        target_column: str,
        smoothing: float = 1.0,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Target encoding (mean encoding) for categorical column.

        Args:
            df: Input DataFrame
            column: Column to encode
            target_column: Target variable
            smoothing: Smoothing parameter (higher = more regularization)

        Returns:
            Tuple of (encoded DataFrame, mapping dictionary)
        """
        # Calculate global mean
        global_mean = df[target_column].mean()

        # Calculate category statistics
        stats = df.groupby(column)[target_column].agg(['mean', 'count'])

        # Apply smoothing
        smoothed_mean = (
            (stats['count'] * stats['mean'] + smoothing * global_mean) /
            (stats['count'] + smoothing)
        )

        mapping = smoothed_mean.to_dict()

        new_col = f"{column}_target_encoded"
        df[new_col] = df[column].map(mapping).fillna(global_mean)

        return df, mapping

    @classmethod
    def frequency_encode(
        cls,
        df: pd.DataFrame,
        column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Frequency encoding - replace categories with their frequency.

        Args:
            df: Input DataFrame
            column: Column to encode

        Returns:
            Tuple of (encoded DataFrame, mapping dictionary)
        """
        freq = df[column].value_counts(normalize=True)
        mapping = freq.to_dict()

        new_col = f"{column}_freq_encoded"
        df[new_col] = df[column].map(mapping)

        return df, mapping

    def get_encoding_summary(self) -> str:
        """Get human-readable encoding summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("FEATURE ENCODING SUMMARY")
        lines.append("=" * 60)

        report = self.encoding_report

        lines.append(f"\nBinary Encoded: {len(report.get('binary_encoded', []))}")
        for item in report.get('binary_encoded', []):
            lines.append(f"  - {item['column']}: {item['mapping']}")

        lines.append(f"\nOne-Hot Encoded: {len(report.get('onehot_encoded', []))}")
        for item in report.get('onehot_encoded', []):
            lines.append(f"  - {item['column']} ({item['cardinality']} categories)")
            lines.append(f"    New columns: {', '.join(item['new_columns'][:5])}...")

        lines.append(f"\nOrdinal Encoded: {len(report.get('ordinal_encoded', []))}")
        for item in report.get('ordinal_encoded', []):
            lines.append(f"  - {item['column']} (type: {item['ordinal_type']})")

        lines.append(f"\nLabel Encoded: {len(report.get('label_encoded', []))}")
        for item in report.get('label_encoded', []):
            lines.append(f"  - {item['column']} ({item['cardinality']} categories)")

        lines.append(f"\nSkipped: {len(report.get('skipped', []))}")
        for item in report.get('skipped', []):
            lines.append(f"  - {item['column']}: {item['reason']}")

        return "\n".join(lines)

    def save_encodings(self, filepath: str) -> None:
        """Save encoding mappings to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_encodings = convert_to_serializable(self.encodings)

        with open(filepath, 'w') as f:
            json.dump(serializable_encodings, f, indent=2)

    def load_encodings(self, filepath: str) -> None:
        """Load encoding mappings from JSON file."""
        with open(filepath, 'r') as f:
            self.encodings = json.load(f)
