"""
Data Profiling Module for Smart Cleaner.
Generates comprehensive data quality reports and statistics.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ColumnProfile:
    """Profile for a single column."""
    name: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float

    # Numeric stats (if applicable)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    zeros: Optional[int] = None
    negatives: Optional[int] = None

    # Categorical stats (if applicable)
    top_values: Optional[Dict[str, int]] = None
    mode: Optional[Any] = None

    # Quality indicators
    outliers: Optional[int] = None
    duplicates: Optional[int] = None

    # Inferred info
    inferred_type: Optional[str] = None  # numeric, categorical, datetime, text, id
    is_potential_id: bool = False
    is_constant: bool = False


@dataclass
class DataProfile:
    """Complete data profile."""
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_name: Optional[str] = None

    # Overview
    n_rows: int = 0
    n_columns: int = 0
    memory_usage_mb: float = 0.0

    # Quality metrics
    total_missing: int = 0
    total_missing_pct: float = 0.0
    duplicate_rows: int = 0
    duplicate_rows_pct: float = 0.0
    completeness_score: float = 0.0

    # Column profiles
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)

    # Correlations
    correlations: Optional[Dict[str, Dict[str, float]]] = None

    # Type distribution
    type_distribution: Dict[str, int] = field(default_factory=dict)

    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class DataProfiler:
    """
    Generate comprehensive data profiles.

    Usage:
        profiler = DataProfiler()
        profile = profiler.profile(df)
        print(profile.completeness_score)
    """

    def __init__(
        self,
        sample_size: Optional[int] = None,
        correlation_threshold: float = 0.9,
        cardinality_threshold: float = 0.95,
    ):
        """
        Initialize profiler.

        Args:
            sample_size: If set, profile a sample instead of full data
            correlation_threshold: Threshold for high correlation warnings
            cardinality_threshold: Threshold for potential ID column detection
        """
        self.sample_size = sample_size
        self.correlation_threshold = correlation_threshold
        self.cardinality_threshold = cardinality_threshold

    def profile(self, df: pd.DataFrame, file_name: str = None) -> DataProfile:
        """
        Generate complete profile for a DataFrame.

        Args:
            df: DataFrame to profile
            file_name: Optional file name for metadata

        Returns:
            DataProfile object with all statistics
        """
        # Sample if needed
        if self.sample_size and len(df) > self.sample_size:
            df_work = df.sample(self.sample_size, random_state=42)
        else:
            df_work = df

        profile = DataProfile(
            file_name=file_name,
            n_rows=len(df),
            n_columns=len(df.columns),
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
        )

        # Calculate overview stats
        profile.total_missing = df.isnull().sum().sum()
        profile.total_missing_pct = (profile.total_missing / df.size) * 100
        profile.duplicate_rows = df.duplicated().sum()
        profile.duplicate_rows_pct = (profile.duplicate_rows / len(df)) * 100 if len(df) > 0 else 0
        profile.completeness_score = 100 - profile.total_missing_pct

        # Profile each column
        for col in df.columns:
            profile.columns[col] = self._profile_column(df, col)

        # Type distribution
        for col_profile in profile.columns.values():
            dtype = col_profile.inferred_type or col_profile.dtype
            profile.type_distribution[dtype] = profile.type_distribution.get(dtype, 0) + 1

        # Calculate correlations for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            profile.correlations = corr_matrix.to_dict()

            # Check for high correlations
            self._check_correlations(profile, corr_matrix)

        # Generate warnings
        self._generate_warnings(profile, df)

        # Generate recommendations
        self._generate_recommendations(profile, df)

        return profile

    def _profile_column(self, df: pd.DataFrame, column: str) -> ColumnProfile:
        """Profile a single column."""
        col = df[column]

        # Basic stats
        profile = ColumnProfile(
            name=column,
            dtype=str(col.dtype),
            count=len(col),
            missing=col.isnull().sum(),
            missing_pct=(col.isnull().sum() / len(col)) * 100,
            unique=col.nunique(),
            unique_pct=(col.nunique() / len(col)) * 100 if len(col) > 0 else 0,
        )

        # Check for constant column
        profile.is_constant = profile.unique <= 1

        # Check for potential ID column
        profile.is_potential_id = (
            profile.unique_pct >= self.cardinality_threshold * 100 and
            profile.missing_pct < 1
        )

        # Infer type
        profile.inferred_type = self._infer_column_type(col, profile)

        # Numeric statistics
        if pd.api.types.is_numeric_dtype(col):
            self._add_numeric_stats(col, profile)

        # Categorical statistics
        if pd.api.types.is_object_dtype(col) or pd.api.types.is_categorical_dtype(col):
            self._add_categorical_stats(col, profile)

        # Detect duplicates in column
        profile.duplicates = len(col) - col.nunique()

        return profile

    def _add_numeric_stats(self, col: pd.Series, profile: ColumnProfile):
        """Add numeric statistics to column profile."""
        non_null = col.dropna()

        if len(non_null) == 0:
            return

        profile.mean = float(non_null.mean())
        profile.std = float(non_null.std())
        profile.min_val = float(non_null.min())
        profile.max_val = float(non_null.max())
        profile.median = float(non_null.median())
        profile.q1 = float(non_null.quantile(0.25))
        profile.q3 = float(non_null.quantile(0.75))
        profile.iqr = profile.q3 - profile.q1

        # Skewness and kurtosis
        if len(non_null) > 2:
            profile.skewness = float(non_null.skew())
        if len(non_null) > 3:
            profile.kurtosis = float(non_null.kurtosis())

        # Count zeros and negatives
        profile.zeros = int((non_null == 0).sum())
        profile.negatives = int((non_null < 0).sum())

        # Detect outliers (IQR method)
        if profile.iqr > 0:
            lower = profile.q1 - 1.5 * profile.iqr
            upper = profile.q3 + 1.5 * profile.iqr
            profile.outliers = int(((non_null < lower) | (non_null > upper)).sum())

    def _add_categorical_stats(self, col: pd.Series, profile: ColumnProfile):
        """Add categorical statistics to column profile."""
        non_null = col.dropna()

        if len(non_null) == 0:
            return

        # Top values
        value_counts = non_null.value_counts().head(10)
        profile.top_values = value_counts.to_dict()

        # Mode
        mode = non_null.mode()
        profile.mode = mode.iloc[0] if len(mode) > 0 else None

    def _infer_column_type(self, col: pd.Series, profile: ColumnProfile) -> str:
        """Infer the semantic type of a column."""
        # Already detected as potential ID
        if profile.is_potential_id:
            return "id"

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(col):
            return "datetime"

        # Numeric
        if pd.api.types.is_numeric_dtype(col):
            if profile.unique <= 2:
                return "binary"
            elif profile.unique <= 10:
                return "ordinal"
            else:
                return "numeric"

        # Text/Categorical
        if pd.api.types.is_object_dtype(col):
            # Check for datetime strings
            if self._looks_like_datetime(col):
                return "datetime_string"

            # Check average string length
            non_null = col.dropna()
            if len(non_null) > 0:
                avg_len = non_null.astype(str).str.len().mean()
                if avg_len > 50:
                    return "text"

            if profile.unique_pct < 5:
                return "categorical"
            else:
                return "text"

        return "unknown"

    def _looks_like_datetime(self, col: pd.Series, sample_size: int = 100) -> bool:
        """Check if column looks like datetime strings."""
        sample = col.dropna().head(sample_size)
        if len(sample) == 0:
            return False

        try:
            pd.to_datetime(sample)
            return True
        except:
            return False

    def _check_correlations(self, profile: DataProfile, corr_matrix: pd.DataFrame):
        """Check for highly correlated features."""
        checked = set()
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2 and (col2, col1) not in checked:
                    corr = abs(corr_matrix.loc[col1, col2])
                    if corr >= self.correlation_threshold:
                        profile.warnings.append(
                            f"High correlation ({corr:.2f}) between '{col1}' and '{col2}'"
                        )
                    checked.add((col1, col2))

    def _generate_warnings(self, profile: DataProfile, df: pd.DataFrame):
        """Generate data quality warnings."""
        # High missing data
        if profile.total_missing_pct > 20:
            profile.warnings.append(
                f"High missing data: {profile.total_missing_pct:.1f}% of values are missing"
            )

        # Many duplicates
        if profile.duplicate_rows_pct > 10:
            profile.warnings.append(
                f"Many duplicate rows: {profile.duplicate_rows} ({profile.duplicate_rows_pct:.1f}%)"
            )

        # Column-specific warnings
        for col_name, col_profile in profile.columns.items():
            # High missing
            if col_profile.missing_pct > 50:
                profile.warnings.append(
                    f"Column '{col_name}' has {col_profile.missing_pct:.1f}% missing values"
                )

            # Constant column
            if col_profile.is_constant:
                profile.warnings.append(
                    f"Column '{col_name}' is constant (single value)"
                )

            # High cardinality
            if col_profile.unique_pct > 95 and col_profile.inferred_type != "id":
                profile.warnings.append(
                    f"Column '{col_name}' has very high cardinality ({col_profile.unique_pct:.1f}%)"
                )

            # Many outliers
            if col_profile.outliers and col_profile.outliers > len(df) * 0.1:
                profile.warnings.append(
                    f"Column '{col_name}' has {col_profile.outliers} outliers ({col_profile.outliers / len(df) * 100:.1f}%)"
                )

            # Highly skewed
            if col_profile.skewness is not None and abs(col_profile.skewness) > 2:
                direction = "right" if col_profile.skewness > 0 else "left"
                profile.warnings.append(
                    f"Column '{col_name}' is highly {direction}-skewed (skewness: {col_profile.skewness:.2f})"
                )

    def _generate_recommendations(self, profile: DataProfile, df: pd.DataFrame):
        """Generate actionable recommendations."""
        # Missing data recommendations
        columns_to_impute = []
        columns_to_drop = []

        for col_name, col_profile in profile.columns.items():
            if col_profile.missing_pct > 0:
                if col_profile.missing_pct > 70:
                    columns_to_drop.append(col_name)
                elif col_profile.missing_pct > 0:
                    columns_to_impute.append(col_name)

        if columns_to_impute:
            profile.recommendations.append(
                f"Consider imputing missing values in: {', '.join(columns_to_impute[:5])}"
                + (f" and {len(columns_to_impute) - 5} more" if len(columns_to_impute) > 5 else "")
            )

        if columns_to_drop:
            profile.recommendations.append(
                f"Consider dropping columns with >70% missing: {', '.join(columns_to_drop)}"
            )

        # Duplicate recommendations
        if profile.duplicate_rows > 0:
            profile.recommendations.append(
                f"Remove {profile.duplicate_rows} duplicate rows"
            )

        # Constant column recommendations
        constant_cols = [
            name for name, p in profile.columns.items() if p.is_constant
        ]
        if constant_cols:
            profile.recommendations.append(
                f"Consider removing constant columns: {', '.join(constant_cols)}"
            )

        # Outlier recommendations
        outlier_cols = [
            name for name, p in profile.columns.items()
            if p.outliers and p.outliers > 10
        ]
        if outlier_cols:
            profile.recommendations.append(
                f"Review outliers in: {', '.join(outlier_cols[:5])}"
            )

    def to_dict(self, profile: DataProfile) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "metadata": {
                "created_at": profile.created_at,
                "file_name": profile.file_name,
            },
            "overview": {
                "n_rows": profile.n_rows,
                "n_columns": profile.n_columns,
                "memory_usage_mb": round(profile.memory_usage_mb, 2),
                "total_missing": profile.total_missing,
                "total_missing_pct": round(profile.total_missing_pct, 2),
                "duplicate_rows": profile.duplicate_rows,
                "duplicate_rows_pct": round(profile.duplicate_rows_pct, 2),
                "completeness_score": round(profile.completeness_score, 2),
            },
            "type_distribution": profile.type_distribution,
            "columns": {
                name: {
                    "dtype": p.dtype,
                    "inferred_type": p.inferred_type,
                    "count": p.count,
                    "missing": p.missing,
                    "missing_pct": round(p.missing_pct, 2),
                    "unique": p.unique,
                    "unique_pct": round(p.unique_pct, 2),
                    "is_potential_id": p.is_potential_id,
                    "is_constant": p.is_constant,
                    "mean": round(p.mean, 2) if p.mean is not None else None,
                    "std": round(p.std, 2) if p.std is not None else None,
                    "min": p.min_val,
                    "max": p.max_val,
                    "median": round(p.median, 2) if p.median is not None else None,
                    "outliers": p.outliers,
                    "top_values": p.top_values,
                }
                for name, p in profile.columns.items()
            },
            "warnings": profile.warnings,
            "recommendations": profile.recommendations,
        }

    def to_json(self, profile: DataProfile, indent: int = 2) -> str:
        """Convert profile to JSON string."""
        return json.dumps(self.to_dict(profile), indent=indent, default=str)

    def print_summary(self, profile: DataProfile):
        """Print a human-readable summary."""
        print("\n" + "=" * 60)
        print("DATA PROFILE SUMMARY")
        print("=" * 60)

        print(f"\n📊 Overview:")
        print(f"   Rows: {profile.n_rows:,}")
        print(f"   Columns: {profile.n_columns}")
        print(f"   Memory: {profile.memory_usage_mb:.2f} MB")
        print(f"   Completeness: {profile.completeness_score:.1f}%")

        print(f"\n📈 Quality Metrics:")
        print(f"   Missing values: {profile.total_missing:,} ({profile.total_missing_pct:.1f}%)")
        print(f"   Duplicate rows: {profile.duplicate_rows:,} ({profile.duplicate_rows_pct:.1f}%)")

        print(f"\n📋 Type Distribution:")
        for dtype, count in profile.type_distribution.items():
            print(f"   {dtype}: {count}")

        if profile.warnings:
            print(f"\n⚠️  Warnings ({len(profile.warnings)}):")
            for warning in profile.warnings[:5]:
                print(f"   • {warning}")
            if len(profile.warnings) > 5:
                print(f"   ... and {len(profile.warnings) - 5} more")

        if profile.recommendations:
            print(f"\n💡 Recommendations ({len(profile.recommendations)}):")
            for rec in profile.recommendations[:5]:
                print(f"   • {rec}")

        print("\n" + "=" * 60)
