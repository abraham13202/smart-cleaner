"""
Advanced Exploratory Data Analysis module.
Provides comprehensive statistical analysis and data profiling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from collections import Counter


class AdvancedEDA:
    """
    Comprehensive exploratory data analysis for health datasets.
    Generates detailed statistics, distribution analysis, and insights.
    """

    def __init__(self):
        """Initialize EDA with storage for analysis results."""
        self.analysis_results = {}

    def full_analysis(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive EDA on the dataset.

        Args:
            df: Input DataFrame
            target_column: Target variable for relationship analysis

        Returns:
            Dictionary with all analysis results
        """
        results = {
            "dataset_overview": self._dataset_overview(df),
            "column_profiles": self._profile_all_columns(df),
            "missing_patterns": self._analyze_missing_patterns(df),
            "distribution_analysis": self._analyze_distributions(df),
            "correlation_analysis": self._analyze_correlations(df),
            "outlier_analysis": self._analyze_outliers(df),
            "data_quality_score": self._calculate_quality_score(df),
        }

        if target_column and target_column in df.columns:
            results["target_analysis"] = self._analyze_target(df, target_column)
            results["feature_target_relationships"] = self._analyze_feature_target(
                df, target_column
            )

        self.analysis_results = results
        return results

    def _dataset_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate high-level dataset overview."""
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024

        return {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "memory_mb": round(memory_usage, 2),
            "n_numeric": len(df.select_dtypes(include=[np.number]).columns),
            "n_categorical": len(df.select_dtypes(include=['object', 'category']).columns),
            "n_datetime": len(df.select_dtypes(include=['datetime64']).columns),
            "n_duplicates": df.duplicated().sum(),
            "duplicate_percentage": round(df.duplicated().sum() / len(df) * 100, 2),
            "total_missing": df.isnull().sum().sum(),
            "total_cells": df.shape[0] * df.shape[1],
            "missing_percentage": round(
                df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2
            ),
            "column_types": df.dtypes.astype(str).to_dict(),
        }

    def _profile_all_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Generate detailed profile for each column."""
        profiles = {}

        for col in df.columns:
            profiles[col] = self._profile_column(df, col)

        return profiles

    def _profile_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Generate detailed profile for a single column."""
        col_data = df[column]
        profile = {
            "dtype": str(col_data.dtype),
            "n_values": len(col_data),
            "n_missing": col_data.isnull().sum(),
            "missing_pct": round(col_data.isnull().sum() / len(col_data) * 100, 2),
            "n_unique": col_data.nunique(),
            "unique_pct": round(col_data.nunique() / len(col_data) * 100, 2),
        }

        # Numeric column statistics
        if pd.api.types.is_numeric_dtype(col_data):
            col_clean = col_data.dropna()
            if len(col_clean) > 0:
                profile.update({
                    "mean": round(col_clean.mean(), 4),
                    "std": round(col_clean.std(), 4),
                    "min": round(col_clean.min(), 4),
                    "max": round(col_clean.max(), 4),
                    "median": round(col_clean.median(), 4),
                    "q1": round(col_clean.quantile(0.25), 4),
                    "q3": round(col_clean.quantile(0.75), 4),
                    "iqr": round(col_clean.quantile(0.75) - col_clean.quantile(0.25), 4),
                    "skewness": round(col_clean.skew(), 4),
                    "kurtosis": round(col_clean.kurtosis(), 4),
                    "zeros": (col_clean == 0).sum(),
                    "zeros_pct": round((col_clean == 0).sum() / len(col_clean) * 100, 2),
                    "negative": (col_clean < 0).sum(),
                    "negative_pct": round((col_clean < 0).sum() / len(col_clean) * 100, 2),
                })

                # Distribution type detection
                profile["distribution_type"] = self._detect_distribution(col_clean)

        # Categorical column statistics
        else:
            value_counts = col_data.value_counts()
            profile.update({
                "top_values": value_counts.head(10).to_dict(),
                "mode": col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None,
                "mode_frequency": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                "mode_pct": round(
                    value_counts.iloc[0] / len(col_data) * 100, 2
                ) if len(value_counts) > 0 else 0,
            })

        return profile

    def _detect_distribution(self, data: pd.Series) -> str:
        """Detect the type of distribution."""
        if len(data) < 20:
            return "insufficient_data"

        skewness = data.skew()
        kurtosis = data.kurtosis()

        # Normality test
        try:
            _, p_normal = stats.normaltest(data.sample(min(5000, len(data))))
            is_normal = p_normal > 0.05
        except Exception:
            is_normal = False

        if is_normal:
            return "normal"
        elif abs(skewness) < 0.5 and abs(kurtosis) < 1:
            return "approximately_normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 3:
            return "leptokurtic"
        elif kurtosis < -1:
            return "platykurtic"
        else:
            return "non_normal"

    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data."""
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)

        # Missing data patterns (which combinations of columns have missing)
        missing_patterns = df.isnull().apply(lambda x: tuple(x), axis=1).value_counts()

        # Correlation of missingness between columns
        missing_matrix = df.isnull().astype(int)
        missing_corr = {}

        if len(missing_cols) > 1:
            corr_matrix = missing_matrix[missing_cols.index].corr()
            for i, col1 in enumerate(missing_cols.index):
                for col2 in list(missing_cols.index)[i+1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.3:  # Only report notable correlations
                        missing_corr[f"{col1}_vs_{col2}"] = round(corr, 4)

        return {
            "columns_with_missing": missing_cols.to_dict(),
            "n_complete_rows": (~df.isnull().any(axis=1)).sum(),
            "complete_rows_pct": round(
                (~df.isnull().any(axis=1)).sum() / len(df) * 100, 2
            ),
            "n_patterns": len(missing_patterns),
            "top_patterns": missing_patterns.head(5).to_dict(),
            "missing_correlations": missing_corr,
        }

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distributions = {}

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue

            # Basic distribution info
            distributions[col] = {
                "type": self._detect_distribution(col_data),
                "skewness": round(col_data.skew(), 4),
                "kurtosis": round(col_data.kurtosis(), 4),
            }

            # Normality tests
            try:
                if len(col_data) < 5000:
                    _, shapiro_p = stats.shapiro(col_data.sample(min(5000, len(col_data))))
                    distributions[col]["shapiro_p_value"] = round(shapiro_p, 6)
            except Exception:
                pass

            # Percentile values
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            distributions[col]["percentiles"] = {
                p: round(col_data.quantile(p/100), 4) for p in percentiles
            }

        return distributions

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return {"status": "insufficient_numeric_columns"}

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()

        # Find strong correlations
        strong_correlations = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.5:
                        strong_correlations.append({
                            "feature_1": col1,
                            "feature_2": col2,
                            "correlation": round(corr, 4),
                            "strength": "strong" if abs(corr) > 0.7 else "moderate",
                        })

        # Sort by absolute correlation
        strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "correlation_matrix": corr_matrix.round(4).to_dict(),
            "strong_correlations": strong_correlations[:20],  # Top 20
            "n_strong_correlations": len([c for c in strong_correlations if abs(c["correlation"]) > 0.7]),
            "n_moderate_correlations": len([c for c in strong_correlations if 0.5 < abs(c["correlation"]) <= 0.7]),
        }

    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue

            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers_low = (col_data < lower_bound).sum()
            outliers_high = (col_data > upper_bound).sum()
            total_outliers = outliers_low + outliers_high

            # Z-score based outliers (|z| > 3)
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = (z_scores > 3).sum()

            outlier_analysis[col] = {
                "iqr_method": {
                    "lower_bound": round(lower_bound, 4),
                    "upper_bound": round(upper_bound, 4),
                    "n_outliers_low": int(outliers_low),
                    "n_outliers_high": int(outliers_high),
                    "total_outliers": int(total_outliers),
                    "outlier_pct": round(total_outliers / len(col_data) * 100, 2),
                },
                "zscore_method": {
                    "n_outliers": int(z_outliers),
                    "outlier_pct": round(z_outliers / len(col_data) * 100, 2),
                },
            }

        return outlier_analysis

    def _calculate_quality_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall data quality score."""
        scores = {}

        # Completeness (% non-missing)
        completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        scores["completeness"] = round(completeness * 100, 2)

        # Uniqueness (no duplicates)
        uniqueness = 1 - (df.duplicated().sum() / len(df))
        scores["uniqueness"] = round(uniqueness * 100, 2)

        # Validity (% of values in valid range for numeric cols)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            valid_count = 0
            total_count = 0
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    q1, q3 = col_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    valid = ((col_data >= q1 - 3*iqr) & (col_data <= q3 + 3*iqr)).sum()
                    valid_count += valid
                    total_count += len(col_data)
            validity = valid_count / total_count if total_count > 0 else 1
        else:
            validity = 1
        scores["validity"] = round(validity * 100, 2)

        # Overall score (weighted average)
        scores["overall"] = round(
            scores["completeness"] * 0.4 +
            scores["uniqueness"] * 0.3 +
            scores["validity"] * 0.3,
            2
        )

        # Quality grade
        overall = scores["overall"]
        if overall >= 90:
            scores["grade"] = "A"
        elif overall >= 80:
            scores["grade"] = "B"
        elif overall >= 70:
            scores["grade"] = "C"
        elif overall >= 60:
            scores["grade"] = "D"
        else:
            scores["grade"] = "F"

        return scores

    def _analyze_target(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze target variable."""
        target = df[target_column]
        profile = self._profile_column(df, target_column)

        is_classification = target.nunique() <= 10

        analysis = {
            "profile": profile,
            "task_type": "classification" if is_classification else "regression",
        }

        if is_classification:
            value_counts = target.value_counts()
            analysis["class_distribution"] = value_counts.to_dict()
            analysis["class_percentages"] = (value_counts / len(target) * 100).round(2).to_dict()

            # Class imbalance ratio
            if len(value_counts) >= 2:
                majority = value_counts.iloc[0]
                minority = value_counts.iloc[-1]
                analysis["imbalance_ratio"] = round(majority / minority, 2)
                analysis["is_imbalanced"] = (majority / minority) > 3

        return analysis

    def _analyze_feature_target(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Analyze relationships between features and target."""
        relationships = {}
        target = df[target_column]
        is_classification = target.nunique() <= 10

        for col in df.columns:
            if col == target_column:
                continue

            col_data = df[col]
            relationship = {"column_type": str(col_data.dtype)}

            # Numeric feature
            if pd.api.types.is_numeric_dtype(col_data):
                valid_mask = col_data.notna() & target.notna()
                if valid_mask.sum() > 10:
                    try:
                        corr, p_value = stats.pearsonr(
                            col_data[valid_mask], target[valid_mask]
                        )
                        relationship["correlation"] = round(corr, 4)
                        relationship["p_value"] = round(p_value, 6)
                        relationship["is_significant"] = p_value < 0.05

                        # Effect size interpretation
                        abs_corr = abs(corr)
                        if abs_corr < 0.1:
                            relationship["effect_size"] = "negligible"
                        elif abs_corr < 0.3:
                            relationship["effect_size"] = "small"
                        elif abs_corr < 0.5:
                            relationship["effect_size"] = "medium"
                        else:
                            relationship["effect_size"] = "large"
                    except Exception:
                        pass

                    # ANOVA for classification targets
                    if is_classification:
                        try:
                            groups = [
                                col_data[valid_mask][target[valid_mask] == cls]
                                for cls in target.unique()
                            ]
                            groups = [g for g in groups if len(g) > 0]
                            if len(groups) >= 2:
                                f_stat, p_val = stats.f_oneway(*groups)
                                relationship["anova_f"] = round(f_stat, 4)
                                relationship["anova_p"] = round(p_val, 6)
                        except Exception:
                            pass

            relationships[col] = relationship

        return relationships

    def generate_report(self) -> str:
        """Generate comprehensive EDA report as text."""
        lines = []
        r = self.analysis_results

        lines.append("=" * 80)
        lines.append("COMPREHENSIVE EXPLORATORY DATA ANALYSIS REPORT")
        lines.append("=" * 80)

        # Dataset Overview
        lines.append("\n" + "=" * 40)
        lines.append("1. DATASET OVERVIEW")
        lines.append("=" * 40)
        overview = r.get("dataset_overview", {})
        lines.append(f"  Rows: {overview.get('n_rows', 'N/A'):,}")
        lines.append(f"  Columns: {overview.get('n_columns', 'N/A')}")
        lines.append(f"  Memory Usage: {overview.get('memory_mb', 'N/A')} MB")
        lines.append(f"  Numeric Columns: {overview.get('n_numeric', 'N/A')}")
        lines.append(f"  Categorical Columns: {overview.get('n_categorical', 'N/A')}")
        lines.append(f"  Duplicates: {overview.get('n_duplicates', 'N/A')} ({overview.get('duplicate_percentage', 'N/A')}%)")
        lines.append(f"  Missing Values: {overview.get('total_missing', 'N/A'):,} ({overview.get('missing_percentage', 'N/A')}%)")

        # Data Quality
        lines.append("\n" + "=" * 40)
        lines.append("2. DATA QUALITY SCORE")
        lines.append("=" * 40)
        quality = r.get("data_quality_score", {})
        lines.append(f"  Overall Score: {quality.get('overall', 'N/A')}% (Grade: {quality.get('grade', 'N/A')})")
        lines.append(f"  Completeness: {quality.get('completeness', 'N/A')}%")
        lines.append(f"  Uniqueness: {quality.get('uniqueness', 'N/A')}%")
        lines.append(f"  Validity: {quality.get('validity', 'N/A')}%")

        # Missing Data
        lines.append("\n" + "=" * 40)
        lines.append("3. MISSING DATA ANALYSIS")
        lines.append("=" * 40)
        missing = r.get("missing_patterns", {})
        lines.append(f"  Complete Rows: {missing.get('n_complete_rows', 'N/A'):,} ({missing.get('complete_rows_pct', 'N/A')}%)")
        lines.append(f"  Missing Patterns: {missing.get('n_patterns', 'N/A')}")
        lines.append("\n  Columns with Missing Values:")
        for col, count in list(missing.get("columns_with_missing", {}).items())[:10]:
            lines.append(f"    - {col}: {count:,}")

        # Correlations
        lines.append("\n" + "=" * 40)
        lines.append("4. CORRELATION ANALYSIS")
        lines.append("=" * 40)
        corr = r.get("correlation_analysis", {})
        lines.append(f"  Strong Correlations (|r| > 0.7): {corr.get('n_strong_correlations', 'N/A')}")
        lines.append(f"  Moderate Correlations (0.5 < |r| ≤ 0.7): {corr.get('n_moderate_correlations', 'N/A')}")
        lines.append("\n  Top Correlations:")
        for item in corr.get("strong_correlations", [])[:10]:
            lines.append(f"    - {item['feature_1']} vs {item['feature_2']}: {item['correlation']}")

        # Target Analysis
        if "target_analysis" in r:
            lines.append("\n" + "=" * 40)
            lines.append("5. TARGET VARIABLE ANALYSIS")
            lines.append("=" * 40)
            target = r.get("target_analysis", {})
            lines.append(f"  Task Type: {target.get('task_type', 'N/A')}")
            if target.get('task_type') == 'classification':
                lines.append(f"  Classes: {len(target.get('class_distribution', {}))}")
                lines.append(f"  Imbalance Ratio: {target.get('imbalance_ratio', 'N/A')}")
                lines.append(f"  Is Imbalanced: {target.get('is_imbalanced', 'N/A')}")
                lines.append("\n  Class Distribution:")
                for cls, count in target.get("class_distribution", {}).items():
                    pct = target.get("class_percentages", {}).get(cls, 0)
                    lines.append(f"    - {cls}: {count:,} ({pct}%)")

        lines.append("\n" + "=" * 80)
        lines.append("END OF EDA REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def export_report(self, filepath: str) -> None:
        """Export EDA report to file."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
