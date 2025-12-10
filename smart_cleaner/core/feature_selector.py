"""
Feature Selection module for identifying important features.
Uses multiple methods: correlation, variance, mutual information, statistical tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats


class FeatureSelector:
    """
    Automated feature selection using multiple methods.
    Identifies redundant, low-variance, and irrelevant features.
    """

    def __init__(self):
        """Initialize selector with storage for selection results."""
        self.selection_report = {}
        self.selected_features = []
        self.removed_features = []

    def auto_select(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        importance_threshold: float = 0.01,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Automatically select features using multiple criteria.

        Steps:
        1. Remove near-zero variance features
        2. Remove highly correlated features (keep one)
        3. Rank features by importance (if target provided)
        4. Statistical tests for feature-target relationship

        Args:
            df: Input DataFrame
            target_column: Target variable for importance analysis
            variance_threshold: Min variance to keep feature
            correlation_threshold: Max correlation between features
            importance_threshold: Min importance score to keep feature

        Returns:
            Tuple of (DataFrame with selected features, selection report)
        """
        df_result = df.copy()
        report = {
            "original_features": list(df.columns),
            "low_variance_removed": [],
            "high_correlation_removed": [],
            "low_importance_removed": [],
            "feature_importance": {},
            "correlation_analysis": {},
            "statistical_tests": {},
            "final_features": [],
        }

        # Get numeric columns (excluding target)
        numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)

        # Step 1: Remove low variance features
        df_result, low_var_removed = self._remove_low_variance(
            df_result, numeric_cols, variance_threshold
        )
        report["low_variance_removed"] = low_var_removed

        # Update numeric columns list
        numeric_cols = [c for c in numeric_cols if c not in low_var_removed]

        # Step 2: Remove highly correlated features
        df_result, high_corr_removed, corr_analysis = self._remove_high_correlation(
            df_result, numeric_cols, correlation_threshold
        )
        report["high_correlation_removed"] = high_corr_removed
        report["correlation_analysis"] = corr_analysis

        # Update numeric columns list
        numeric_cols = [c for c in numeric_cols if c not in high_corr_removed]

        # Step 3: Calculate feature importance (if target provided)
        if target_column and target_column in df_result.columns:
            importance_scores, stat_tests = self._calculate_importance(
                df_result, numeric_cols, target_column
            )
            report["feature_importance"] = importance_scores
            report["statistical_tests"] = stat_tests

            # Remove low importance features
            low_importance = [
                col for col, score in importance_scores.items()
                if score < importance_threshold
            ]
            report["low_importance_removed"] = low_importance

        # Final feature list
        all_removed = (
            report["low_variance_removed"] +
            report["high_correlation_removed"] +
            report.get("low_importance_removed", [])
        )

        report["final_features"] = [
            col for col in df_result.columns
            if col not in all_removed
        ]

        self.selection_report = report
        self.selected_features = report["final_features"]
        self.removed_features = all_removed

        return df_result, report

    def _remove_low_variance(
        self,
        df: pd.DataFrame,
        columns: List[str],
        threshold: float
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with variance below threshold."""
        removed = []

        for col in columns:
            variance = df[col].var()
            if variance < threshold:
                removed.append(col)

        # Don't actually drop - just report
        return df, removed

    def _remove_high_correlation(
        self,
        df: pd.DataFrame,
        columns: List[str],
        threshold: float
    ) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Remove one of each pair of highly correlated features."""
        if len(columns) < 2:
            return df, [], {}

        # Calculate correlation matrix
        corr_matrix = df[columns].corr().abs()

        # Find highly correlated pairs
        high_corr_pairs = []
        removed = set()
        correlation_analysis = {}

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                corr = corr_matrix.loc[col1, col2]

                if corr > threshold:
                    high_corr_pairs.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": corr,
                    })

                    # Remove the one with lower variance (keep more informative)
                    var1 = df[col1].var()
                    var2 = df[col2].var()

                    if var1 < var2:
                        removed.add(col1)
                        correlation_analysis[col1] = {
                            "removed_due_to_correlation_with": col2,
                            "correlation": float(corr),
                        }
                    else:
                        removed.add(col2)
                        correlation_analysis[col2] = {
                            "removed_due_to_correlation_with": col1,
                            "correlation": float(corr),
                        }

        return df, list(removed), correlation_analysis

    def _calculate_importance(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_column: str
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Calculate feature importance using multiple methods.
        Combines correlation, mutual information, and statistical tests.
        """
        importance_scores = {}
        stat_tests = {}

        target = df[target_column]
        is_classification = target.nunique() <= 10

        for col in columns:
            feature = df[col]

            # Skip if too many nulls
            valid_mask = feature.notna() & target.notna()
            if valid_mask.sum() < 10:
                importance_scores[col] = 0.0
                continue

            feature_valid = feature[valid_mask]
            target_valid = target[valid_mask]

            scores = []

            # 1. Correlation-based importance
            try:
                corr, p_value = stats.pearsonr(feature_valid, target_valid)
                corr_importance = abs(corr)
                scores.append(corr_importance)

                stat_tests[col] = {
                    "pearson_correlation": float(corr),
                    "pearson_p_value": float(p_value),
                }
            except Exception:
                stat_tests[col] = {}

            # 2. Spearman correlation (for non-linear relationships)
            try:
                spearman_corr, spearman_p = stats.spearmanr(feature_valid, target_valid)
                scores.append(abs(spearman_corr))
                stat_tests[col]["spearman_correlation"] = float(spearman_corr)
                stat_tests[col]["spearman_p_value"] = float(spearman_p)
            except Exception:
                pass

            # 3. Statistical test based on target type
            if is_classification:
                # ANOVA F-test for classification
                try:
                    groups = [
                        feature_valid[target_valid == cls]
                        for cls in target_valid.unique()
                        if len(feature_valid[target_valid == cls]) > 0
                    ]
                    if len(groups) >= 2:
                        f_stat, anova_p = stats.f_oneway(*groups)
                        # Normalize F-statistic to 0-1 range
                        f_importance = min(f_stat / 100, 1.0)
                        scores.append(f_importance)
                        stat_tests[col]["anova_f_statistic"] = float(f_stat)
                        stat_tests[col]["anova_p_value"] = float(anova_p)
                except Exception:
                    pass
            else:
                # For regression, use R-squared as additional metric
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        feature_valid, target_valid
                    )
                    scores.append(r_value ** 2)
                    stat_tests[col]["r_squared"] = float(r_value ** 2)
                    stat_tests[col]["regression_p_value"] = float(p_value)
                except Exception:
                    pass

            # Combine scores
            if scores:
                importance_scores[col] = float(np.mean(scores))
            else:
                importance_scores[col] = 0.0

            # Add significance indicator
            p_values = [
                stat_tests[col].get("pearson_p_value", 1.0),
                stat_tests[col].get("anova_p_value", 1.0),
                stat_tests[col].get("regression_p_value", 1.0),
            ]
            min_p = min(p_values)
            stat_tests[col]["is_significant"] = min_p < 0.05
            stat_tests[col]["min_p_value"] = float(min_p)

        return importance_scores, stat_tests

    def get_feature_ranking(self) -> List[Tuple[str, float]]:
        """Get features ranked by importance."""
        importance = self.selection_report.get("feature_importance", {})
        ranking = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return ranking

    def get_top_features(self, n: int = 10) -> List[str]:
        """Get top N features by importance."""
        ranking = self.get_feature_ranking()
        return [feat for feat, score in ranking[:n]]

    @classmethod
    def calculate_mutual_information(
        cls,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        n_neighbors: int = 3,
    ) -> Dict[str, float]:
        """
        Calculate mutual information between features and target.

        Args:
            df: Input DataFrame
            feature_columns: Feature column names
            target_column: Target column name
            n_neighbors: Number of neighbors for MI estimation

        Returns:
            Dictionary of feature -> MI score
        """
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        except ImportError:
            return {}

        X = df[feature_columns].fillna(0)
        y = df[target_column]

        # Determine if classification or regression
        if y.nunique() <= 10:
            mi_scores = mutual_info_classif(X, y, n_neighbors=n_neighbors)
        else:
            mi_scores = mutual_info_regression(X, y, n_neighbors=n_neighbors)

        return dict(zip(feature_columns, mi_scores))

    @classmethod
    def chi_square_test(
        cls,
        df: pd.DataFrame,
        feature_column: str,
        target_column: str
    ) -> Dict[str, float]:
        """
        Chi-square test for categorical feature vs categorical target.

        Args:
            df: Input DataFrame
            feature_column: Categorical feature column
            target_column: Categorical target column

        Returns:
            Dictionary with chi2 statistic, p-value, and cramers_v
        """
        # Create contingency table
        contingency = pd.crosstab(df[feature_column], df[target_column])

        # Chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Cramer's V (effect size)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        return {
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "cramers_v": float(cramers_v),
            "is_significant": p_value < 0.05,
        }

    def get_selection_summary(self) -> str:
        """Get human-readable selection summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("FEATURE SELECTION SUMMARY")
        lines.append("=" * 60)

        report = self.selection_report

        lines.append(f"\nOriginal features: {len(report.get('original_features', []))}")
        lines.append(f"Final features: {len(report.get('final_features', []))}")

        lines.append(f"\nLow Variance Removed: {len(report.get('low_variance_removed', []))}")
        for feat in report.get('low_variance_removed', []):
            lines.append(f"  - {feat}")

        lines.append(f"\nHigh Correlation Removed: {len(report.get('high_correlation_removed', []))}")
        for feat, info in report.get('correlation_analysis', {}).items():
            lines.append(f"  - {feat} (corr={info['correlation']:.3f} with {info['removed_due_to_correlation_with']})")

        lines.append(f"\nLow Importance Removed: {len(report.get('low_importance_removed', []))}")
        for feat in report.get('low_importance_removed', []):
            lines.append(f"  - {feat}")

        lines.append("\nTop 10 Features by Importance:")
        ranking = self.get_feature_ranking()[:10]
        for i, (feat, score) in enumerate(ranking, 1):
            stat = report.get('statistical_tests', {}).get(feat, {})
            sig = "***" if stat.get('is_significant', False) else ""
            lines.append(f"  {i}. {feat}: {score:.4f} {sig}")

        lines.append("\n*** = statistically significant (p < 0.05)")

        return "\n".join(lines)

    def export_feature_report(self, filepath: str) -> None:
        """Export detailed feature analysis report."""
        lines = []

        lines.append("FEATURE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Feature importance table
        lines.append("FEATURE IMPORTANCE RANKING")
        lines.append("-" * 80)
        lines.append(f"{'Rank':<6}{'Feature':<40}{'Importance':<12}{'Significant':<12}")
        lines.append("-" * 80)

        for i, (feat, score) in enumerate(self.get_feature_ranking(), 1):
            stat = self.selection_report.get('statistical_tests', {}).get(feat, {})
            sig = "Yes" if stat.get('is_significant', False) else "No"
            lines.append(f"{i:<6}{feat:<40}{score:<12.4f}{sig:<12}")

        lines.append("")
        lines.append("STATISTICAL TESTS DETAIL")
        lines.append("-" * 80)

        for feat, tests in self.selection_report.get('statistical_tests', {}).items():
            lines.append(f"\n{feat}:")
            for test_name, value in tests.items():
                if isinstance(value, float):
                    lines.append(f"  {test_name}: {value:.6f}")
                else:
                    lines.append(f"  {test_name}: {value}")

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
