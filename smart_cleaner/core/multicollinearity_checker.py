"""
Multicollinearity Checker.
Detects and handles multicollinearity in datasets.

Multicollinearity occurs when independent variables are highly correlated,
which can cause:
- Unstable coefficient estimates
- Inflated standard errors
- Difficulty interpreting feature importance

Methods:
- Variance Inflation Factor (VIF)
- Correlation matrix analysis
- Condition number
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings


class MulticollinearityChecker:
    """
    Comprehensive multicollinearity detection and handling.

    This is essential for:
    - Linear regression models
    - Logistic regression
    - Any model where coefficient interpretation matters
    """

    def __init__(self):
        """Initialize the checker."""
        self.vif_results = {}
        self.correlation_results = {}
        self.recommendations = []

    def full_check(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        vif_threshold: float = 5.0,
        correlation_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Perform comprehensive multicollinearity check.

        Args:
            df: DataFrame with features
            target_column: Target column to exclude from analysis
            vif_threshold: VIF above this indicates multicollinearity (default: 5.0)
            correlation_threshold: Correlation above this is flagged (default: 0.8)

        Returns:
            Complete multicollinearity report
        """
        self.recommendations = []

        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if target_column and target_column in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_column])

        if len(numeric_df.columns) < 2:
            return {
                "error": "Need at least 2 numeric columns for multicollinearity check",
                "numeric_columns": list(numeric_df.columns)
            }

        report = {
            "numeric_columns_analyzed": list(numeric_df.columns),
            "vif_analysis": self._calculate_vif(numeric_df, vif_threshold),
            "correlation_analysis": self._analyze_correlations(numeric_df, correlation_threshold),
            "condition_number": self._calculate_condition_number(numeric_df),
            "recommendations": [],
            "columns_to_remove": [],
            "multicollinearity_severity": "LOW",
        }

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report, vif_threshold)
        report["columns_to_remove"] = self._get_columns_to_remove(report)
        report["multicollinearity_severity"] = self._calculate_severity(report, vif_threshold)

        return report

    def _calculate_vif(
        self,
        df: pd.DataFrame,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Calculate Variance Inflation Factor for each feature.

        VIF = 1 / (1 - R²) where R² is from regressing feature on all others

        Interpretation:
        - VIF = 1: No correlation
        - VIF 1-5: Moderate correlation
        - VIF > 5: High correlation (problematic)
        - VIF > 10: Severe multicollinearity
        """
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            return {"error": "sklearn required for VIF calculation"}

        # Remove columns with missing values for VIF calculation
        df_clean = df.dropna()
        if len(df_clean) < 10:
            return {"error": "Insufficient data after removing missing values"}

        vif_data = []
        high_vif_columns = []

        for i, col in enumerate(df_clean.columns):
            # Get X (all other columns) and y (current column)
            X = df_clean.drop(columns=[col])
            y = df_clean[col]

            # Skip if constant
            if y.std() == 0:
                vif_data.append({
                    "column": col,
                    "vif": float('inf'),
                    "status": "CONSTANT"
                })
                continue

            try:
                # Fit linear regression
                model = LinearRegression()
                model.fit(X, y)
                r_squared = model.score(X, y)

                # Calculate VIF
                vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')

                status = "OK"
                if vif > 10:
                    status = "SEVERE"
                    high_vif_columns.append(col)
                elif vif > threshold:
                    status = "HIGH"
                    high_vif_columns.append(col)
                elif vif > 2:
                    status = "MODERATE"

                vif_data.append({
                    "column": col,
                    "vif": round(vif, 2),
                    "r_squared": round(r_squared, 4),
                    "status": status
                })
            except Exception as e:
                vif_data.append({
                    "column": col,
                    "vif": None,
                    "error": str(e)
                })

        # Sort by VIF descending
        vif_data.sort(key=lambda x: x.get('vif', 0) if x.get('vif') != float('inf') else 999, reverse=True)

        return {
            "vif_scores": vif_data,
            "high_vif_columns": high_vif_columns,
            "threshold_used": threshold,
            "columns_above_threshold": len(high_vif_columns),
        }

    def _analyze_correlations(
        self,
        df: pd.DataFrame,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Analyze correlation matrix for high correlations.
        """
        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Find high correlations
        high_correlations = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Upper triangle only
                    corr = abs(corr_matrix.loc[col1, col2])
                    if corr >= threshold:
                        high_correlations.append({
                            "column_1": col1,
                            "column_2": col2,
                            "correlation": round(corr, 4),
                            "recommendation": f"Consider removing one of these columns"
                        })

        # Sort by correlation
        high_correlations.sort(key=lambda x: x['correlation'], reverse=True)

        # Find columns involved in most high correlations
        column_counts = {}
        for hc in high_correlations:
            column_counts[hc['column_1']] = column_counts.get(hc['column_1'], 0) + 1
            column_counts[hc['column_2']] = column_counts.get(hc['column_2'], 0) + 1

        return {
            "high_correlations": high_correlations,
            "threshold_used": threshold,
            "total_high_correlations": len(high_correlations),
            "most_correlated_columns": sorted(
                column_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }

    def _calculate_condition_number(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate condition number of the feature matrix.

        Condition number measures how sensitive the solution is to small changes.

        Interpretation:
        - < 30: Good
        - 30-100: Moderate multicollinearity
        - > 100: Severe multicollinearity
        """
        try:
            # Standardize the data first
            df_standardized = (df - df.mean()) / df.std()
            df_standardized = df_standardized.dropna()

            if len(df_standardized) < 2:
                return {"error": "Insufficient data"}

            # Calculate condition number using SVD
            _, s, _ = np.linalg.svd(df_standardized.values)
            condition_number = s.max() / s.min() if s.min() > 0 else float('inf')

            if condition_number < 30:
                status = "GOOD"
            elif condition_number < 100:
                status = "MODERATE"
            else:
                status = "SEVERE"

            return {
                "condition_number": round(condition_number, 2),
                "status": status,
                "interpretation": {
                    "GOOD": "No concerning multicollinearity",
                    "MODERATE": "Some multicollinearity present",
                    "SEVERE": "Severe multicollinearity - action needed"
                }[status]
            }
        except Exception as e:
            return {"error": str(e)}

    def _generate_recommendations(
        self,
        report: Dict,
        vif_threshold: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # VIF-based recommendations
        vif_analysis = report.get('vif_analysis', {})
        for vif_item in vif_analysis.get('vif_scores', [])[:5]:
            if vif_item.get('status') == 'SEVERE':
                recommendations.append(
                    f"REMOVE '{vif_item['column']}' - VIF of {vif_item['vif']} indicates severe multicollinearity"
                )
            elif vif_item.get('status') == 'HIGH':
                recommendations.append(
                    f"CONSIDER removing '{vif_item['column']}' - VIF of {vif_item['vif']} is above threshold"
                )

        # Correlation-based recommendations
        corr_analysis = report.get('correlation_analysis', {})
        for hc in corr_analysis.get('high_correlations', [])[:5]:
            recommendations.append(
                f"REMOVE one of '{hc['column_1']}' or '{hc['column_2']}' (correlation: {hc['correlation']})"
            )

        # Condition number recommendation
        cond = report.get('condition_number', {})
        if cond.get('status') == 'SEVERE':
            recommendations.append(
                f"SEVERE multicollinearity detected (condition number: {cond['condition_number']}). "
                "Consider PCA or removing correlated features."
            )

        return recommendations

    def _get_columns_to_remove(self, report: Dict) -> List[str]:
        """
        Get list of columns recommended for removal.

        Uses a greedy approach to remove minimum columns that resolve multicollinearity.
        """
        columns_to_remove = set()

        # Add columns with severe VIF
        vif_analysis = report.get('vif_analysis', {})
        for vif_item in vif_analysis.get('vif_scores', []):
            if vif_item.get('status') == 'SEVERE':
                columns_to_remove.add(vif_item['column'])

        # For high correlations, remove one from each pair
        corr_analysis = report.get('correlation_analysis', {})
        for hc in corr_analysis.get('high_correlations', []):
            # If neither is already marked for removal, remove the one with higher VIF
            if hc['column_1'] not in columns_to_remove and hc['column_2'] not in columns_to_remove:
                # Get VIF scores
                vif_scores = {v['column']: v.get('vif', 0) for v in vif_analysis.get('vif_scores', [])}
                vif1 = vif_scores.get(hc['column_1'], 0)
                vif2 = vif_scores.get(hc['column_2'], 0)

                # Remove the one with higher VIF
                if vif1 > vif2:
                    columns_to_remove.add(hc['column_1'])
                else:
                    columns_to_remove.add(hc['column_2'])

        return list(columns_to_remove)

    def _calculate_severity(self, report: Dict, vif_threshold: float) -> str:
        """Calculate overall severity level."""
        # Check VIF
        vif_analysis = report.get('vif_analysis', {})
        severe_vif = any(v.get('status') == 'SEVERE' for v in vif_analysis.get('vif_scores', []))
        high_vif = len(vif_analysis.get('high_vif_columns', []))

        # Check condition number
        cond = report.get('condition_number', {})
        severe_condition = cond.get('status') == 'SEVERE'

        # Check correlations
        corr_analysis = report.get('correlation_analysis', {})
        many_high_corr = corr_analysis.get('total_high_correlations', 0) > 5

        if severe_vif or severe_condition:
            return "SEVERE"
        elif high_vif > 3 or many_high_corr:
            return "HIGH"
        elif high_vif > 0 or corr_analysis.get('total_high_correlations', 0) > 0:
            return "MODERATE"
        return "LOW"

    def remove_multicollinear_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        vif_threshold: float = 5.0
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Automatically remove features with high multicollinearity.

        Args:
            df: Input DataFrame
            target_column: Target column to preserve
            vif_threshold: VIF threshold for removal

        Returns:
            Tuple of (cleaned DataFrame, list of removed columns)
        """
        report = self.full_check(df, target_column, vif_threshold)
        columns_to_remove = report.get('columns_to_remove', [])

        # Never remove target column
        if target_column and target_column in columns_to_remove:
            columns_to_remove.remove(target_column)

        df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')

        return df_cleaned, columns_to_remove

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate multicollinearity report as text."""
        # Note: This requires full_check to be run first
        lines = []
        lines.append("=" * 80)
        lines.append("MULTICOLLINEARITY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Run full_check() first to populate this report.")
        lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text
