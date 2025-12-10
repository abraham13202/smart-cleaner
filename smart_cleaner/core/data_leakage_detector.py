"""
Data Leakage Detector.
Identifies potential data leakage issues before ML modeling.

Data leakage occurs when information from outside the training dataset
is used to create the model, leading to overly optimistic performance.

Types of leakage detected:
1. Target Leakage: Features that contain info about target after the fact
2. Train-Test Contamination: Overlapping data between train/test
3. Future Information Leakage: Using future data to predict past
4. Proxy Features: Features that are proxies for the target
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class DataLeakageDetector:
    """
    Detects potential data leakage in datasets before ML modeling.

    This is CRITICAL for ensuring model validity and preventing
    unrealistic performance metrics.
    """

    def __init__(self):
        """Initialize the detector."""
        self.leakage_report = {}
        self.warnings = []

    def full_check(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: Optional[str] = None,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data leakage check.

        Args:
            df: Full dataset
            target_column: Target/label column
            date_column: Date column for temporal validation
            train_df: Training set (if already split)
            test_df: Test set (if already split)

        Returns:
            Complete leakage report
        """
        self.warnings = []

        report = {
            "check_timestamp": datetime.now().isoformat(),
            "target_column": target_column,
            "target_leakage": self._check_target_leakage(df, target_column),
            "proxy_features": self._check_proxy_features(df, target_column),
            "temporal_leakage": self._check_temporal_leakage(df, target_column, date_column),
            "high_cardinality_leak": self._check_high_cardinality_leak(df, target_column),
            "warnings": self.warnings,
            "leakage_risk": "LOW",
            "recommendations": [],
        }

        # Check train-test contamination if splits provided
        if train_df is not None and test_df is not None:
            report["train_test_contamination"] = self._check_train_test_contamination(
                train_df, test_df
            )

        # Calculate overall risk
        report["leakage_risk"] = self._calculate_risk(report)
        report["recommendations"] = self._generate_recommendations(report)

        self.leakage_report = report
        return report

    def _check_target_leakage(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Check for target leakage - features that perfectly or nearly
        perfectly predict the target (too good to be true).

        This often happens when a feature is derived from the target
        or contains post-hoc information.
        """
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found"}

        suspicious_features = []
        perfect_predictors = []

        target = df[target_column]

        for col in df.columns:
            if col == target_column:
                continue

            try:
                # For numeric columns, check correlation
                if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target):
                    corr = abs(df[col].corr(target))
                    if corr > 0.95:
                        perfect_predictors.append({
                            "column": col,
                            "correlation": round(corr, 4),
                            "reason": f"Extremely high correlation ({corr:.4f}) with target"
                        })
                        self.warnings.append({
                            "type": "target_leakage",
                            "severity": "CRITICAL",
                            "column": col,
                            "message": f"'{col}' has {corr:.2%} correlation with target - likely leakage!"
                        })
                    elif corr > 0.85:
                        suspicious_features.append({
                            "column": col,
                            "correlation": round(corr, 4),
                            "reason": f"Very high correlation ({corr:.4f}) - investigate"
                        })

                # For categorical vs categorical target
                elif df[col].dtype == 'object' and target.dtype == 'object':
                    # Check if one column perfectly determines the other
                    cross_tab = pd.crosstab(df[col], target, normalize='index')
                    max_probs = cross_tab.max(axis=1)
                    if (max_probs > 0.99).any():
                        suspicious_features.append({
                            "column": col,
                            "reason": "Some categories perfectly predict target class"
                        })

            except Exception:
                pass

        # Check for columns with target-like names
        target_lower = target_column.lower()
        for col in df.columns:
            if col == target_column:
                continue
            col_lower = col.lower()
            if target_lower in col_lower or col_lower in target_lower:
                suspicious_features.append({
                    "column": col,
                    "reason": f"Column name similar to target '{target_column}'"
                })
                self.warnings.append({
                    "type": "name_similarity",
                    "severity": "HIGH",
                    "column": col,
                    "message": f"'{col}' has similar name to target - may be derived from it"
                })

        return {
            "perfect_predictors": perfect_predictors,
            "suspicious_features": suspicious_features,
            "risk_level": "CRITICAL" if perfect_predictors else "MEDIUM" if suspicious_features else "LOW"
        }

    def _check_proxy_features(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Check for proxy features that may indirectly leak target information.

        Examples:
        - 'outcome_code' when predicting 'outcome'
        - 'diagnosis_date' when predicting 'has_disease'
        - ID columns that encode target information
        """
        proxy_candidates = []

        # Common proxy patterns
        proxy_patterns = [
            'result', 'outcome', 'status', 'flag', 'indicator',
            'code', 'category', 'type', 'class', 'label'
        ]

        target_lower = target_column.lower()

        for col in df.columns:
            if col == target_column:
                continue

            col_lower = col.lower()

            # Check for proxy naming patterns
            for pattern in proxy_patterns:
                if pattern in col_lower and pattern not in target_lower:
                    # Check if column has suspicious relationship with target
                    if df[col].dtype == 'object' or df[col].nunique() <= 20:
                        proxy_candidates.append({
                            "column": col,
                            "pattern": pattern,
                            "reason": f"May be a proxy feature (contains '{pattern}')"
                        })
                    break

            # Check for ID columns that might encode target
            if 'id' in col_lower or col_lower.endswith('_id'):
                if df[col].nunique() > len(df) * 0.5:  # High cardinality ID
                    proxy_candidates.append({
                        "column": col,
                        "pattern": "high_cardinality_id",
                        "reason": "High-cardinality ID may encode target information"
                    })

        return {
            "proxy_candidates": proxy_candidates,
            "risk_level": "MEDIUM" if proxy_candidates else "LOW"
        }

    def _check_temporal_leakage(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check for temporal/time-based leakage.

        This occurs when:
        - Future information is used to predict past events
        - Date of outcome is included as feature
        - Time-dependent features leak future info
        """
        issues = []

        # Auto-detect date columns if not provided
        date_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['date', 'time', 'timestamp', 'created', 'updated', '_at', '_on']):
                date_columns.append(col)

        if date_column:
            date_columns = [date_column] + [c for c in date_columns if c != date_column]

        # Check for outcome dates
        target_lower = target_column.lower()
        for col in date_columns:
            col_lower = col.lower()

            # Check if date column is related to target
            if target_lower.replace('_', '') in col_lower.replace('_', ''):
                issues.append({
                    "column": col,
                    "type": "outcome_date",
                    "severity": "CRITICAL",
                    "message": f"'{col}' appears to be the date of the outcome - remove it!"
                })
                self.warnings.append({
                    "type": "temporal_leakage",
                    "severity": "CRITICAL",
                    "column": col,
                    "message": f"'{col}' is likely the outcome date - this causes leakage!"
                })

            # Check for future dates
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                future_dates = dates[dates > datetime.now()].count()
                if future_dates > 0:
                    issues.append({
                        "column": col,
                        "type": "future_dates",
                        "severity": "HIGH",
                        "message": f"'{col}' contains {future_dates} future dates"
                    })
            except Exception:
                pass

        # Check for features that might be post-hoc
        post_hoc_patterns = [
            'after', 'post', 'final', 'end', 'last', 'outcome',
            'result', 'total', 'sum', 'completed'
        ]
        for col in df.columns:
            if col == target_column:
                continue
            col_lower = col.lower()
            for pattern in post_hoc_patterns:
                if pattern in col_lower:
                    issues.append({
                        "column": col,
                        "type": "post_hoc_feature",
                        "severity": "MEDIUM",
                        "message": f"'{col}' may contain post-hoc information (contains '{pattern}')"
                    })
                    break

        return {
            "date_columns_found": date_columns,
            "issues": issues,
            "risk_level": "CRITICAL" if any(i['severity'] == 'CRITICAL' for i in issues) else
                         "HIGH" if any(i['severity'] == 'HIGH' for i in issues) else
                         "MEDIUM" if issues else "LOW"
        }

    def _check_high_cardinality_leak(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Check for high-cardinality features that may cause leakage.

        High-cardinality categorical features (like customer_id) can
        memorize the target during training.
        """
        high_cardinality_cols = []

        for col in df.columns:
            if col == target_column:
                continue

            cardinality = df[col].nunique()
            cardinality_ratio = cardinality / len(df)

            if cardinality_ratio > 0.5 and cardinality > 100:
                high_cardinality_cols.append({
                    "column": col,
                    "unique_values": cardinality,
                    "cardinality_ratio": round(cardinality_ratio, 4),
                    "recommendation": "Consider encoding or removing"
                })

                if cardinality_ratio > 0.9:
                    self.warnings.append({
                        "type": "high_cardinality",
                        "severity": "HIGH",
                        "column": col,
                        "message": f"'{col}' has {cardinality_ratio:.1%} unique values - may cause memorization"
                    })

        return {
            "high_cardinality_columns": high_cardinality_cols,
            "risk_level": "HIGH" if any(c['cardinality_ratio'] > 0.9 for c in high_cardinality_cols) else
                         "MEDIUM" if high_cardinality_cols else "LOW"
        }

    def _check_train_test_contamination(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check for train-test contamination (overlapping data).
        """
        # Check for exact row duplicates
        train_str = train_df.astype(str).apply(lambda x: '|'.join(x), axis=1)
        test_str = test_df.astype(str).apply(lambda x: '|'.join(x), axis=1)

        overlapping = set(train_str) & set(test_str)
        overlap_count = len(overlapping)

        if overlap_count > 0:
            self.warnings.append({
                "type": "train_test_contamination",
                "severity": "CRITICAL",
                "column": "ALL",
                "message": f"{overlap_count} rows appear in both train and test sets!"
            })

        return {
            "overlapping_rows": overlap_count,
            "overlap_percentage": round(overlap_count / len(test_df) * 100, 2) if len(test_df) > 0 else 0,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "risk_level": "CRITICAL" if overlap_count > 0 else "LOW"
        }

    def _calculate_risk(self, report: Dict) -> str:
        """Calculate overall leakage risk."""
        risk_levels = []

        if 'target_leakage' in report:
            risk_levels.append(report['target_leakage'].get('risk_level', 'LOW'))
        if 'proxy_features' in report:
            risk_levels.append(report['proxy_features'].get('risk_level', 'LOW'))
        if 'temporal_leakage' in report:
            risk_levels.append(report['temporal_leakage'].get('risk_level', 'LOW'))
        if 'high_cardinality_leak' in report:
            risk_levels.append(report['high_cardinality_leak'].get('risk_level', 'LOW'))
        if 'train_test_contamination' in report:
            risk_levels.append(report['train_test_contamination'].get('risk_level', 'LOW'))

        if 'CRITICAL' in risk_levels:
            return 'CRITICAL'
        elif 'HIGH' in risk_levels:
            return 'HIGH'
        elif 'MEDIUM' in risk_levels:
            return 'MEDIUM'
        return 'LOW'

    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Target leakage recommendations
        target_leakage = report.get('target_leakage', {})
        for pp in target_leakage.get('perfect_predictors', []):
            recommendations.append(f"REMOVE '{pp['column']}' - it has {pp['correlation']:.2%} correlation with target (leakage)")

        for sf in target_leakage.get('suspicious_features', []):
            recommendations.append(f"INVESTIGATE '{sf['column']}' - {sf['reason']}")

        # Temporal leakage recommendations
        temporal = report.get('temporal_leakage', {})
        for issue in temporal.get('issues', []):
            if issue['severity'] == 'CRITICAL':
                recommendations.append(f"REMOVE '{issue['column']}' - {issue['message']}")

        # High cardinality recommendations
        high_card = report.get('high_cardinality_leak', {})
        for col in high_card.get('high_cardinality_columns', []):
            if col['cardinality_ratio'] > 0.9:
                recommendations.append(f"REMOVE or ENCODE '{col['column']}' - {col['unique_values']} unique values")

        # Train-test contamination
        if report.get('train_test_contamination', {}).get('overlapping_rows', 0) > 0:
            recommendations.append("CRITICAL: Remove overlapping rows between train and test sets")

        return recommendations

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate leakage report as text."""
        if not self.leakage_report:
            return "No leakage check performed. Run full_check() first."

        r = self.leakage_report
        lines = []

        lines.append("=" * 80)
        lines.append("DATA LEAKAGE DETECTION REPORT")
        lines.append("=" * 80)
        lines.append(f"Check performed: {r['check_timestamp']}")
        lines.append(f"Target column: {r['target_column']}")
        lines.append("")

        # Overall Risk
        risk_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
        lines.append(f"OVERALL LEAKAGE RISK: {risk_emoji.get(r['leakage_risk'], '')} {r['leakage_risk']}")
        lines.append("")

        # Warnings
        if r['warnings']:
            lines.append("-" * 40)
            lines.append("WARNINGS")
            lines.append("-" * 40)
            for w in r['warnings']:
                lines.append(f"  [{w['severity']}] {w['column']}: {w['message']}")
            lines.append("")

        # Recommendations
        if r['recommendations']:
            lines.append("-" * 40)
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 40)
            for i, rec in enumerate(r['recommendations'], 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        lines.append("=" * 80)

        report_text = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text
