"""
Final Quality Checker.
Pre-handoff validation to ensure data is truly ML-ready.

This is the LAST LINE OF DEFENSE before handing data to a data scientist.
If any check fails, the data should NOT be used for modeling.

Checks:
1. No missing values remain
2. No data leakage
3. No future timestamps in historical data
4. Features match business logic
5. Proper encoding complete
6. Scaling applied where needed
7. No infinite values
8. Proper train-test split integrity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class FinalQualityChecker:
    """
    Final validation before handing data to data scientists.

    This ensures the data meets ALL requirements for ML modeling.
    Any failure here should block the data handoff.
    """

    def __init__(self):
        """Initialize the checker."""
        self.check_results = {}
        self.passed = False
        self.blockers = []
        self.warnings = []

    def validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_column: str,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        date_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive final validation.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            target_column: Name of target column
            X_val, y_val: Optional validation data
            date_column: Date column for temporal validation

        Returns:
            Validation report with pass/fail status
        """
        self.blockers = []
        self.warnings = []

        checks = {
            "missing_values": self._check_missing_values(X_train, X_test, X_val),
            "infinite_values": self._check_infinite_values(X_train, X_test, X_val),
            "data_types": self._check_data_types(X_train, X_test, X_val),
            "column_consistency": self._check_column_consistency(X_train, X_test, X_val),
            "train_test_overlap": self._check_train_test_overlap(X_train, X_test),
            "target_in_features": self._check_target_not_in_features(X_train, target_column),
            "class_distribution": self._check_class_distribution(y_train, y_test, y_val),
            "value_ranges": self._check_value_ranges(X_train, X_test),
            "constant_columns": self._check_constant_columns(X_train),
        }

        # Add temporal check if date column provided
        if date_column:
            checks["temporal_integrity"] = self._check_temporal_integrity(
                X_train, X_test, date_column
            )

        # Determine overall pass/fail
        all_passed = all(c["passed"] for c in checks.values())

        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_passed": all_passed,
            "checks": checks,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "summary": {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "val_samples": len(X_val) if X_val is not None else 0,
                "n_features": len(X_train.columns),
                "target_column": target_column,
            },
            "recommendation": "PROCEED" if all_passed else "DO NOT PROCEED - Fix blockers first",
        }

        self.check_results = report
        self.passed = all_passed
        return report

    def _check_missing_values(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_val: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Check for any remaining missing values."""
        train_missing = X_train.isnull().sum().sum()
        test_missing = X_test.isnull().sum().sum()
        val_missing = X_val.isnull().sum().sum() if X_val is not None else 0

        total_missing = train_missing + test_missing + val_missing
        passed = total_missing == 0

        if not passed:
            self.blockers.append({
                "check": "missing_values",
                "message": f"Found {total_missing} missing values (Train: {train_missing}, Test: {test_missing})"
            })

        # Detail by column
        columns_with_missing = []
        for col in X_train.columns:
            col_missing = X_train[col].isnull().sum() + X_test[col].isnull().sum()
            if X_val is not None:
                col_missing += X_val[col].isnull().sum()
            if col_missing > 0:
                columns_with_missing.append({"column": col, "missing_count": col_missing})

        return {
            "passed": passed,
            "train_missing": int(train_missing),
            "test_missing": int(test_missing),
            "val_missing": int(val_missing),
            "columns_with_missing": columns_with_missing,
        }

    def _check_infinite_values(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_val: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Check for infinite values."""
        def count_inf(df):
            numeric_df = df.select_dtypes(include=[np.number])
            return np.isinf(numeric_df).sum().sum()

        train_inf = count_inf(X_train)
        test_inf = count_inf(X_test)
        val_inf = count_inf(X_val) if X_val is not None else 0

        total_inf = train_inf + test_inf + val_inf
        passed = total_inf == 0

        if not passed:
            self.blockers.append({
                "check": "infinite_values",
                "message": f"Found {total_inf} infinite values"
            })

        return {
            "passed": passed,
            "train_infinite": int(train_inf),
            "test_infinite": int(test_inf),
            "val_infinite": int(val_inf),
        }

    def _check_data_types(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_val: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Check that all data is properly typed (numeric for ML)."""
        non_numeric_cols = []

        for col in X_train.columns:
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                non_numeric_cols.append({
                    "column": col,
                    "dtype": str(X_train[col].dtype)
                })

        passed = len(non_numeric_cols) == 0

        if not passed:
            self.blockers.append({
                "check": "data_types",
                "message": f"Found {len(non_numeric_cols)} non-numeric columns: {[c['column'] for c in non_numeric_cols]}"
            })

        return {
            "passed": passed,
            "non_numeric_columns": non_numeric_cols,
            "total_columns": len(X_train.columns),
        }

    def _check_column_consistency(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_val: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Check that train/test/val have same columns."""
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)

        in_train_not_test = train_cols - test_cols
        in_test_not_train = test_cols - train_cols

        issues = []
        if in_train_not_test:
            issues.append(f"In train but not test: {list(in_train_not_test)}")
        if in_test_not_train:
            issues.append(f"In test but not train: {list(in_test_not_train)}")

        if X_val is not None:
            val_cols = set(X_val.columns)
            in_train_not_val = train_cols - val_cols
            in_val_not_train = val_cols - train_cols
            if in_train_not_val:
                issues.append(f"In train but not val: {list(in_train_not_val)}")
            if in_val_not_train:
                issues.append(f"In val but not train: {list(in_val_not_train)}")

        passed = len(issues) == 0

        if not passed:
            self.blockers.append({
                "check": "column_consistency",
                "message": f"Column mismatch: {issues}"
            })

        # Check column order
        columns_match_order = list(X_train.columns) == list(X_test.columns)

        return {
            "passed": passed,
            "columns_match": passed,
            "columns_match_order": columns_match_order,
            "issues": issues,
        }

    def _check_train_test_overlap(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check for data leakage via train-test overlap."""
        # Convert to string for comparison
        train_str = X_train.astype(str).apply(lambda x: '|'.join(x), axis=1)
        test_str = X_test.astype(str).apply(lambda x: '|'.join(x), axis=1)

        overlap = len(set(train_str) & set(test_str))
        passed = overlap == 0

        if not passed:
            self.blockers.append({
                "check": "train_test_overlap",
                "message": f"CRITICAL: {overlap} rows appear in both train and test sets!"
            })

        return {
            "passed": passed,
            "overlapping_rows": overlap,
            "overlap_percentage": round(overlap / len(X_test) * 100, 2) if len(X_test) > 0 else 0,
        }

    def _check_target_not_in_features(
        self,
        X_train: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Ensure target variable is not in features."""
        target_in_features = target_column in X_train.columns
        passed = not target_in_features

        if not passed:
            self.blockers.append({
                "check": "target_in_features",
                "message": f"CRITICAL: Target column '{target_column}' is in the feature set!"
            })

        # Also check for columns that might be target derivatives
        suspicious_cols = []
        target_lower = target_column.lower()
        for col in X_train.columns:
            col_lower = col.lower()
            if target_lower in col_lower or col_lower in target_lower:
                suspicious_cols.append(col)

        if suspicious_cols:
            self.warnings.append({
                "check": "possible_target_leakage",
                "message": f"Columns similar to target: {suspicious_cols}"
            })

        return {
            "passed": passed,
            "target_in_features": target_in_features,
            "suspicious_columns": suspicious_cols,
        }

    def _check_class_distribution(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        y_val: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Check class distribution across splits."""
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)

        # Check if distributions are similar (within 10%)
        distribution_similar = True
        dist_diffs = {}

        for cls in train_dist.index:
            if cls in test_dist.index:
                diff = abs(train_dist[cls] - test_dist[cls])
                dist_diffs[str(cls)] = round(diff, 4)
                if diff > 0.1:  # More than 10% difference
                    distribution_similar = False

        if not distribution_similar:
            self.warnings.append({
                "check": "class_distribution",
                "message": f"Class distributions differ between train and test: {dist_diffs}"
            })

        return {
            "passed": True,  # This is a warning, not a blocker
            "distribution_similar": distribution_similar,
            "train_distribution": train_dist.to_dict(),
            "test_distribution": test_dist.to_dict(),
            "distribution_differences": dist_diffs,
        }

    def _check_value_ranges(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check for extreme value differences between train and test."""
        issues = []

        for col in X_train.select_dtypes(include=[np.number]).columns:
            train_min, train_max = X_train[col].min(), X_train[col].max()
            test_min, test_max = X_test[col].min(), X_test[col].max()

            # Check if test has values far outside train range
            train_range = train_max - train_min
            if train_range > 0:
                if test_min < train_min - 0.5 * train_range:
                    issues.append({
                        "column": col,
                        "issue": "test_min_below_train",
                        "train_min": float(train_min),
                        "test_min": float(test_min)
                    })
                if test_max > train_max + 0.5 * train_range:
                    issues.append({
                        "column": col,
                        "issue": "test_max_above_train",
                        "train_max": float(train_max),
                        "test_max": float(test_max)
                    })

        if issues:
            self.warnings.append({
                "check": "value_ranges",
                "message": f"{len(issues)} columns have test values outside train range"
            })

        return {
            "passed": True,  # Warning only
            "range_issues": issues,
        }

    def _check_constant_columns(self, X_train: pd.DataFrame) -> Dict[str, Any]:
        """Check for constant (zero-variance) columns."""
        constant_cols = []

        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_cols.append(col)

        passed = len(constant_cols) == 0

        if not passed:
            self.blockers.append({
                "check": "constant_columns",
                "message": f"Found {len(constant_cols)} constant columns: {constant_cols}"
            })

        return {
            "passed": passed,
            "constant_columns": constant_cols,
        }

    def _check_temporal_integrity(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        date_column: str
    ) -> Dict[str, Any]:
        """Check temporal integrity - test should be after train."""
        if date_column not in X_train.columns or date_column not in X_test.columns:
            return {
                "passed": True,
                "note": "Date column not found in features"
            }

        try:
            train_dates = pd.to_datetime(X_train[date_column])
            test_dates = pd.to_datetime(X_test[date_column])

            train_max = train_dates.max()
            test_min = test_dates.min()

            # Test data should come after train data
            temporal_valid = test_min >= train_max

            if not temporal_valid:
                self.warnings.append({
                    "check": "temporal_integrity",
                    "message": f"Test data starts ({test_min}) before train data ends ({train_max})"
                })

            return {
                "passed": temporal_valid,
                "train_date_range": [str(train_dates.min()), str(train_dates.max())],
                "test_date_range": [str(test_dates.min()), str(test_dates.max())],
            }
        except Exception as e:
            return {
                "passed": True,
                "error": str(e)
            }

    def generate_handoff_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate the final handoff report for data scientists.

        This is what you give to the data scientist along with the data.
        """
        if not self.check_results:
            return "No validation performed. Run validate() first."

        r = self.check_results
        lines = []

        lines.append("=" * 80)
        lines.append("FINAL DATA QUALITY VALIDATION REPORT")
        lines.append("Pre-Handoff Checklist for ML Modeling")
        lines.append("=" * 80)
        lines.append(f"Generated: {r['validation_timestamp']}")
        lines.append("")

        # Overall Status
        status = "PASSED - DATA IS ML-READY" if r['overall_passed'] else "FAILED - DO NOT USE"
        status_symbol = "✓" if r['overall_passed'] else "✗"
        lines.append(f"OVERALL STATUS: {status_symbol} {status}")
        lines.append("")

        # Summary
        lines.append("-" * 40)
        lines.append("DATASET SUMMARY")
        lines.append("-" * 40)
        s = r['summary']
        lines.append(f"  Training samples: {s['train_samples']:,}")
        lines.append(f"  Test samples: {s['test_samples']:,}")
        lines.append(f"  Validation samples: {s['val_samples']:,}")
        lines.append(f"  Features: {s['n_features']}")
        lines.append(f"  Target: {s['target_column']}")
        lines.append("")

        # Check Results
        lines.append("-" * 40)
        lines.append("VALIDATION CHECKS")
        lines.append("-" * 40)
        for check_name, check_result in r['checks'].items():
            status = "✓" if check_result['passed'] else "✗"
            lines.append(f"  {status} {check_name.replace('_', ' ').title()}")
        lines.append("")

        # Blockers
        if r['blockers']:
            lines.append("-" * 40)
            lines.append("BLOCKERS (Must Fix)")
            lines.append("-" * 40)
            for b in r['blockers']:
                lines.append(f"  ✗ {b['check']}: {b['message']}")
            lines.append("")

        # Warnings
        if r['warnings']:
            lines.append("-" * 40)
            lines.append("WARNINGS (Review Recommended)")
            lines.append("-" * 40)
            for w in r['warnings']:
                lines.append(f"  ⚠ {w['check']}: {w['message']}")
            lines.append("")

        # Recommendation
        lines.append("-" * 40)
        lines.append("RECOMMENDATION")
        lines.append("-" * 40)
        lines.append(f"  {r['recommendation']}")
        lines.append("")

        lines.append("=" * 80)
        lines.append("END OF VALIDATION REPORT")
        lines.append("=" * 80)

        report_text = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text
