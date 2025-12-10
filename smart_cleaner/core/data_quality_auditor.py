"""
Comprehensive Data Quality Auditor.
Performs complete data quality assessment following industry standards.

Evaluates:
- Completeness: Missing values, sparse columns
- Accuracy: Invalid values, impossible data
- Consistency: Format mismatches, value standardization
- Uniqueness: Duplicates, ID uniqueness
- Timeliness: Outdated data, timestamp validity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re


class DataQualityAuditor:
    """
    Comprehensive data quality auditor that produces a complete
    Data Quality Report covering all dimensions of data quality.
    """

    def __init__(self):
        """Initialize the auditor."""
        self.audit_results = {}
        self.issues = []
        self.recommendations = []

    def full_audit(
        self,
        df: pd.DataFrame,
        id_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
        business_rules: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data quality audit.

        Args:
            df: DataFrame to audit
            id_columns: Columns that should be unique identifiers
            date_columns: Columns containing dates/timestamps
            business_rules: Custom validation rules per column

        Returns:
            Complete audit report
        """
        self.issues = []
        self.recommendations = []

        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "dataset_overview": self._dataset_overview(df),
            "completeness": self._assess_completeness(df),
            "accuracy": self._assess_accuracy(df, business_rules),
            "consistency": self._assess_consistency(df),
            "uniqueness": self._assess_uniqueness(df, id_columns),
            "timeliness": self._assess_timeliness(df, date_columns),
            "overall_quality_score": 0,
            "quality_grade": "",
            "issues_summary": [],
            "recommendations": [],
            "detailed_column_audit": self._audit_columns(df),
        }

        # Calculate overall score
        scores = [
            report["completeness"]["score"],
            report["accuracy"]["score"],
            report["consistency"]["score"],
            report["uniqueness"]["score"],
            report["timeliness"]["score"],
        ]
        report["overall_quality_score"] = round(sum(scores) / len(scores), 1)
        report["quality_grade"] = self._get_grade(report["overall_quality_score"])
        report["issues_summary"] = self.issues
        report["recommendations"] = self.recommendations

        self.audit_results = report
        return report

    def _dataset_overview(self, df: pd.DataFrame) -> Dict:
        """Get basic dataset overview."""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns),
            "total_cells": df.shape[0] * df.shape[1],
            "total_missing": int(df.isnull().sum().sum()),
        }

    def _assess_completeness(self, df: pd.DataFrame) -> Dict:
        """
        Assess data completeness.

        Checks:
        - Missing value counts and percentages
        - Sparse columns (>50% missing)
        - Columns to consider dropping (>90% missing)
        """
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        completeness_pct = (1 - total_missing / total_cells) * 100 if total_cells > 0 else 100

        # Column-level analysis
        missing_by_column = {}
        sparse_columns = []
        columns_to_drop = []

        for col in df.columns:
            missing = df[col].isnull().sum()
            missing_pct = missing / len(df) * 100

            missing_by_column[col] = {
                "missing_count": int(missing),
                "missing_pct": round(missing_pct, 2),
                "status": "OK" if missing_pct < 5 else "WARNING" if missing_pct < 50 else "CRITICAL"
            }

            if missing_pct >= 50:
                sparse_columns.append(col)
                self.issues.append({
                    "type": "completeness",
                    "severity": "high" if missing_pct >= 90 else "medium",
                    "column": col,
                    "message": f"Column '{col}' is {missing_pct:.1f}% missing"
                })

            if missing_pct >= 90:
                columns_to_drop.append(col)
                self.recommendations.append({
                    "type": "drop_column",
                    "column": col,
                    "reason": f"{missing_pct:.1f}% missing - consider dropping"
                })

        # Row-level analysis
        rows_with_missing = df.isnull().any(axis=1).sum()
        complete_rows = len(df) - rows_with_missing

        result = {
            "score": round(completeness_pct, 1),
            "total_missing_values": int(total_missing),
            "completeness_percentage": round(completeness_pct, 2),
            "missing_by_column": missing_by_column,
            "sparse_columns": sparse_columns,
            "columns_to_consider_dropping": columns_to_drop,
            "rows_with_any_missing": int(rows_with_missing),
            "complete_rows": int(complete_rows),
            "complete_rows_pct": round(complete_rows / len(df) * 100, 2) if len(df) > 0 else 100,
        }

        return result

    def _assess_accuracy(
        self,
        df: pd.DataFrame,
        business_rules: Optional[Dict] = None
    ) -> Dict:
        """
        Assess data accuracy.

        Checks:
        - Impossible values (negative ages, invalid dates)
        - Out-of-range values
        - Business rule violations
        """
        issues_found = []
        accuracy_score = 100

        # Auto-detect and validate common patterns
        for col in df.columns:
            col_lower = col.lower()

            # Age validation
            if 'age' in col_lower:
                invalid = df[(df[col] < 0) | (df[col] > 120)][col]
                if len(invalid) > 0:
                    issues_found.append({
                        "column": col,
                        "issue": "invalid_age",
                        "count": len(invalid),
                        "examples": invalid.head(5).tolist(),
                        "message": f"Found {len(invalid)} ages outside 0-120 range"
                    })
                    accuracy_score -= min(10, len(invalid) / len(df) * 100)
                    self.issues.append({
                        "type": "accuracy",
                        "severity": "high",
                        "column": col,
                        "message": f"Invalid ages found: {len(invalid)} values outside 0-120"
                    })

            # Price/Amount validation (shouldn't be negative usually)
            if any(x in col_lower for x in ['price', 'amount', 'cost', 'revenue', 'income', 'salary']):
                if pd.api.types.is_numeric_dtype(df[col]):
                    negative = df[df[col] < 0][col]
                    if len(negative) > 0:
                        issues_found.append({
                            "column": col,
                            "issue": "negative_value",
                            "count": len(negative),
                            "examples": negative.head(5).tolist(),
                            "message": f"Found {len(negative)} negative values (may indicate refunds)"
                        })
                        self.issues.append({
                            "type": "accuracy",
                            "severity": "medium",
                            "column": col,
                            "message": f"Negative values in {col}: {len(negative)} records (verify if valid)"
                        })

            # Percentage validation (should be 0-100 or 0-1)
            if any(x in col_lower for x in ['percent', 'pct', 'rate', 'ratio']):
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].max() <= 1:
                        invalid = df[(df[col] < 0) | (df[col] > 1)][col]
                    else:
                        invalid = df[(df[col] < 0) | (df[col] > 100)][col]
                    if len(invalid) > 0:
                        issues_found.append({
                            "column": col,
                            "issue": "invalid_percentage",
                            "count": len(invalid),
                            "message": f"Found {len(invalid)} invalid percentage values"
                        })

            # BMI validation (typical range 10-60)
            if 'bmi' in col_lower:
                if pd.api.types.is_numeric_dtype(df[col]):
                    invalid = df[(df[col] < 10) | (df[col] > 60)][col].dropna()
                    if len(invalid) > 0:
                        issues_found.append({
                            "column": col,
                            "issue": "suspicious_bmi",
                            "count": len(invalid),
                            "message": f"Found {len(invalid)} BMI values outside typical range (10-60)"
                        })

            # Blood pressure validation
            if any(x in col_lower for x in ['systolic', 'diastolic', 'bp']):
                if pd.api.types.is_numeric_dtype(df[col]):
                    if 'systolic' in col_lower:
                        invalid = df[(df[col] < 60) | (df[col] > 250)][col].dropna()
                    elif 'diastolic' in col_lower:
                        invalid = df[(df[col] < 30) | (df[col] > 150)][col].dropna()
                    else:
                        invalid = df[(df[col] < 30) | (df[col] > 250)][col].dropna()
                    if len(invalid) > 0:
                        issues_found.append({
                            "column": col,
                            "issue": "suspicious_blood_pressure",
                            "count": len(invalid),
                            "message": f"Found {len(invalid)} suspicious blood pressure values"
                        })

            # Heart rate validation (typical 30-220)
            if any(x in col_lower for x in ['heart_rate', 'heartrate', 'pulse', 'hr']):
                if pd.api.types.is_numeric_dtype(df[col]):
                    invalid = df[(df[col] < 30) | (df[col] > 220)][col].dropna()
                    if len(invalid) > 0:
                        issues_found.append({
                            "column": col,
                            "issue": "suspicious_heart_rate",
                            "count": len(invalid),
                            "message": f"Found {len(invalid)} suspicious heart rate values"
                        })

        # Apply custom business rules if provided
        if business_rules:
            for col, rules in business_rules.items():
                if col in df.columns:
                    if 'min' in rules:
                        violations = df[df[col] < rules['min']]
                        if len(violations) > 0:
                            issues_found.append({
                                "column": col,
                                "issue": f"below_minimum_{rules['min']}",
                                "count": len(violations),
                                "message": f"{len(violations)} values below minimum {rules['min']}"
                            })
                    if 'max' in rules:
                        violations = df[df[col] > rules['max']]
                        if len(violations) > 0:
                            issues_found.append({
                                "column": col,
                                "issue": f"above_maximum_{rules['max']}",
                                "count": len(violations),
                                "message": f"{len(violations)} values above maximum {rules['max']}"
                            })

        return {
            "score": max(0, round(accuracy_score, 1)),
            "issues_found": issues_found,
            "total_issues": len(issues_found),
            "columns_with_issues": list(set(i["column"] for i in issues_found)),
        }

    def _assess_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Assess data consistency.

        Checks:
        - Format inconsistencies (Yes/No vs Y/N vs 1/0)
        - Case inconsistencies
        - Whitespace issues
        - Mixed data types
        """
        issues_found = []
        consistency_score = 100

        for col in df.columns:
            if df[col].dtype == 'object':
                non_null = df[col].dropna()
                if len(non_null) == 0:
                    continue

                # Check for whitespace issues
                has_leading_space = non_null.str.startswith(' ').any()
                has_trailing_space = non_null.str.endswith(' ').any()
                if has_leading_space or has_trailing_space:
                    issues_found.append({
                        "column": col,
                        "issue": "whitespace",
                        "message": "Contains leading/trailing whitespace"
                    })
                    consistency_score -= 2

                # Check for case inconsistencies in categorical-like columns
                unique_values = non_null.unique()
                if len(unique_values) <= 20:
                    lower_unique = set(v.lower().strip() for v in unique_values if isinstance(v, str))
                    if len(lower_unique) < len(unique_values):
                        issues_found.append({
                            "column": col,
                            "issue": "case_inconsistency",
                            "message": f"Case inconsistencies found (e.g., 'Yes' vs 'yes')",
                            "unique_values": list(unique_values)[:10]
                        })
                        consistency_score -= 5
                        self.recommendations.append({
                            "type": "standardize_case",
                            "column": col,
                            "reason": "Standardize case for consistency"
                        })

                # Check for Yes/No vs Y/N vs 1/0 patterns
                unique_lower = [str(v).lower().strip() for v in unique_values]
                yes_no_variants = {'yes', 'no', 'y', 'n', '1', '0', 'true', 'false', 't', 'f'}
                if set(unique_lower).issubset(yes_no_variants) and len(set(unique_lower) & yes_no_variants) > 2:
                    issues_found.append({
                        "column": col,
                        "issue": "mixed_boolean_format",
                        "message": "Mixed boolean formats detected",
                        "formats_found": list(unique_values)
                    })
                    consistency_score -= 5
                    self.recommendations.append({
                        "type": "standardize_boolean",
                        "column": col,
                        "reason": "Standardize to consistent format (e.g., 1/0 or True/False)"
                    })

                # Check for mixed numeric/string in same column
                numeric_pattern = re.compile(r'^-?\d+\.?\d*$')
                numeric_count = sum(1 for v in non_null if numeric_pattern.match(str(v)))
                if 0 < numeric_count < len(non_null):
                    issues_found.append({
                        "column": col,
                        "issue": "mixed_types",
                        "message": f"Mixed numeric and text values ({numeric_count} numeric, {len(non_null) - numeric_count} text)"
                    })
                    consistency_score -= 10
                    self.issues.append({
                        "type": "consistency",
                        "severity": "high",
                        "column": col,
                        "message": "Mixed data types in column"
                    })

        return {
            "score": max(0, round(consistency_score, 1)),
            "issues_found": issues_found,
            "total_issues": len(issues_found),
        }

    def _assess_uniqueness(
        self,
        df: pd.DataFrame,
        id_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Assess data uniqueness.

        Checks:
        - Exact duplicate rows
        - Duplicate IDs
        - Near-duplicates
        """
        # Exact duplicates
        exact_duplicates = df.duplicated().sum()
        duplicate_rows = df[df.duplicated(keep=False)]

        uniqueness_score = 100 - (exact_duplicates / len(df) * 100) if len(df) > 0 else 100

        if exact_duplicates > 0:
            self.issues.append({
                "type": "uniqueness",
                "severity": "high",
                "column": "ALL",
                "message": f"Found {exact_duplicates} exact duplicate rows"
            })
            self.recommendations.append({
                "type": "remove_duplicates",
                "column": "ALL",
                "reason": f"Remove {exact_duplicates} duplicate rows"
            })

        # ID column uniqueness
        id_uniqueness = {}
        if id_columns:
            for col in id_columns:
                if col in df.columns:
                    duplicates = df[col].duplicated().sum()
                    id_uniqueness[col] = {
                        "is_unique": duplicates == 0,
                        "duplicate_count": int(duplicates),
                        "unique_values": int(df[col].nunique()),
                        "total_values": int(df[col].count())
                    }
                    if duplicates > 0:
                        self.issues.append({
                            "type": "uniqueness",
                            "severity": "critical",
                            "column": col,
                            "message": f"ID column '{col}' has {duplicates} duplicates"
                        })

        # Auto-detect potential ID columns
        potential_id_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['id', 'key', 'code', 'number', 'no']):
                if df[col].nunique() == len(df):
                    potential_id_cols.append(col)

        return {
            "score": max(0, round(uniqueness_score, 1)),
            "exact_duplicate_rows": int(exact_duplicates),
            "duplicate_percentage": round(exact_duplicates / len(df) * 100, 2) if len(df) > 0 else 0,
            "id_column_uniqueness": id_uniqueness,
            "potential_id_columns": potential_id_cols,
            "unique_row_count": int(len(df) - exact_duplicates),
        }

    def _assess_timeliness(
        self,
        df: pd.DataFrame,
        date_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Assess data timeliness.

        Checks:
        - Outdated records
        - Future dates
        - Invalid timestamps
        """
        timeliness_score = 100
        issues_found = []

        # Auto-detect date columns if not provided
        if date_columns is None:
            date_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(x in col_lower for x in ['date', 'time', 'timestamp', 'created', 'updated', 'modified']):
                    date_columns.append(col)

        current_time = datetime.now()

        for col in date_columns:
            if col not in df.columns:
                continue

            # Try to parse as datetime
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                valid_dates = dates.dropna()

                if len(valid_dates) == 0:
                    continue

                # Check for future dates
                future_dates = valid_dates[valid_dates > current_time]
                if len(future_dates) > 0:
                    issues_found.append({
                        "column": col,
                        "issue": "future_dates",
                        "count": len(future_dates),
                        "message": f"Found {len(future_dates)} future dates"
                    })
                    timeliness_score -= 5
                    self.issues.append({
                        "type": "timeliness",
                        "severity": "high",
                        "column": col,
                        "message": f"{len(future_dates)} future dates detected - possible data leakage"
                    })

                # Check for very old dates (before 1900)
                very_old = valid_dates[valid_dates < pd.Timestamp('1900-01-01')]
                if len(very_old) > 0:
                    issues_found.append({
                        "column": col,
                        "issue": "very_old_dates",
                        "count": len(very_old),
                        "message": f"Found {len(very_old)} dates before 1900"
                    })

                # Check date range
                date_range = {
                    "min": str(valid_dates.min()),
                    "max": str(valid_dates.max()),
                    "span_days": (valid_dates.max() - valid_dates.min()).days
                }
                issues_found.append({
                    "column": col,
                    "issue": "date_range_info",
                    "date_range": date_range,
                    "message": f"Date range: {date_range['min']} to {date_range['max']}"
                })

                # Check for parsing failures
                parse_failures = df[col].notna().sum() - len(valid_dates)
                if parse_failures > 0:
                    issues_found.append({
                        "column": col,
                        "issue": "unparseable_dates",
                        "count": parse_failures,
                        "message": f"{parse_failures} dates could not be parsed"
                    })

            except Exception:
                pass

        return {
            "score": max(0, round(timeliness_score, 1)),
            "date_columns_analyzed": date_columns,
            "issues_found": issues_found,
        }

    def _audit_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detailed audit for each column."""
        column_audit = {}

        for col in df.columns:
            audit = {
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isnull().sum()),
                "missing_pct": round(df[col].isnull().sum() / len(df) * 100, 2),
                "unique_count": int(df[col].nunique()),
                "unique_pct": round(df[col].nunique() / len(df) * 100, 2),
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                col_clean = df[col].dropna()
                if len(col_clean) > 0:
                    audit.update({
                        "min": float(col_clean.min()),
                        "max": float(col_clean.max()),
                        "mean": round(float(col_clean.mean()), 4),
                        "median": float(col_clean.median()),
                        "std": round(float(col_clean.std()), 4),
                        "zeros_count": int((col_clean == 0).sum()),
                        "negative_count": int((col_clean < 0).sum()),
                    })
            else:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    value_counts = non_null.value_counts()
                    audit.update({
                        "top_values": value_counts.head(5).to_dict(),
                        "most_common": str(value_counts.index[0]),
                        "most_common_count": int(value_counts.iloc[0]),
                    })

            column_audit[col] = audit

        return column_audit

    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive data quality report.

        Args:
            output_path: If provided, save report to this file

        Returns:
            Report as string
        """
        if not self.audit_results:
            return "No audit results. Run full_audit() first."

        r = self.audit_results
        lines = []

        lines.append("=" * 80)
        lines.append("DATA QUALITY AUDIT REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {r['audit_timestamp']}")
        lines.append("")

        # Overall Score
        lines.append(f"OVERALL QUALITY SCORE: {r['overall_quality_score']}% (Grade: {r['quality_grade']})")
        lines.append("")

        # Dataset Overview
        lines.append("-" * 40)
        lines.append("DATASET OVERVIEW")
        lines.append("-" * 40)
        overview = r['dataset_overview']
        lines.append(f"  Rows: {overview['rows']:,}")
        lines.append(f"  Columns: {overview['columns']}")
        lines.append(f"  Memory: {overview['memory_mb']} MB")
        lines.append(f"  Total Cells: {overview['total_cells']:,}")
        lines.append(f"  Total Missing: {overview['total_missing']:,}")
        lines.append("")

        # Dimension Scores
        lines.append("-" * 40)
        lines.append("QUALITY DIMENSIONS")
        lines.append("-" * 40)
        lines.append(f"  Completeness: {r['completeness']['score']}%")
        lines.append(f"  Accuracy: {r['accuracy']['score']}%")
        lines.append(f"  Consistency: {r['consistency']['score']}%")
        lines.append(f"  Uniqueness: {r['uniqueness']['score']}%")
        lines.append(f"  Timeliness: {r['timeliness']['score']}%")
        lines.append("")

        # Critical Issues
        critical_issues = [i for i in r['issues_summary'] if i.get('severity') in ['critical', 'high']]
        if critical_issues:
            lines.append("-" * 40)
            lines.append("CRITICAL ISSUES")
            lines.append("-" * 40)
            for issue in critical_issues:
                lines.append(f"  [{issue['severity'].upper()}] {issue['column']}: {issue['message']}")
            lines.append("")

        # Recommendations
        if r['recommendations']:
            lines.append("-" * 40)
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 40)
            for rec in r['recommendations'][:10]:
                lines.append(f"  - {rec['column']}: {rec['reason']}")
            lines.append("")

        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        report_text = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text

    def get_cleaning_priorities(self) -> List[Dict]:
        """Get prioritized list of cleaning actions."""
        priorities = []

        # High priority: Critical issues
        for issue in self.issues:
            if issue.get('severity') == 'critical':
                priorities.append({
                    "priority": 1,
                    "action": f"Fix {issue['type']} issue in {issue['column']}",
                    "reason": issue['message']
                })
            elif issue.get('severity') == 'high':
                priorities.append({
                    "priority": 2,
                    "action": f"Address {issue['type']} issue in {issue['column']}",
                    "reason": issue['message']
                })

        # Medium priority: Recommendations
        for rec in self.recommendations:
            priorities.append({
                "priority": 3,
                "action": f"{rec['type']} for {rec['column']}",
                "reason": rec['reason']
            })

        return sorted(priorities, key=lambda x: x['priority'])
