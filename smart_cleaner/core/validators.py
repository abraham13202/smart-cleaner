"""
Custom Validation Rules for Smart Cleaner.
Define and apply data quality validation rules.
"""

import re
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class ValidationAction(Enum):
    """Actions to take when validation fails."""
    WARN = "warn"       # Log warning, keep data
    DROP = "drop"       # Drop invalid rows
    FILL = "fill"       # Fill with specified value
    FLAG = "flag"       # Add a flag column
    RAISE = "raise"     # Raise exception


class ValidationSeverity(Enum):
    """Severity levels for validation failures."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    column: str
    rule_name: str
    passed: bool
    failed_count: int = 0
    failed_indices: List[int] = field(default_factory=list)
    failed_values: List[Any] = field(default_factory=list)
    message: str = ""
    severity: ValidationSeverity = ValidationSeverity.WARNING


@dataclass
class ValidationReport:
    """Complete validation report for a DataFrame."""
    total_rules: int = 0
    passed_rules: int = 0
    failed_rules: int = 0
    total_violations: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    is_valid: bool = True


class ValidationRule(ABC):
    """Base class for validation rules."""

    def __init__(
        self,
        column: str,
        action: ValidationAction = ValidationAction.WARN,
        fill_value: Any = None,
        severity: ValidationSeverity = ValidationSeverity.WARNING,
        name: str = None,
    ):
        self.column = column
        self.action = action
        self.fill_value = fill_value
        self.severity = severity
        self.name = name or self.__class__.__name__

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Check if data passes validation."""
        pass

    def apply(self, df: pd.DataFrame, result: ValidationResult) -> pd.DataFrame:
        """Apply the action for failed validations."""
        if result.passed or not result.failed_indices:
            return df

        df = df.copy()

        if self.action == ValidationAction.DROP:
            df = df.drop(index=result.failed_indices)
        elif self.action == ValidationAction.FILL:
            if self.fill_value is not None:
                df.loc[result.failed_indices, self.column] = self.fill_value
        elif self.action == ValidationAction.FLAG:
            flag_col = f"{self.column}_invalid"
            df[flag_col] = False
            df.loc[result.failed_indices, flag_col] = True
        elif self.action == ValidationAction.RAISE:
            raise ValueError(result.message)

        return df


class NotNullRule(ValidationRule):
    """Validate that values are not null."""

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        mask = df[self.column].isnull()
        failed_idx = df[mask].index.tolist()

        return ValidationResult(
            column=self.column,
            rule_name=self.name,
            passed=len(failed_idx) == 0,
            failed_count=len(failed_idx),
            failed_indices=failed_idx,
            message=f"Column '{self.column}' has {len(failed_idx)} null values",
            severity=self.severity,
        )


class UniqueRule(ValidationRule):
    """Validate that values are unique."""

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        duplicates = df[self.column].duplicated(keep='first')
        failed_idx = df[duplicates].index.tolist()
        failed_vals = df.loc[failed_idx, self.column].tolist()[:10]

        return ValidationResult(
            column=self.column,
            rule_name=self.name,
            passed=len(failed_idx) == 0,
            failed_count=len(failed_idx),
            failed_indices=failed_idx,
            failed_values=failed_vals,
            message=f"Column '{self.column}' has {len(failed_idx)} duplicate values",
            severity=self.severity,
        )


class RangeRule(ValidationRule):
    """Validate that numeric values are within a range."""

    def __init__(
        self,
        column: str,
        min_val: float = None,
        max_val: float = None,
        inclusive: bool = True,
        **kwargs
    ):
        super().__init__(column, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.inclusive = inclusive

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        col = df[self.column]

        if self.inclusive:
            if self.min_val is not None and self.max_val is not None:
                mask = ~((col >= self.min_val) & (col <= self.max_val))
            elif self.min_val is not None:
                mask = col < self.min_val
            elif self.max_val is not None:
                mask = col > self.max_val
            else:
                mask = pd.Series(False, index=df.index)
        else:
            if self.min_val is not None and self.max_val is not None:
                mask = ~((col > self.min_val) & (col < self.max_val))
            elif self.min_val is not None:
                mask = col <= self.min_val
            elif self.max_val is not None:
                mask = col >= self.max_val
            else:
                mask = pd.Series(False, index=df.index)

        # Exclude nulls from validation failures
        mask = mask & col.notna()
        failed_idx = df[mask].index.tolist()
        failed_vals = df.loc[failed_idx, self.column].tolist()[:10]

        range_str = f"[{self.min_val}, {self.max_val}]"
        return ValidationResult(
            column=self.column,
            rule_name=self.name,
            passed=len(failed_idx) == 0,
            failed_count=len(failed_idx),
            failed_indices=failed_idx,
            failed_values=failed_vals,
            message=f"Column '{self.column}' has {len(failed_idx)} values outside range {range_str}",
            severity=self.severity,
        )


class RegexRule(ValidationRule):
    """Validate that string values match a regex pattern."""

    def __init__(self, column: str, pattern: str, **kwargs):
        super().__init__(column, **kwargs)
        self.pattern = pattern
        self.compiled = re.compile(pattern)

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        col = df[self.column].astype(str)
        mask = ~col.str.match(self.pattern, na=False)

        # Exclude nulls
        mask = mask & df[self.column].notna()
        failed_idx = df[mask].index.tolist()
        failed_vals = df.loc[failed_idx, self.column].tolist()[:10]

        return ValidationResult(
            column=self.column,
            rule_name=self.name,
            passed=len(failed_idx) == 0,
            failed_count=len(failed_idx),
            failed_indices=failed_idx,
            failed_values=failed_vals,
            message=f"Column '{self.column}' has {len(failed_idx)} values not matching pattern '{self.pattern}'",
            severity=self.severity,
        )


class AllowedValuesRule(ValidationRule):
    """Validate that values are from an allowed set."""

    def __init__(self, column: str, allowed_values: List[Any], **kwargs):
        super().__init__(column, **kwargs)
        self.allowed_values = set(allowed_values)

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        mask = ~df[self.column].isin(self.allowed_values)

        # Exclude nulls
        mask = mask & df[self.column].notna()
        failed_idx = df[mask].index.tolist()
        failed_vals = df.loc[failed_idx, self.column].unique().tolist()[:10]

        return ValidationResult(
            column=self.column,
            rule_name=self.name,
            passed=len(failed_idx) == 0,
            failed_count=len(failed_idx),
            failed_indices=failed_idx,
            failed_values=failed_vals,
            message=f"Column '{self.column}' has {len(failed_idx)} values not in allowed set",
            severity=self.severity,
        )


class TypeRule(ValidationRule):
    """Validate that values can be converted to a specific type."""

    def __init__(self, column: str, dtype: str, **kwargs):
        super().__init__(column, **kwargs)
        self.dtype = dtype

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        failed_idx = []

        for idx, val in df[self.column].items():
            if pd.isna(val):
                continue

            try:
                if self.dtype == 'int':
                    int(val)
                elif self.dtype == 'float':
                    float(val)
                elif self.dtype == 'datetime':
                    pd.to_datetime(val)
                elif self.dtype == 'bool':
                    if str(val).lower() not in ['true', 'false', '1', '0', 'yes', 'no']:
                        raise ValueError()
            except (ValueError, TypeError):
                failed_idx.append(idx)

        failed_vals = df.loc[failed_idx[:10], self.column].tolist() if failed_idx else []

        return ValidationResult(
            column=self.column,
            rule_name=self.name,
            passed=len(failed_idx) == 0,
            failed_count=len(failed_idx),
            failed_indices=failed_idx,
            failed_values=failed_vals,
            message=f"Column '{self.column}' has {len(failed_idx)} values that cannot be converted to {self.dtype}",
            severity=self.severity,
        )


class LengthRule(ValidationRule):
    """Validate string length."""

    def __init__(
        self,
        column: str,
        min_length: int = None,
        max_length: int = None,
        **kwargs
    ):
        super().__init__(column, **kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        lengths = df[self.column].astype(str).str.len()

        if self.min_length is not None and self.max_length is not None:
            mask = ~((lengths >= self.min_length) & (lengths <= self.max_length))
        elif self.min_length is not None:
            mask = lengths < self.min_length
        elif self.max_length is not None:
            mask = lengths > self.max_length
        else:
            mask = pd.Series(False, index=df.index)

        # Exclude nulls
        mask = mask & df[self.column].notna()
        failed_idx = df[mask].index.tolist()

        return ValidationResult(
            column=self.column,
            rule_name=self.name,
            passed=len(failed_idx) == 0,
            failed_count=len(failed_idx),
            failed_indices=failed_idx,
            message=f"Column '{self.column}' has {len(failed_idx)} values with invalid length",
            severity=self.severity,
        )


class CustomRule(ValidationRule):
    """Custom validation using a user-defined function."""

    def __init__(
        self,
        column: str,
        func: Callable[[Any], bool],
        error_message: str = None,
        **kwargs
    ):
        super().__init__(column, **kwargs)
        self.func = func
        self.error_message = error_message

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        failed_idx = []

        for idx, val in df[self.column].items():
            if pd.isna(val):
                continue
            try:
                if not self.func(val):
                    failed_idx.append(idx)
            except Exception:
                failed_idx.append(idx)

        failed_vals = df.loc[failed_idx[:10], self.column].tolist() if failed_idx else []

        message = self.error_message or f"Column '{self.column}' has {len(failed_idx)} values failing custom validation"

        return ValidationResult(
            column=self.column,
            rule_name=self.name,
            passed=len(failed_idx) == 0,
            failed_count=len(failed_idx),
            failed_indices=failed_idx,
            failed_values=failed_vals,
            message=message,
            severity=self.severity,
        )


class DataValidator:
    """
    Apply multiple validation rules to a DataFrame.

    Usage:
        validator = DataValidator()
        validator.add_rule(NotNullRule("id"))
        validator.add_rule(RangeRule("age", min_val=0, max_val=150))
        validator.add_rule(RegexRule("email", r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"))

        report = validator.validate(df)
        cleaned_df = validator.apply(df)
    """

    def __init__(self):
        self.rules: List[ValidationRule] = []

    def add_rule(self, rule: ValidationRule) -> "DataValidator":
        """Add a validation rule."""
        self.rules.append(rule)
        return self

    def add_not_null(self, column: str, **kwargs) -> "DataValidator":
        """Add not-null validation."""
        return self.add_rule(NotNullRule(column, **kwargs))

    def add_unique(self, column: str, **kwargs) -> "DataValidator":
        """Add uniqueness validation."""
        return self.add_rule(UniqueRule(column, **kwargs))

    def add_range(
        self,
        column: str,
        min_val: float = None,
        max_val: float = None,
        **kwargs
    ) -> "DataValidator":
        """Add range validation."""
        return self.add_rule(RangeRule(column, min_val, max_val, **kwargs))

    def add_regex(self, column: str, pattern: str, **kwargs) -> "DataValidator":
        """Add regex validation."""
        return self.add_rule(RegexRule(column, pattern, **kwargs))

    def add_allowed_values(
        self,
        column: str,
        values: List[Any],
        **kwargs
    ) -> "DataValidator":
        """Add allowed values validation."""
        return self.add_rule(AllowedValuesRule(column, values, **kwargs))

    def add_type(self, column: str, dtype: str, **kwargs) -> "DataValidator":
        """Add type validation."""
        return self.add_rule(TypeRule(column, dtype, **kwargs))

    def add_length(
        self,
        column: str,
        min_length: int = None,
        max_length: int = None,
        **kwargs
    ) -> "DataValidator":
        """Add length validation."""
        return self.add_rule(LengthRule(column, min_length, max_length, **kwargs))

    def add_custom(
        self,
        column: str,
        func: Callable[[Any], bool],
        **kwargs
    ) -> "DataValidator":
        """Add custom validation."""
        return self.add_rule(CustomRule(column, func, **kwargs))

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate DataFrame against all rules.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationReport with results
        """
        report = ValidationReport(total_rules=len(self.rules))

        for rule in self.rules:
            if rule.column not in df.columns:
                result = ValidationResult(
                    column=rule.column,
                    rule_name=rule.name,
                    passed=False,
                    message=f"Column '{rule.column}' not found in DataFrame",
                    severity=ValidationSeverity.ERROR,
                )
            else:
                result = rule.validate(df)

            report.results.append(result)

            if result.passed:
                report.passed_rules += 1
            else:
                report.failed_rules += 1
                report.total_violations += result.failed_count
                if result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                    report.is_valid = False

        return report

    def apply(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        """
        Validate and apply actions to DataFrame.

        Args:
            df: DataFrame to validate and clean

        Returns:
            Tuple of (cleaned DataFrame, ValidationReport)
        """
        report = ValidationReport(total_rules=len(self.rules))
        df = df.copy()

        for rule in self.rules:
            if rule.column not in df.columns:
                result = ValidationResult(
                    column=rule.column,
                    rule_name=rule.name,
                    passed=False,
                    message=f"Column '{rule.column}' not found in DataFrame",
                    severity=ValidationSeverity.ERROR,
                )
            else:
                result = rule.validate(df)
                if not result.passed:
                    df = rule.apply(df, result)

            report.results.append(result)

            if result.passed:
                report.passed_rules += 1
            else:
                report.failed_rules += 1
                report.total_violations += result.failed_count

        return df, report

    def print_report(self, report: ValidationReport):
        """Print validation report."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        status = "✓ VALID" if report.is_valid else "✗ INVALID"
        print(f"\nStatus: {status}")
        print(f"Rules: {report.passed_rules}/{report.total_rules} passed")
        print(f"Total violations: {report.total_violations}")

        if report.failed_rules > 0:
            print(f"\n⚠️ Failed Rules:")
            for result in report.results:
                if not result.passed:
                    icon = "❌" if result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] else "⚠️"
                    print(f"  {icon} {result.rule_name} ({result.column})")
                    print(f"     {result.message}")
                    if result.failed_values:
                        vals_str = ", ".join(str(v) for v in result.failed_values[:5])
                        if len(result.failed_values) > 5:
                            vals_str += f" ... and {len(result.failed_values) - 5} more"
                        print(f"     Sample values: {vals_str}")

        print("\n" + "=" * 60)
