"""Tests for validation rules."""

import pytest
import pandas as pd
import numpy as np
from smart_cleaner.core.validators import (
    DataValidator,
    NotNullRule,
    UniqueRule,
    RangeRule,
    RegexRule,
    AllowedValuesRule,
    ValidationAction,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve'],
        'age': [25, 30, 35, 150, 28],  # 150 is invalid
        'email': ['alice@test.com', 'bob@test.com', 'invalid', 'david@test.com', 'eve@test.com'],
        'department': ['Engineering', 'HR', 'HR', 'Sales', 'Engineering'],
    })


def test_not_null_rule(sample_df):
    """Test not null validation."""
    rule = NotNullRule('name')
    result = rule.validate(sample_df)
    
    assert not result.passed
    assert result.failed_count == 1
    assert 2 in result.failed_indices


def test_unique_rule(sample_df):
    """Test unique validation."""
    rule = UniqueRule('department')
    result = rule.validate(sample_df)
    
    assert not result.passed
    assert result.failed_count == 2  # HR and Engineering are duplicated


def test_range_rule(sample_df):
    """Test range validation."""
    rule = RangeRule('age', min_val=0, max_val=120)
    result = rule.validate(sample_df)
    
    assert not result.passed
    assert result.failed_count == 1
    assert 3 in result.failed_indices  # Index 3 has age=150


def test_regex_rule(sample_df):
    """Test regex validation."""
    rule = RegexRule('email', r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
    result = rule.validate(sample_df)
    
    assert not result.passed
    assert result.failed_count == 1  # 'invalid' doesn't match


def test_allowed_values_rule(sample_df):
    """Test allowed values validation."""
    rule = AllowedValuesRule('department', ['Engineering', 'HR'])
    result = rule.validate(sample_df)
    
    assert not result.passed
    assert result.failed_count == 1  # 'Sales' is not allowed


def test_data_validator(sample_df):
    """Test DataValidator with multiple rules."""
    validator = DataValidator()
    validator.add_not_null('name')
    validator.add_range('age', min_val=0, max_val=120)
    validator.add_unique('id')
    
    report = validator.validate(sample_df)
    
    assert report.total_rules == 3
    assert report.failed_rules == 2  # name null and age range
    assert report.passed_rules == 1  # id is unique


def test_validator_apply_drop(sample_df):
    """Test validator with DROP action."""
    validator = DataValidator()
    validator.add_rule(RangeRule('age', min_val=0, max_val=120, action=ValidationAction.DROP))
    
    cleaned_df, report = validator.apply(sample_df)
    
    assert len(cleaned_df) == 4  # One row dropped
    assert 150 not in cleaned_df['age'].values
