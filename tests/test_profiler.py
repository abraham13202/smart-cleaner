"""Tests for data profiler."""

import pytest
import pandas as pd
import numpy as np
from smart_cleaner.core.profiler import DataProfiler, DataProfile


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(100),
        'name': [f'Person_{i}' for i in range(100)],
        'age': np.random.randint(20, 60, 100),
        'salary': np.random.normal(75000, 15000, 100),
        'department': np.random.choice(['Engineering', 'HR', 'Sales', 'Marketing'], 100),
        'missing_col': [None if i % 10 == 0 else i for i in range(100)],
    })


def test_profiler_basic(sample_df):
    """Test basic profiling."""
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    
    assert profile.n_rows == 100
    assert profile.n_columns == 6
    assert profile.completeness_score > 0


def test_profiler_missing_detection(sample_df):
    """Test missing value detection."""
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    
    assert profile.total_missing > 0
    assert 'missing_col' in profile.columns
    assert profile.columns['missing_col'].missing > 0


def test_profiler_type_inference(sample_df):
    """Test type inference."""
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    
    assert profile.columns['id'].is_potential_id
    assert profile.columns['age'].inferred_type == 'numeric'
    assert profile.columns['department'].inferred_type == 'categorical'


def test_profiler_numeric_stats(sample_df):
    """Test numeric statistics."""
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    
    age_profile = profile.columns['age']
    assert age_profile.mean is not None
    assert age_profile.std is not None
    assert age_profile.min_val is not None
    assert age_profile.max_val is not None
    assert age_profile.median is not None


def test_profiler_categorical_stats(sample_df):
    """Test categorical statistics."""
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    
    dept_profile = profile.columns['department']
    assert dept_profile.top_values is not None
    assert len(dept_profile.top_values) <= 10
    assert dept_profile.mode is not None


def test_profiler_to_dict(sample_df):
    """Test profile to dictionary conversion."""
    profiler = DataProfiler()
    profile = profiler.profile(sample_df)
    
    profile_dict = profiler.to_dict(profile)
    
    assert 'metadata' in profile_dict
    assert 'overview' in profile_dict
    assert 'columns' in profile_dict
    assert profile_dict['overview']['n_rows'] == 100


def test_profiler_warnings(sample_df):
    """Test warning generation."""
    # Create a DataFrame with issues
    df = sample_df.copy()
    df['constant'] = 1  # Constant column
    df.loc[0:50, 'high_missing'] = None  # High missing column
    
    profiler = DataProfiler()
    profile = profiler.profile(df)
    
    assert len(profile.warnings) > 0
