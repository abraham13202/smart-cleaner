"""
Unit tests for imputation strategies.
Run with: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
from smart_cleaner.core.imputation import (
    ImputationEngine,
    MeanImputation,
    MedianImputation,
    ModeImputation,
    CohortMeanImputation,
)


class TestBasicImputation:
    """Test basic imputation strategies."""

    @pytest.fixture
    def sample_numeric_data(self):
        """Create sample numeric DataFrame with missing values."""
        data = {
            'values': [1.0, 2.0, np.nan, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_categorical_data(self):
        """Create sample categorical DataFrame with missing values."""
        data = {
            'category': ['A', 'B', np.nan, 'A', 'B', np.nan, 'A', 'A', 'B', 'A']
        }
        return pd.DataFrame(data)

    def test_mean_imputation(self, sample_numeric_data):
        """Test mean imputation."""
        df = sample_numeric_data
        result = MeanImputation().impute(df, 'values')

        # Mean of non-missing values: (1+2+4+6+7+8+9+10)/8 = 5.875
        expected_mean = 5.875

        # Check no missing values
        assert result.isnull().sum() == 0

        # Check imputed values are mean
        assert result.iloc[2] == expected_mean
        assert result.iloc[4] == expected_mean

    def test_median_imputation(self, sample_numeric_data):
        """Test median imputation."""
        df = sample_numeric_data
        result = MedianImputation().impute(df, 'values')

        # Median of non-missing values: 6.5
        expected_median = 6.5

        assert result.isnull().sum() == 0
        assert result.iloc[2] == expected_median

    def test_mode_imputation(self, sample_categorical_data):
        """Test mode imputation."""
        df = sample_categorical_data
        result = ModeImputation().impute(df, 'category')

        # Mode is 'A' (appears 5 times)
        assert result.isnull().sum() == 0
        assert result.iloc[2] == 'A'
        assert result.iloc[5] == 'A'

    def test_imputation_engine(self, sample_numeric_data):
        """Test ImputationEngine."""
        df = sample_numeric_data

        # Test mean strategy
        result = ImputationEngine.impute(df, 'values', 'mean')
        assert result.isnull().sum() == 0

        # Test median strategy
        result = ImputationEngine.impute(df, 'values', 'median')
        assert result.isnull().sum() == 0

        # Test invalid strategy
        with pytest.raises(ValueError):
            ImputationEngine.impute(df, 'values', 'invalid_strategy')


class TestCohortImputation:
    """Test cohort-based imputation."""

    @pytest.fixture
    def age_bmi_data(self):
        """Create sample age-BMI dataset."""
        data = {
            'age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            'bmi': [22, np.nan, 25, np.nan, 27, 28, np.nan, 30, 31, 29]
        }
        return pd.DataFrame(data)

    def test_cohort_mean_imputation(self, age_bmi_data):
        """Test cohort-based mean imputation."""
        df = age_bmi_data

        result = CohortMeanImputation().impute(
            df,
            'bmi',
            cohort_column='age',
            cohort_bins=[0, 30, 50, 100]
        )

        # Check no missing values
        assert result.isnull().sum() == 0

        # Values should be imputed based on age cohort
        # Cohort 0-30: ages 20, 25, 30 -> BMI values 22, ?, 25 -> mean = 23.5
        # Cohort 30-50: ages 35, 40, 45, 50 -> BMI values ?, 27, 28, ? -> mean = 27.5
        # Cohort 50-100: ages 55, 60, 65 -> BMI values 30, 31, 29 -> mean = 30

        # Age 25 should get cohort 0-30 mean
        assert abs(result.iloc[1] - 23.5) < 0.1

        # Age 35 should get cohort 30-50 mean
        assert abs(result.iloc[3] - 27.5) < 0.1

    def test_cohort_without_bins(self):
        """Test cohort imputation with categorical cohorts."""
        data = {
            'gender': ['M', 'M', 'F', 'F', 'M', 'F'],
            'height': [175, np.nan, 162, 165, 180, np.nan]
        }
        df = pd.DataFrame(data)

        result = CohortMeanImputation().impute(
            df,
            'height',
            cohort_column='gender'
        )

        assert result.isnull().sum() == 0

        # Male mean: (175 + 180) / 2 = 177.5
        # Female mean: (162 + 165) / 2 = 163.5
        assert abs(result.iloc[1] - 177.5) < 0.1
        assert abs(result.iloc[5] - 163.5) < 0.1


class TestImputationReport:
    """Test imputation reporting."""

    def test_imputation_report(self):
        """Test generation of imputation report."""
        original = pd.Series([1, 2, np.nan, 4, np.nan, 6])
        imputed = pd.Series([1, 2, 3, 4, 5, 6])

        report = ImputationEngine.get_imputation_report(original, imputed)

        assert report['total_values'] == 6
        assert report['missing_before'] == 2
        assert report['missing_after'] == 0
        assert report['values_imputed'] == 2
        assert 'mean_before' in report
        assert 'mean_after' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
