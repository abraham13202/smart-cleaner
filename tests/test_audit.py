"""Tests for audit trail."""

import pytest
import pandas as pd
import numpy as np
from smart_cleaner.core.audit import AuditTrail, AuditedDataFrame, OperationType


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25.0, np.nan, 35.0, np.nan, 28.0],
        'salary': [50000, 60000, 70000, 60000, 55000],
    })


def test_audit_trail_start(sample_df):
    """Test starting audit trail."""
    audit = AuditTrail()
    audit.start(sample_df)
    
    assert audit.started_at is not None
    assert audit.initial_hash is not None
    assert len(audit.checkpoints) == 1


def test_audit_trail_record(sample_df):
    """Test recording operations."""
    audit = AuditTrail()
    audit.start(sample_df)
    
    op_id = audit.record(
        operation_type=OperationType.IMPUTATION,
        column='age',
        description='Imputed with median',
        parameters={'strategy': 'median', 'value': 30.0},
        rows_affected=2,
    )
    
    assert op_id is not None
    assert len(audit.operations) == 1
    assert audit.operations[0].column == 'age'


def test_audit_trail_undo(sample_df):
    """Test undo functionality."""
    audit = AuditTrail()
    audit.start(sample_df)
    
    # Record an imputation
    df = sample_df.copy()
    before_values = {1: np.nan, 3: np.nan}
    
    audit.record(
        operation_type=OperationType.IMPUTATION,
        column='age',
        description='Imputed with median',
        parameters={'strategy': 'median'},
        rows_affected=2,
        before_values=before_values,
    )
    
    # Modify DataFrame
    df.loc[1, 'age'] = 30.0
    df.loc[3, 'age'] = 30.0
    
    # Undo
    restored_df, undone = audit.undo(df)
    
    assert len(undone) == 1
    assert pd.isna(restored_df.loc[1, 'age'])
    assert pd.isna(restored_df.loc[3, 'age'])


def test_audit_trail_checkpoint(sample_df):
    """Test checkpoint creation."""
    audit = AuditTrail()
    audit.start(sample_df)
    
    cp_id = audit.create_checkpoint(sample_df, 'after_cleaning', 'Data after initial cleaning')
    
    assert cp_id is not None
    assert len(audit.checkpoints) == 2  # Initial + new checkpoint


def test_audit_trail_history(sample_df):
    """Test getting operation history."""
    audit = AuditTrail()
    audit.start(sample_df)
    
    audit.record(
        operation_type=OperationType.IMPUTATION,
        column='age',
        description='Test operation',
    )
    
    history = audit.get_history()
    
    assert len(history) == 1
    assert history[0]['column'] == 'age'


def test_audited_dataframe_fillna(sample_df):
    """Test AuditedDataFrame fillna."""
    adf = AuditedDataFrame(sample_df)
    adf.fillna(30.0, column='age')
    
    assert len(adf.audit.operations) == 1
    assert adf.df['age'].isna().sum() == 0


def test_audited_dataframe_undo(sample_df):
    """Test AuditedDataFrame undo."""
    adf = AuditedDataFrame(sample_df)
    
    # Fill NA
    adf.fillna(30.0, column='age')
    assert adf.df['age'].isna().sum() == 0
    
    # Undo
    adf.undo()
    assert adf.df['age'].isna().sum() == 2
