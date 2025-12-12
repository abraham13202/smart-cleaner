"""
Audit Trail Module for Smart Cleaner.
Track all data transformations for reproducibility and undo functionality.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import copy
import hashlib


class OperationType(Enum):
    """Types of data operations."""
    IMPUTATION = "imputation"
    DROP_ROWS = "drop_rows"
    DROP_COLUMNS = "drop_columns"
    TYPE_CONVERSION = "type_conversion"
    RENAME = "rename"
    FILL_VALUE = "fill_value"
    OUTLIER_HANDLING = "outlier_handling"
    DUPLICATE_REMOVAL = "duplicate_removal"
    CUSTOM = "custom"


@dataclass
class Operation:
    """Record of a single data operation."""
    id: str
    timestamp: str
    operation_type: OperationType
    column: Optional[str]
    description: str
    parameters: Dict[str, Any]
    rows_affected: int = 0
    reversible: bool = True

    # For undo functionality
    before_values: Optional[Dict[int, Any]] = None  # {index: original_value}
    dropped_data: Optional[pd.DataFrame] = None


@dataclass
class AuditCheckpoint:
    """Snapshot of data at a point in time."""
    id: str
    timestamp: str
    name: str
    description: str
    data_hash: str
    row_count: int
    column_count: int
    operations_before: int  # Number of operations before this checkpoint


class AuditTrail:
    """
    Track all data transformations for reproducibility.

    Usage:
        audit = AuditTrail()
        audit.start(df)

        # After each operation
        audit.record(
            operation_type=OperationType.IMPUTATION,
            column="age",
            description="Imputed with median",
            parameters={"strategy": "median", "value": 35},
            before_values={5: np.nan, 12: np.nan}
        )

        # Undo last operation
        df = audit.undo(df)

        # Get history
        history = audit.get_history()
    """

    def __init__(self, max_undo_steps: int = 50):
        """
        Initialize audit trail.

        Args:
            max_undo_steps: Maximum number of operations to keep for undo
        """
        self.operations: List[Operation] = []
        self.checkpoints: List[AuditCheckpoint] = []
        self.max_undo_steps = max_undo_steps
        self.started_at: Optional[str] = None
        self.initial_hash: Optional[str] = None
        self._operation_counter = 0

    def start(self, df: pd.DataFrame, description: str = "Initial data"):
        """Start tracking a DataFrame."""
        self.started_at = datetime.now().isoformat()
        self.initial_hash = self._hash_dataframe(df)

        # Create initial checkpoint
        self.create_checkpoint(df, "initial", description)

    def record(
        self,
        operation_type: OperationType,
        column: str = None,
        description: str = "",
        parameters: Dict[str, Any] = None,
        rows_affected: int = 0,
        before_values: Dict[int, Any] = None,
        dropped_data: pd.DataFrame = None,
        reversible: bool = True,
    ) -> str:
        """
        Record an operation.

        Args:
            operation_type: Type of operation
            column: Column affected (if applicable)
            description: Human-readable description
            parameters: Operation parameters
            rows_affected: Number of rows affected
            before_values: Original values for undo
            dropped_data: Dropped DataFrame for undo
            reversible: Whether operation can be undone

        Returns:
            Operation ID
        """
        self._operation_counter += 1
        op_id = f"op_{self._operation_counter:04d}"

        operation = Operation(
            id=op_id,
            timestamp=datetime.now().isoformat(),
            operation_type=operation_type,
            column=column,
            description=description,
            parameters=parameters or {},
            rows_affected=rows_affected,
            reversible=reversible,
            before_values=before_values,
            dropped_data=dropped_data,
        )

        self.operations.append(operation)

        # Trim old operations if exceeding max
        if len(self.operations) > self.max_undo_steps:
            # Remove oldest non-checkpoint operations
            self.operations = self.operations[-self.max_undo_steps:]

        return op_id

    def record_imputation(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        column: str,
        strategy: str,
        parameters: Dict[str, Any] = None,
    ) -> str:
        """
        Record an imputation operation.

        Args:
            df_before: DataFrame before imputation
            df_after: DataFrame after imputation
            column: Column that was imputed
            strategy: Imputation strategy used
            parameters: Strategy parameters

        Returns:
            Operation ID
        """
        # Find changed values
        mask = df_before[column].isna() & df_after[column].notna()
        changed_indices = df_before[mask].index.tolist()

        before_values = {idx: np.nan for idx in changed_indices}

        return self.record(
            operation_type=OperationType.IMPUTATION,
            column=column,
            description=f"Imputed {column} using {strategy}",
            parameters={
                "strategy": strategy,
                **(parameters or {}),
            },
            rows_affected=len(changed_indices),
            before_values=before_values,
        )

    def record_drop_rows(
        self,
        df_before: pd.DataFrame,
        dropped_indices: List[int],
        reason: str = "Unknown",
    ) -> str:
        """Record row dropping operation."""
        dropped_data = df_before.loc[dropped_indices].copy()

        return self.record(
            operation_type=OperationType.DROP_ROWS,
            description=f"Dropped {len(dropped_indices)} rows: {reason}",
            parameters={"reason": reason, "indices": dropped_indices},
            rows_affected=len(dropped_indices),
            dropped_data=dropped_data,
        )

    def record_drop_columns(
        self,
        df_before: pd.DataFrame,
        columns: List[str],
        reason: str = "Unknown",
    ) -> str:
        """Record column dropping operation."""
        dropped_data = df_before[columns].copy()

        return self.record(
            operation_type=OperationType.DROP_COLUMNS,
            description=f"Dropped columns: {', '.join(columns)}",
            parameters={"columns": columns, "reason": reason},
            rows_affected=len(df_before),
            dropped_data=dropped_data,
        )

    def undo(self, df: pd.DataFrame, steps: int = 1) -> Tuple[pd.DataFrame, List[str]]:
        """
        Undo the last N operations.

        Args:
            df: Current DataFrame
            steps: Number of operations to undo

        Returns:
            Tuple of (restored DataFrame, list of undone operation IDs)
        """
        undone = []
        df = df.copy()

        for _ in range(steps):
            if not self.operations:
                break

            op = self.operations[-1]

            if not op.reversible:
                print(f"Warning: Operation {op.id} is not reversible")
                break

            # Perform undo based on operation type
            if op.operation_type == OperationType.IMPUTATION:
                if op.before_values and op.column:
                    for idx, val in op.before_values.items():
                        if idx in df.index:
                            df.loc[idx, op.column] = val

            elif op.operation_type == OperationType.DROP_ROWS:
                if op.dropped_data is not None:
                    df = pd.concat([df, op.dropped_data]).sort_index()

            elif op.operation_type == OperationType.DROP_COLUMNS:
                if op.dropped_data is not None:
                    for col in op.dropped_data.columns:
                        df[col] = op.dropped_data[col]

            elif op.operation_type == OperationType.FILL_VALUE:
                if op.before_values and op.column:
                    for idx, val in op.before_values.items():
                        if idx in df.index:
                            df.loc[idx, op.column] = val

            undone.append(op.id)
            self.operations.pop()

        return df, undone

    def create_checkpoint(
        self,
        df: pd.DataFrame,
        name: str,
        description: str = "",
    ) -> str:
        """
        Create a named checkpoint.

        Args:
            df: Current DataFrame state
            name: Checkpoint name
            description: Description

        Returns:
            Checkpoint ID
        """
        checkpoint = AuditCheckpoint(
            id=f"cp_{len(self.checkpoints):03d}",
            timestamp=datetime.now().isoformat(),
            name=name,
            description=description,
            data_hash=self._hash_dataframe(df),
            row_count=len(df),
            column_count=len(df.columns),
            operations_before=len(self.operations),
        )

        self.checkpoints.append(checkpoint)
        return checkpoint.id

    def get_history(self) -> List[Dict[str, Any]]:
        """Get operation history as list of dictionaries."""
        return [
            {
                "id": op.id,
                "timestamp": op.timestamp,
                "type": op.operation_type.value,
                "column": op.column,
                "description": op.description,
                "rows_affected": op.rows_affected,
                "reversible": op.reversible,
            }
            for op in self.operations
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get audit trail summary."""
        ops_by_type = {}
        total_rows_affected = 0

        for op in self.operations:
            op_type = op.operation_type.value
            ops_by_type[op_type] = ops_by_type.get(op_type, 0) + 1
            total_rows_affected += op.rows_affected

        return {
            "started_at": self.started_at,
            "total_operations": len(self.operations),
            "total_checkpoints": len(self.checkpoints),
            "total_rows_affected": total_rows_affected,
            "operations_by_type": ops_by_type,
            "reversible_operations": sum(1 for op in self.operations if op.reversible),
        }

    def export_log(self, filepath: str):
        """Export audit log to JSON file."""
        log = {
            "started_at": self.started_at,
            "initial_hash": self.initial_hash,
            "exported_at": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "checkpoints": [
                {
                    "id": cp.id,
                    "timestamp": cp.timestamp,
                    "name": cp.name,
                    "description": cp.description,
                    "data_hash": cp.data_hash,
                    "row_count": cp.row_count,
                    "column_count": cp.column_count,
                }
                for cp in self.checkpoints
            ],
            "operations": self.get_history(),
        }

        with open(filepath, 'w') as f:
            json.dump(log, f, indent=2, default=str)

    def print_history(self, last_n: int = None):
        """Print operation history."""
        print("\n" + "=" * 60)
        print("AUDIT TRAIL")
        print("=" * 60)
        print(f"Started: {self.started_at}")
        print(f"Total operations: {len(self.operations)}")
        print(f"Checkpoints: {len(self.checkpoints)}")

        ops_to_show = self.operations[-last_n:] if last_n else self.operations

        print("\nOperations:")
        for op in ops_to_show:
            reversible = "↩️" if op.reversible else "🔒"
            print(f"  {reversible} [{op.id}] {op.operation_type.value}")
            print(f"     {op.description}")
            if op.rows_affected > 0:
                print(f"     Rows affected: {op.rows_affected}")

        print("\nCheckpoints:")
        for cp in self.checkpoints:
            print(f"  📍 [{cp.id}] {cp.name}")
            print(f"     {cp.description}")
            print(f"     Rows: {cp.row_count}, Columns: {cp.column_count}")

        print("=" * 60)

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        """Generate hash for DataFrame state."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()


class AuditedDataFrame:
    """
    Wrapper that automatically tracks all operations.

    Usage:
        adf = AuditedDataFrame(df)
        adf.fillna({"age": adf.df["age"].median()})
        adf.drop_duplicates()

        # Undo
        adf.undo()

        # Get clean df
        clean_df = adf.df
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.audit = AuditTrail()
        self.audit.start(self.df)

    def fillna(self, value: Any, column: str = None) -> "AuditedDataFrame":
        """Fill NA values with tracking."""
        df_before = self.df.copy()

        if column:
            mask = self.df[column].isna()
            before_values = {idx: np.nan for idx in self.df[mask].index}
            self.df[column] = self.df[column].fillna(value)

            self.audit.record(
                operation_type=OperationType.FILL_VALUE,
                column=column,
                description=f"Filled NA in {column} with {value}",
                parameters={"value": value},
                rows_affected=len(before_values),
                before_values=before_values,
            )
        else:
            self.df = self.df.fillna(value)
            self.audit.record(
                operation_type=OperationType.FILL_VALUE,
                description=f"Filled all NA with {value}",
                parameters={"value": value},
                rows_affected=df_before.isna().sum().sum(),
                reversible=False,  # Complex to reverse without column info
            )

        return self

    def dropna(self, subset: List[str] = None) -> "AuditedDataFrame":
        """Drop NA rows with tracking."""
        mask = self.df.isna().any(axis=1) if subset is None else self.df[subset].isna().any(axis=1)
        dropped_indices = self.df[mask].index.tolist()

        if dropped_indices:
            self.audit.record_drop_rows(self.df, dropped_indices, "NA values")
            self.df = self.df.dropna(subset=subset)

        return self

    def drop_duplicates(self) -> "AuditedDataFrame":
        """Remove duplicates with tracking."""
        mask = self.df.duplicated(keep='first')
        dropped_indices = self.df[mask].index.tolist()

        if dropped_indices:
            self.audit.record_drop_rows(self.df, dropped_indices, "Duplicate rows")
            self.df = self.df.drop_duplicates()

        return self

    def drop_columns(self, columns: List[str]) -> "AuditedDataFrame":
        """Drop columns with tracking."""
        existing = [c for c in columns if c in self.df.columns]
        if existing:
            self.audit.record_drop_columns(self.df, existing, "User requested")
            self.df = self.df.drop(columns=existing)

        return self

    def undo(self, steps: int = 1) -> "AuditedDataFrame":
        """Undo last operations."""
        self.df, undone = self.audit.undo(self.df, steps)
        if undone:
            print(f"Undone: {', '.join(undone)}")
        return self

    def checkpoint(self, name: str, description: str = "") -> "AuditedDataFrame":
        """Create a checkpoint."""
        self.audit.create_checkpoint(self.df, name, description)
        return self

    def history(self, last_n: int = None):
        """Print operation history."""
        self.audit.print_history(last_n)
