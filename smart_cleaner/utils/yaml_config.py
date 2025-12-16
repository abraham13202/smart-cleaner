"""
YAML Configuration Support for Smart Cleaner.
Allows users to define cleaning pipelines in YAML format.
"""

import yaml
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field, asdict

if TYPE_CHECKING:
    from ..core.auto_pipeline import PipelineConfig


@dataclass
class ColumnConfig:
    """Configuration for a specific column."""
    name: str
    strategy: str = "auto"  # auto, mean, median, mode, cohort_mean, knn, drop
    cohort_columns: List[str] = field(default_factory=list)
    cohort_bins: Dict[str, List[float]] = field(default_factory=dict)
    fill_value: Optional[Any] = None
    dtype: Optional[str] = None  # Force column type
    rename_to: Optional[str] = None
    drop: bool = False


@dataclass
class ValidationRule:
    """Custom validation rule."""
    column: str
    rule_type: str  # range, regex, values, not_null, unique
    params: Dict[str, Any] = field(default_factory=dict)
    action: str = "warn"  # warn, drop, fill


@dataclass
class YAMLPipelineConfig:
    """
    Complete pipeline configuration from YAML.

    Example YAML:
    ```yaml
    pipeline:
      name: "Employee Data Cleaning"
      version: "1.0"

    ai:
      enabled: true
      provider: ollama
      model: llama3.2

    processing:
      handle_missing: true
      handle_duplicates: true
      detect_outliers: true
      target_column: salary

    columns:
      - name: age
        strategy: cohort_mean
        cohort_columns: [gender, department]

      - name: salary
        strategy: median

    validation:
      - column: age
        rule_type: range
        params:
          min: 18
          max: 100
        action: drop

    export:
      format: csv
      include_report: true
    ```
    """

    # Pipeline metadata
    name: str = "Data Cleaning Pipeline"
    version: str = "1.0"

    # AI settings
    ai_enabled: bool = True
    ai_provider: str = "ollama"
    ai_model: str = "llama3.2"

    # Processing options
    handle_missing: bool = True
    handle_duplicates: bool = True
    detect_outliers: bool = True
    target_column: Optional[str] = None

    # Column-specific configurations
    columns: List[ColumnConfig] = field(default_factory=list)

    # Validation rules
    validation_rules: List[ValidationRule] = field(default_factory=list)

    # Export settings
    export_format: str = "csv"
    include_report: bool = True
    output_dir: str = "output"

    @classmethod
    def from_yaml_file(cls, filepath: str) -> "YAMLPipelineConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> "YAMLPipelineConfig":
        """Load configuration from a YAML string."""
        data = yaml.safe_load(yaml_string)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "YAMLPipelineConfig":
        """Create config from dictionary."""
        config = cls()

        # Pipeline metadata
        pipeline = data.get('pipeline', {})
        config.name = pipeline.get('name', config.name)
        config.version = pipeline.get('version', config.version)

        # AI settings
        ai = data.get('ai', {})
        config.ai_enabled = ai.get('enabled', config.ai_enabled)
        config.ai_provider = ai.get('provider', config.ai_provider)
        config.ai_model = ai.get('model', config.ai_model)

        # Processing options
        processing = data.get('processing', {})
        config.handle_missing = processing.get('handle_missing', config.handle_missing)
        config.handle_duplicates = processing.get('handle_duplicates', config.handle_duplicates)
        config.detect_outliers = processing.get('detect_outliers', config.detect_outliers)
        config.target_column = processing.get('target_column', config.target_column)

        # Column configurations
        columns = data.get('columns', [])
        for col_data in columns:
            col_config = ColumnConfig(
                name=col_data.get('name', ''),
                strategy=col_data.get('strategy', 'auto'),
                cohort_columns=col_data.get('cohort_columns', []),
                cohort_bins=col_data.get('cohort_bins', {}),
                fill_value=col_data.get('fill_value'),
                dtype=col_data.get('dtype'),
                rename_to=col_data.get('rename_to'),
                drop=col_data.get('drop', False)
            )
            config.columns.append(col_config)

        # Validation rules
        validation = data.get('validation', [])
        for rule_data in validation:
            rule = ValidationRule(
                column=rule_data.get('column', ''),
                rule_type=rule_data.get('rule_type', 'not_null'),
                params=rule_data.get('params', {}),
                action=rule_data.get('action', 'warn')
            )
            config.validation_rules.append(rule)

        # Export settings
        export = data.get('export', {})
        config.export_format = export.get('format', config.export_format)
        config.include_report = export.get('include_report', config.include_report)
        config.output_dir = export.get('output_dir', config.output_dir)

        return config

    def to_pipeline_config(self):
        """Convert to PipelineConfig for use with AutoPreprocessor."""
        # Import here to avoid circular imports
        from ..core.auto_pipeline import PipelineConfig
        return PipelineConfig(
            use_ai_recommendations=self.ai_enabled,
            ai_provider=self.ai_provider,
            ollama_model=self.ai_model,
            target_column=self.target_column,
        )

    def to_yaml(self) -> str:
        """Export configuration as YAML string."""
        data = {
            'pipeline': {
                'name': self.name,
                'version': self.version,
            },
            'ai': {
                'enabled': self.ai_enabled,
                'provider': self.ai_provider,
                'model': self.ai_model,
            },
            'processing': {
                'handle_missing': self.handle_missing,
                'handle_duplicates': self.handle_duplicates,
                'detect_outliers': self.detect_outliers,
                'target_column': self.target_column,
            },
            'columns': [
                {
                    'name': col.name,
                    'strategy': col.strategy,
                    'cohort_columns': col.cohort_columns,
                    'cohort_bins': col.cohort_bins,
                    'fill_value': col.fill_value,
                    'dtype': col.dtype,
                    'rename_to': col.rename_to,
                    'drop': col.drop,
                }
                for col in self.columns
            ],
            'validation': [
                {
                    'column': rule.column,
                    'rule_type': rule.rule_type,
                    'params': rule.params,
                    'action': rule.action,
                }
                for rule in self.validation_rules
            ],
            'export': {
                'format': self.export_format,
                'include_report': self.include_report,
                'output_dir': self.output_dir,
            }
        }

        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def save_yaml(self, filepath: str) -> None:
        """Save configuration to a YAML file."""
        with open(filepath, 'w') as f:
            f.write(self.to_yaml())

    def get_column_config(self, column_name: str) -> Optional[ColumnConfig]:
        """Get configuration for a specific column."""
        for col in self.columns:
            if col.name == column_name:
                return col
        return None


def load_config(config_path: str) -> YAMLPipelineConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        YAMLPipelineConfig object
    """
    return YAMLPipelineConfig.from_yaml_file(config_path)


def create_sample_config(output_path: str = "sample_config.yaml") -> str:
    """
    Create a sample configuration file.

    Args:
        output_path: Where to save the sample config

    Returns:
        Path to the created file
    """
    sample = YAMLPipelineConfig(
        name="Sample Data Cleaning Pipeline",
        version="1.0",
        ai_enabled=True,
        ai_provider="ollama",
        ai_model="llama3.2",
        handle_missing=True,
        handle_duplicates=True,
        detect_outliers=True,
        target_column="salary",
        columns=[
            ColumnConfig(
                name="age",
                strategy="cohort_mean",
                cohort_columns=["gender", "department"],
                cohort_bins={"age": [0, 30, 50, 70, 100]}
            ),
            ColumnConfig(
                name="salary",
                strategy="median"
            ),
            ColumnConfig(
                name="department",
                strategy="mode"
            ),
        ],
        validation_rules=[
            ValidationRule(
                column="age",
                rule_type="range",
                params={"min": 18, "max": 100},
                action="warn"
            ),
            ValidationRule(
                column="salary",
                rule_type="range",
                params={"min": 0},
                action="warn"
            ),
        ],
        export_format="csv",
        include_report=True,
        output_dir="output"
    )

    sample.save_yaml(output_path)
    return output_path
