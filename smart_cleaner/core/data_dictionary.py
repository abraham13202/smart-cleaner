"""
Data Dictionary Generator module.
Creates comprehensive documentation for datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class DataDictionary:
    """
    Generates comprehensive data dictionaries for datasets.
    Documents all columns, transformations, and metadata.
    """

    # Common health column descriptions
    HEALTH_COLUMN_DESCRIPTIONS = {
        'age': 'Patient age in years',
        'bmi': 'Body Mass Index (weight in kg / height in m²)',
        'blood_pressure': 'Blood pressure measurement (mmHg)',
        'bp_sys': 'Systolic blood pressure (mmHg)',
        'bp_dia': 'Diastolic blood pressure (mmHg)',
        'systolic': 'Systolic blood pressure (mmHg)',
        'diastolic': 'Diastolic blood pressure (mmHg)',
        'heart_rate': 'Heart rate in beats per minute (bpm)',
        'hr': 'Heart rate in beats per minute (bpm)',
        'pulse': 'Pulse rate in beats per minute',
        'cholesterol': 'Total cholesterol level (mg/dL)',
        'hdl': 'HDL (good) cholesterol level (mg/dL)',
        'ldl': 'LDL (bad) cholesterol level (mg/dL)',
        'glucose': 'Blood glucose level (mg/dL)',
        'blood_sugar': 'Blood sugar level (mg/dL)',
        'temperature': 'Body temperature',
        'weight': 'Body weight',
        'height': 'Height measurement',
        'gender': 'Patient gender/sex',
        'sex': 'Patient sex',
        'smoker': 'Smoking status (Yes/No)',
        'smoking': 'Smoking status',
        'diabetes': 'Diabetes diagnosis status',
        'diabetic': 'Diabetic status',
        'stroke': 'History of stroke',
        'heart_disease': 'Heart disease status',
        'hypertension': 'High blood pressure status',
        'education': 'Education level',
        'income': 'Income level or amount',
        'physactivity': 'Physical activity level',
        'physical_activity': 'Physical activity level',
        'exercise': 'Exercise habits',
        'fruits': 'Fruit consumption',
        'veggies': 'Vegetable consumption',
        'vegetables': 'Vegetable consumption',
        'alcohol': 'Alcohol consumption',
        'alcoholic': 'Alcohol consumption status',
        'mental': 'Mental health days affected',
        'physical': 'Physical health days affected',
        'diffwalk': 'Difficulty walking',
        'healthcare': 'Healthcare access',
        'general_health': 'General health status',
        'generalhealth': 'General health status',
    }

    def __init__(self):
        """Initialize data dictionary generator."""
        self.dictionary = {}
        self.transformations = []

    def generate(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        custom_descriptions: Optional[Dict[str, str]] = None,
        transformations_log: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data dictionary.

        Args:
            df: DataFrame to document
            target_column: Target variable name
            custom_descriptions: Custom column descriptions
            transformations_log: List of transformations applied

        Returns:
            Dictionary with full documentation
        """
        custom_descriptions = custom_descriptions or {}

        self.dictionary = {
            "metadata": self._generate_metadata(df, target_column),
            "columns": self._document_columns(df, target_column, custom_descriptions),
            "transformations": transformations_log or [],
            "data_types_summary": self._summarize_data_types(df),
            "value_ranges": self._document_value_ranges(df),
            "categorical_values": self._document_categorical_values(df),
            "recommendations": self._generate_recommendations(df, target_column),
        }

        if transformations_log:
            self.transformations = transformations_log

        return self.dictionary

    def _generate_metadata(
        self,
        df: pd.DataFrame,
        target_column: Optional[str]
    ) -> Dict[str, Any]:
        """Generate dataset metadata."""
        return {
            "generated_at": datetime.now().isoformat(),
            "dataset_name": "Health Dataset",
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "target_column": target_column,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "completeness_pct": round(
                (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2
            ),
        }

    def _document_columns(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        custom_descriptions: Dict[str, str]
    ) -> Dict[str, Dict]:
        """Document each column in detail."""
        columns = {}

        for col in df.columns:
            col_data = df[col]
            col_doc = {
                "name": col,
                "dtype": str(col_data.dtype),
                "description": self._get_description(col, custom_descriptions),
                "is_target": col == target_column,
                "n_missing": int(col_data.isnull().sum()),
                "missing_pct": round(col_data.isnull().sum() / len(col_data) * 100, 2),
                "n_unique": int(col_data.nunique()),
            }

            # Numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                col_clean = col_data.dropna()
                if len(col_clean) > 0:
                    col_doc.update({
                        "category": "numeric",
                        "min": float(col_clean.min()),
                        "max": float(col_clean.max()),
                        "mean": round(float(col_clean.mean()), 4),
                        "median": float(col_clean.median()),
                        "std": round(float(col_clean.std()), 4),
                    })
            else:
                # Categorical columns
                value_counts = col_data.value_counts()
                col_doc.update({
                    "category": "categorical",
                    "unique_values": list(col_data.dropna().unique())[:20],
                    "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                })

            # Feature engineering info
            if '_encoded' in col or '_scaled' in col or '_log' in col:
                col_doc["is_derived"] = True
                col_doc["derived_from"] = col.replace('_encoded', '').replace('_scaled', '').replace('_log', '')

            # Health metric detection
            col_doc["is_health_metric"] = self._is_health_metric(col)

            columns[col] = col_doc

        return columns

    def _get_description(
        self,
        column: str,
        custom_descriptions: Dict[str, str]
    ) -> str:
        """Get column description from custom or default."""
        # Check custom first
        if column in custom_descriptions:
            return custom_descriptions[column]

        # Check health descriptions
        col_lower = column.lower().replace(' ', '_').replace('-', '_')
        for key, desc in self.HEALTH_COLUMN_DESCRIPTIONS.items():
            if key in col_lower:
                return desc

        # Default
        return f"Column: {column}"

    def _is_health_metric(self, column: str) -> bool:
        """Check if column is a health metric."""
        health_keywords = [
            'age', 'bmi', 'blood', 'pressure', 'heart', 'cholesterol',
            'glucose', 'sugar', 'temperature', 'weight', 'height',
            'pulse', 'rate', 'oxygen', 'respiratory'
        ]
        col_lower = column.lower()
        return any(kw in col_lower for kw in health_keywords)

    def _summarize_data_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Summarize columns by data type."""
        return {
            "numeric_continuous": [
                col for col in df.select_dtypes(include=[np.number]).columns
                if df[col].nunique() > 20
            ],
            "numeric_discrete": [
                col for col in df.select_dtypes(include=[np.number]).columns
                if df[col].nunique() <= 20
            ],
            "categorical": list(df.select_dtypes(include=['object', 'category']).columns),
            "binary": [
                col for col in df.columns
                if df[col].nunique() == 2
            ],
            "datetime": list(df.select_dtypes(include=['datetime64']).columns),
        }

    def _document_value_ranges(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Document value ranges for numeric columns."""
        ranges = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                ranges[col] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "range": float(col_data.max() - col_data.min()),
                    "q1": float(col_data.quantile(0.25)),
                    "q3": float(col_data.quantile(0.75)),
                    "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                }
        return ranges

    def _document_categorical_values(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Document categorical column values."""
        categorical = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            value_counts = df[col].value_counts()
            categorical[col] = {
                "unique_count": df[col].nunique(),
                "values": {
                    str(k): int(v)
                    for k, v in value_counts.head(20).items()
                },
                "has_more": df[col].nunique() > 20,
            }
        return categorical

    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        target_column: Optional[str]
    ) -> List[Dict[str, str]]:
        """Generate recommendations for data scientists."""
        recommendations = []

        # Missing data recommendations
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if missing_pct > 5:
            recommendations.append({
                "type": "missing_data",
                "severity": "high" if missing_pct > 20 else "medium",
                "message": f"Dataset has {missing_pct:.1f}% missing values. Consider imputation strategies.",
            })

        # Imbalanced target
        if target_column and target_column in df.columns:
            if df[target_column].nunique() <= 10:
                value_counts = df[target_column].value_counts()
                if len(value_counts) >= 2:
                    ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                    if ratio > 3:
                        recommendations.append({
                            "type": "class_imbalance",
                            "severity": "high" if ratio > 10 else "medium",
                            "message": f"Target variable is imbalanced (ratio: {ratio:.1f}:1). Consider SMOTE, class weights, or stratified sampling.",
                        })

        # High cardinality categoricals
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].nunique() > 50:
                recommendations.append({
                    "type": "high_cardinality",
                    "severity": "medium",
                    "message": f"Column '{col}' has {df[col].nunique()} unique values. Consider frequency or target encoding.",
                })

        # Highly correlated features
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            high_corr = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j and corr_matrix.loc[col1, col2] > 0.9:
                        high_corr.append((col1, col2, corr_matrix.loc[col1, col2]))

            if high_corr:
                recommendations.append({
                    "type": "multicollinearity",
                    "severity": "medium",
                    "message": f"Found {len(high_corr)} highly correlated feature pairs (r > 0.9). Consider removing redundant features.",
                })

        # Skewed features
        for col in numeric_df.columns:
            skewness = df[col].skew()
            if abs(skewness) > 2:
                recommendations.append({
                    "type": "skewed_distribution",
                    "severity": "low",
                    "message": f"Column '{col}' is highly skewed (skewness: {skewness:.2f}). Consider log transformation.",
                })

        return recommendations

    def to_markdown(self) -> str:
        """Export data dictionary as Markdown."""
        lines = []

        lines.append("# Data Dictionary")
        lines.append("")
        lines.append(f"Generated: {self.dictionary['metadata']['generated_at']}")
        lines.append("")

        # Overview
        lines.append("## Dataset Overview")
        lines.append("")
        meta = self.dictionary['metadata']
        lines.append(f"- **Rows**: {meta['n_rows']:,}")
        lines.append(f"- **Columns**: {meta['n_columns']}")
        lines.append(f"- **Target Column**: {meta['target_column'] or 'Not specified'}")
        lines.append(f"- **Completeness**: {meta['completeness_pct']}%")
        lines.append(f"- **Memory Usage**: {meta['memory_usage_mb']} MB")
        lines.append("")

        # Column Details
        lines.append("## Column Details")
        lines.append("")
        lines.append("| Column | Type | Description | Missing % | Unique |")
        lines.append("|--------|------|-------------|-----------|--------|")

        for col_name, col_info in self.dictionary['columns'].items():
            desc = col_info['description'][:50] + "..." if len(col_info['description']) > 50 else col_info['description']
            target_marker = " (TARGET)" if col_info.get('is_target') else ""
            lines.append(
                f"| {col_name}{target_marker} | {col_info['dtype']} | {desc} | "
                f"{col_info['missing_pct']}% | {col_info['n_unique']} |"
            )

        lines.append("")

        # Numeric Features
        lines.append("## Numeric Features")
        lines.append("")
        lines.append("| Column | Min | Max | Mean | Median | Std |")
        lines.append("|--------|-----|-----|------|--------|-----|")

        for col_name, col_info in self.dictionary['columns'].items():
            if col_info.get('category') == 'numeric':
                lines.append(
                    f"| {col_name} | {col_info.get('min', 'N/A')} | "
                    f"{col_info.get('max', 'N/A')} | {col_info.get('mean', 'N/A')} | "
                    f"{col_info.get('median', 'N/A')} | {col_info.get('std', 'N/A')} |"
                )

        lines.append("")

        # Categorical Features
        lines.append("## Categorical Features")
        lines.append("")

        for col_name, col_info in self.dictionary['columns'].items():
            if col_info.get('category') == 'categorical':
                lines.append(f"### {col_name}")
                lines.append(f"- **Unique values**: {col_info['n_unique']}")
                lines.append(f"- **Mode**: {col_info.get('mode', 'N/A')}")
                values = col_info.get('unique_values', [])[:10]
                lines.append(f"- **Sample values**: {', '.join(str(v) for v in values)}")
                lines.append("")

        # Recommendations
        if self.dictionary.get('recommendations'):
            lines.append("## Recommendations for Data Scientists")
            lines.append("")
            for rec in self.dictionary['recommendations']:
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    rec['severity'], "⚪"
                )
                lines.append(f"- {severity_emoji} **{rec['type'].replace('_', ' ').title()}**: {rec['message']}")
            lines.append("")

        # Transformations
        if self.dictionary.get('transformations'):
            lines.append("## Applied Transformations")
            lines.append("")
            for i, transform in enumerate(self.dictionary['transformations'], 1):
                lines.append(f"{i}. {transform}")
            lines.append("")

        return "\n".join(lines)

    def to_json(self, filepath: str) -> None:
        """Export data dictionary as JSON."""
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        with open(filepath, 'w') as f:
            json.dump(convert(self.dictionary), f, indent=2, default=str)

    def export(self, filepath: str, format: str = 'markdown') -> None:
        """Export data dictionary to file."""
        if format == 'markdown':
            content = self.to_markdown()
            with open(filepath, 'w') as f:
                f.write(content)
        elif format == 'json':
            self.to_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
