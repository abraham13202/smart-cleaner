"""
AI Strategy Advisor - Uses Gemini to understand any dataset and create intelligent strategies.
This is the brain of the system that makes context-aware decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class AIStrategyAdvisor:
    """
    AI-powered strategic advisor that analyzes any dataset and provides
    intelligent recommendations for cleaning, imputation, and feature engineering.

    This replaces the need for a human data analyst to understand the data.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AI Strategy Advisor.

        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env variable)
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-pro")

        self.dataset_understanding = {}
        self.strategies = {}

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive AI-powered analysis of the dataset.

        This is the first step - understanding what the data is about.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with complete dataset understanding
        """
        # Gather dataset statistics
        stats = self._gather_statistics(df)

        # Use AI to understand the dataset
        understanding = self._ai_understand_dataset(stats)

        self.dataset_understanding = understanding
        return understanding

    def _gather_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gather comprehensive statistics about the dataset."""
        stats = {
            "shape": df.shape,
            "columns": {},
            "missing_summary": {},
            "correlations": [],
            "sample_data": {},
        }

        for col in df.columns:
            col_data = df[col]
            col_stats = {
                "dtype": str(col_data.dtype),
                "n_unique": int(col_data.nunique()),
                "n_missing": int(col_data.isnull().sum()),
                "missing_pct": round(col_data.isnull().sum() / len(df) * 100, 2),
            }

            if pd.api.types.is_numeric_dtype(col_data):
                col_clean = col_data.dropna()
                if len(col_clean) > 0:
                    col_stats.update({
                        "min": float(col_clean.min()),
                        "max": float(col_clean.max()),
                        "mean": round(float(col_clean.mean()), 2),
                        "median": float(col_clean.median()),
                        "std": round(float(col_clean.std()), 2),
                    })
            else:
                # Sample unique values for categorical - convert to native Python types
                sample_values = col_data.dropna().unique()[:10]
                sample_values_list = []
                for val in sample_values:
                    if pd.isna(val):
                        continue
                    if isinstance(val, (np.integer, np.floating)):
                        sample_values_list.append(float(val) if isinstance(val, np.floating) else int(val))
                    else:
                        sample_values_list.append(str(val))
                col_stats["sample_values"] = sample_values_list

            stats["columns"][col] = col_stats

            if col_stats["n_missing"] > 0:
                stats["missing_summary"][col] = col_stats["n_missing"]

        # Get correlations between numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            for i, col1 in enumerate(corr_matrix.columns):
                for col2 in list(corr_matrix.columns)[i+1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.3:
                        stats["correlations"].append({
                            "col1": col1,
                            "col2": col2,
                            "correlation": round(corr, 3)
                        })

        # Sample data (first 3 rows)
        stats["sample_data"] = df.head(3).to_dict()

        return stats

    def _ai_understand_dataset(self, stats: Dict) -> Dict[str, Any]:
        """Use AI to understand what the dataset is about."""
        prompt = f"""You are a senior data scientist. Analyze this dataset and provide a comprehensive understanding.

DATASET STATISTICS:
- Shape: {stats['shape'][0]} rows, {stats['shape'][1]} columns
- Columns: {list(stats['columns'].keys())}

COLUMN DETAILS:
{json.dumps(stats['columns'], indent=2)}

CORRELATIONS FOUND:
{json.dumps(stats['correlations'], indent=2)}

MISSING VALUES:
{json.dumps(stats['missing_summary'], indent=2)}

Provide your analysis in this exact JSON format:
{{
    "dataset_type": "healthcare|financial|retail|general|etc",
    "domain": "Brief description of what domain this data belongs to",
    "purpose": "What this dataset is likely used for",
    "target_column_guess": "Most likely target/outcome column name or null",
    "column_understanding": {{
        "column_name": {{
            "meaning": "What this column represents",
            "data_type": "continuous|categorical|binary|ordinal|identifier",
            "importance": "high|medium|low",
            "related_columns": ["list of related column names"],
            "domain_knowledge": "Any domain-specific knowledge about valid ranges or categories"
        }}
    }},
    "key_relationships": [
        {{
            "columns": ["col1", "col2"],
            "relationship": "Description of how these columns relate",
            "implication": "What this means for analysis"
        }}
    ],
    "data_quality_concerns": [
        "List of potential data quality issues to watch for"
    ],
    "recommended_target": "If no obvious target, suggest what could be predicted",
    "summary": "2-3 sentence summary of the dataset"
}}

Respond with ONLY valid JSON, no other text."""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Extract JSON
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
        except Exception as e:
            print(f"AI understanding failed: {e}")

        # Fallback
        return {
            "dataset_type": "unknown",
            "domain": "Unknown domain",
            "purpose": "Unknown purpose",
            "target_column_guess": None,
            "column_understanding": {},
            "key_relationships": [],
            "data_quality_concerns": [],
            "summary": "Unable to analyze dataset with AI"
        }

    def get_imputation_strategies(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Get AI-recommended imputation strategies for each column with missing values.

        This is the KEY feature - context-aware, cohort-based imputation.

        Example: For a 35-year-old female missing BMI, impute using mean BMI
        of females aged 30-40, not the overall mean.

        Args:
            df: DataFrame with missing values
            target_column: Target column (excluded from imputation)

        Returns:
            Dictionary mapping column names to imputation strategies
        """
        missing_cols = df.columns[df.isnull().any()].tolist()
        if target_column and target_column in missing_cols:
            missing_cols.remove(target_column)

        if not missing_cols:
            return {}

        # Build context for AI
        context = self._build_imputation_context(df, missing_cols)

        # Get AI recommendations
        strategies = self._ai_get_imputation_strategies(context)

        self.strategies["imputation"] = strategies
        return strategies

    def _build_imputation_context(
        self,
        df: pd.DataFrame,
        missing_cols: List[str]
    ) -> Dict:
        """Build context for imputation strategy recommendations."""
        context = {
            "columns_needing_imputation": {},
            "available_cohort_columns": [],
            "correlations": {},
        }

        # Identify potential cohort columns (categorical or can be binned)
        for col in df.columns:
            if col in missing_cols:
                continue

            n_unique = int(df[col].nunique())
            missing_pct = float(df[col].isnull().sum() / len(df))

            # Good cohort columns: low cardinality, few missing values
            if missing_pct < 0.1:
                if n_unique <= 10:  # Categorical
                    # Convert all values to native Python types
                    categories = df[col].dropna().unique()[:10]
                    categories_list = []
                    for cat in categories:
                        if pd.isna(cat):
                            continue
                        if isinstance(cat, (np.integer, np.floating)):
                            categories_list.append(float(cat) if isinstance(cat, np.floating) else int(cat))
                        else:
                            categories_list.append(str(cat))

                    context["available_cohort_columns"].append({
                        "column": col,
                        "type": "categorical",
                        "categories": categories_list
                    })
                elif pd.api.types.is_numeric_dtype(df[col]) and n_unique > 10:
                    context["available_cohort_columns"].append({
                        "column": col,
                        "type": "numeric_binnable",
                        "range": [float(df[col].min()), float(df[col].max())]
                    })

        # Info about columns needing imputation
        for col in missing_cols:
            col_data = df[col]
            info = {
                "dtype": str(col_data.dtype),
                "missing_count": int(col_data.isnull().sum()),
                "missing_pct": round(col_data.isnull().sum() / len(df) * 100, 2),
            }

            if pd.api.types.is_numeric_dtype(col_data):
                col_clean = col_data.dropna()
                info.update({
                    "mean": round(float(col_clean.mean()), 2),
                    "median": float(col_clean.median()),
                    "std": round(float(col_clean.std()), 2),
                    "skewness": round(float(col_clean.skew()), 2),
                })

                # Find correlations
                correlations = []
                for other_col in df.select_dtypes(include=[np.number]).columns:
                    if other_col != col:
                        corr = df[col].corr(df[other_col])
                        if abs(corr) > 0.3:
                            correlations.append({
                                "column": other_col,
                                "correlation": round(corr, 3)
                            })
                info["correlations"] = correlations
            else:
                info["type"] = "categorical"
                # Convert categorical values to native Python types
                categories = col_data.dropna().unique()[:10]
                categories_list = []
                for cat in categories:
                    if pd.isna(cat):
                        continue
                    if isinstance(cat, (np.integer, np.floating)):
                        categories_list.append(float(cat) if isinstance(cat, np.floating) else int(cat))
                    else:
                        categories_list.append(str(cat))
                info["categories"] = categories_list

            context["columns_needing_imputation"][col] = info

        return context

    def _ai_get_imputation_strategies(self, context: Dict) -> Dict[str, Dict]:
        """Use AI to recommend imputation strategies."""
        prompt = f"""You are a senior data scientist specializing in missing data imputation.

Your task: Recommend the BEST imputation strategy for each column with missing values.

KEY PRINCIPLE: Use COHORT-BASED imputation when possible.
Example: If BMI is missing for a 35-year-old female, don't use overall mean.
Instead, use mean BMI of females aged 30-40. This is much more accurate.

COLUMNS NEEDING IMPUTATION:
{json.dumps(context['columns_needing_imputation'], indent=2)}

AVAILABLE COHORT COLUMNS (can be used to group data):
{json.dumps(context['available_cohort_columns'], indent=2)}

For each column, provide a strategy in this JSON format:
{{
    "column_name": {{
        "strategy": "cohort_mean|cohort_median|mean|median|mode|knn|forward_fill|drop",
        "cohort_columns": ["list of columns to group by, if using cohort strategy"],
        "cohort_bins": {{
            "column_name": [0, 20, 40, 60, 80, 100]  // Only for numeric cohort columns
        }},
        "reasoning": "Detailed explanation of why this strategy",
        "expected_accuracy": "high|medium|low",
        "fallback_strategy": "What to do if cohort has no data",
        "technical_notes": "Any technical notes for implementation"
    }}
}}

IMPORTANT RULES:
1. ALWAYS prefer cohort_mean or cohort_median for numeric columns when good cohort columns exist
2. Use age, gender, or other demographic columns as cohorts when available
3. Consider correlations - if column A is highly correlated with B, use B as a cohort
4. For categorical columns, use mode within cohorts
5. Explain your reasoning clearly

Respond with ONLY valid JSON mapping column names to strategies."""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
        except Exception as e:
            print(f"AI imputation strategy failed: {e}")

        # Fallback to simple strategies
        strategies = {}
        for col, info in context["columns_needing_imputation"].items():
            if info.get("dtype", "").startswith(("int", "float")):
                strategies[col] = {
                    "strategy": "median",
                    "reasoning": "Fallback: using median for numeric column",
                    "expected_accuracy": "low"
                }
            else:
                strategies[col] = {
                    "strategy": "mode",
                    "reasoning": "Fallback: using mode for categorical column",
                    "expected_accuracy": "low"
                }

        return strategies

    def get_feature_engineering_recommendations(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> List[Dict]:
        """
        Get AI recommendations for feature engineering.

        Args:
            df: Input DataFrame
            target_column: Target variable

        Returns:
            List of feature engineering recommendations
        """
        # Build context
        columns_info = {}
        for col in df.columns:
            if col == target_column:
                continue

            col_data = df[col]
            info = {
                "dtype": str(col_data.dtype),
                "n_unique": int(col_data.nunique()),
            }

            if pd.api.types.is_numeric_dtype(col_data):
                col_clean = col_data.dropna()
                if len(col_clean) > 0:
                    info["range"] = [float(col_clean.min()), float(col_clean.max())]
            else:
                # Convert sample values to native Python types
                sample_values = col_data.dropna().unique()[:5]
                sample_values_list = []
                for val in sample_values:
                    if pd.isna(val):
                        continue
                    if isinstance(val, (np.integer, np.floating)):
                        sample_values_list.append(float(val) if isinstance(val, np.floating) else int(val))
                    else:
                        sample_values_list.append(str(val))
                info["sample_values"] = sample_values_list

            columns_info[col] = info

        prompt = f"""You are a senior data scientist. Recommend feature engineering for this dataset.

COLUMNS:
{json.dumps(columns_info, indent=2)}

TARGET COLUMN: {target_column}

DATASET UNDERSTANDING:
{json.dumps(self.dataset_understanding.get('column_understanding', {}), indent=2)}

Recommend feature engineering in this JSON format:
{{
    "recommendations": [
        {{
            "name": "feature_name",
            "type": "binning|interaction|ratio|aggregation|encoding|transformation",
            "source_columns": ["col1", "col2"],
            "description": "What this feature represents",
            "implementation": "How to create it (e.g., col1 * col2, or bin col1 into ranges)",
            "rationale": "Why this feature would be useful",
            "priority": "high|medium|low"
        }}
    ]
}}

Focus on:
1. Binning continuous variables into meaningful categories (e.g., age groups)
2. Interaction terms between related variables
3. Ratio features (e.g., BMI if height and weight exist)
4. Risk scores combining multiple factors
5. Domain-specific transformations

Respond with ONLY valid JSON."""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(response_text[start:end])
                return result.get("recommendations", [])
        except Exception as e:
            print(f"AI feature engineering failed: {e}")

        return []

    def get_analysis_plan(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get a complete analysis plan from AI.

        This is the strategic overview that guides the entire pipeline.

        Args:
            df: Input DataFrame
            target_column: Target variable

        Returns:
            Complete analysis plan
        """
        # First understand the dataset
        if not self.dataset_understanding:
            self.analyze_dataset(df)

        stats = self._gather_statistics(df)

        prompt = f"""You are a principal data scientist creating an analysis plan.

DATASET UNDERSTANDING:
{json.dumps(self.dataset_understanding, indent=2)}

STATISTICS:
- Shape: {stats['shape']}
- Missing values: {stats['missing_summary']}
- Correlations: {stats['correlations'][:10]}

TARGET: {target_column}

Create a COMPREHENSIVE analysis plan in this JSON format:
{{
    "executive_summary": "2-3 sentence summary for stakeholders",
    "data_quality_assessment": {{
        "overall_quality": "good|moderate|poor",
        "key_issues": ["list of main issues"],
        "recommendations": ["list of recommendations"]
    }},
    "preprocessing_plan": {{
        "step_1_duplicates": {{
            "action": "remove|keep",
            "reasoning": "why"
        }},
        "step_2_missing_values": {{
            "overall_strategy": "Description of approach",
            "columns_to_drop": ["columns with too much missing"],
            "columns_to_impute": ["columns to impute"]
        }},
        "step_3_outliers": {{
            "method": "iqr|zscore|isolation_forest",
            "strategy": "cap|remove|keep",
            "reasoning": "why"
        }},
        "step_4_validation": {{
            "checks_to_perform": ["list of validation checks"]
        }}
    }},
    "feature_engineering_plan": {{
        "binning": ["columns to bin and why"],
        "interactions": ["interaction features to create"],
        "transformations": ["transformations to apply"],
        "domain_features": ["domain-specific features"]
    }},
    "encoding_plan": {{
        "binary_encode": ["columns"],
        "onehot_encode": ["columns"],
        "ordinal_encode": ["columns with order"],
        "target_encode": ["high cardinality columns"]
    }},
    "scaling_plan": {{
        "standardize": ["columns - normally distributed"],
        "normalize": ["columns - bounded"],
        "log_transform": ["columns - skewed"]
    }},
    "feature_selection_plan": {{
        "remove_low_variance": true,
        "remove_high_correlation": true,
        "correlation_threshold": 0.95,
        "importance_method": "statistical|model_based"
    }},
    "modeling_recommendations": {{
        "task_type": "classification|regression",
        "recommended_models": ["list of models to try"],
        "evaluation_metrics": ["metrics to use"],
        "cross_validation": "stratified_kfold|kfold|time_series",
        "class_imbalance_handling": "none|smote|class_weight|undersampling"
    }},
    "risks_and_considerations": [
        "List of things to watch out for"
    ]
}}

Be specific and actionable. This plan will be executed automatically.
Respond with ONLY valid JSON."""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
        except Exception as e:
            print(f"AI analysis plan failed: {e}")

        # Return default plan
        return self._get_default_plan(df, target_column)

    def _get_default_plan(
        self,
        df: pd.DataFrame,
        target_column: Optional[str]
    ) -> Dict[str, Any]:
        """Get default analysis plan if AI fails."""
        return {
            "executive_summary": "Standard data preprocessing pipeline",
            "data_quality_assessment": {
                "overall_quality": "unknown",
                "key_issues": ["Unable to perform AI analysis"],
                "recommendations": ["Review data manually"]
            },
            "preprocessing_plan": {
                "step_1_duplicates": {"action": "remove", "reasoning": "Standard practice"},
                "step_2_missing_values": {"overall_strategy": "Use median for numeric, mode for categorical"},
                "step_3_outliers": {"method": "iqr", "strategy": "cap", "reasoning": "Preserve data points"},
            },
            "feature_engineering_plan": {"binning": [], "interactions": [], "transformations": []},
            "encoding_plan": {"binary_encode": [], "onehot_encode": [], "ordinal_encode": []},
            "scaling_plan": {"standardize": [], "normalize": [], "log_transform": []},
            "modeling_recommendations": {
                "task_type": "classification" if target_column and df[target_column].nunique() <= 10 else "regression",
                "recommended_models": ["Random Forest", "Gradient Boosting"],
            }
        }

    def explain_decision(self, decision_type: str, context: Dict) -> str:
        """
        Get AI explanation for a decision made during preprocessing.

        Used for documentation purposes.

        Args:
            decision_type: Type of decision (imputation, encoding, etc.)
            context: Context about the decision

        Returns:
            Human-readable explanation
        """
        prompt = f"""Explain this data preprocessing decision in clear, technical language.

DECISION TYPE: {decision_type}

CONTEXT:
{json.dumps(context, indent=2)}

Write a clear explanation (2-3 sentences) that a data scientist would understand.
Focus on the "why" not just the "what"."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return f"Decision: {decision_type} applied based on data characteristics."
