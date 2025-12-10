"""
Feature Engineering module for health data.
Creates new features from existing ones to improve ML model performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class FeatureEngineer:
    """
    Automated feature engineering for health datasets.
    Creates meaningful features based on domain knowledge.
    """

    # Health-specific age groups
    AGE_BINS = [0, 18, 30, 45, 60, 75, 100, 150]
    AGE_LABELS = ['Child', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior', 'Elderly']

    # BMI categories (WHO standard)
    BMI_BINS = [0, 18.5, 25, 30, 35, 40, 100]
    BMI_LABELS = ['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II', 'Obese_III']

    # Blood pressure categories
    BP_SYSTOLIC_BINS = [0, 90, 120, 130, 140, 180, 300]
    BP_SYSTOLIC_LABELS = ['Low', 'Normal', 'Elevated', 'High_Stage1', 'High_Stage2', 'Crisis']

    # Cholesterol categories
    CHOLESTEROL_BINS = [0, 200, 240, 500]
    CHOLESTEROL_LABELS = ['Desirable', 'Borderline_High', 'High']

    # Glucose categories
    GLUCOSE_BINS = [0, 70, 100, 126, 200, 700]
    GLUCOSE_LABELS = ['Low', 'Normal', 'Prediabetes', 'Diabetes', 'Very_High']

    @classmethod
    def auto_engineer_features(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Automatically engineer features based on detected columns.

        Args:
            df: Input DataFrame
            target_column: Target variable name (excluded from engineering)

        Returns:
            Tuple of (DataFrame with new features, engineering report)
        """
        df_result = df.copy()
        report = {
            "features_created": [],
            "binned_features": [],
            "interaction_features": [],
            "ratio_features": [],
            "risk_scores": [],
            "aggregated_features": [],
        }

        # 1. Create health category bins
        df_result, bin_report = cls._create_health_bins(df_result)
        report["binned_features"] = bin_report

        # 2. Create interaction features
        df_result, interaction_report = cls._create_interactions(df_result, target_column)
        report["interaction_features"] = interaction_report

        # 3. Create ratio features
        df_result, ratio_report = cls._create_ratios(df_result)
        report["ratio_features"] = ratio_report

        # 4. Create risk scores
        df_result, risk_report = cls._create_risk_scores(df_result)
        report["risk_scores"] = risk_report

        # 5. Create aggregated features
        df_result, agg_report = cls._create_aggregations(df_result)
        report["aggregated_features"] = agg_report

        # Compile all created features
        report["features_created"] = (
            report["binned_features"] +
            report["interaction_features"] +
            report["ratio_features"] +
            report["risk_scores"] +
            report["aggregated_features"]
        )

        report["total_new_features"] = len(report["features_created"])

        return df_result, report

    @classmethod
    def _create_health_bins(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create categorical bins for health metrics."""
        created = []

        # Age binning
        age_cols = cls._find_columns(df, ['age'])
        for col in age_cols:
            new_col = f"{col}_group"
            try:
                df[new_col] = pd.cut(
                    df[col],
                    bins=cls.AGE_BINS,
                    labels=cls.AGE_LABELS,
                    include_lowest=True
                )
                created.append(new_col)
            except Exception:
                pass

        # BMI binning
        bmi_cols = cls._find_columns(df, ['bmi', 'body_mass_index'])
        for col in bmi_cols:
            new_col = f"{col}_category"
            try:
                df[new_col] = pd.cut(
                    df[col],
                    bins=cls.BMI_BINS,
                    labels=cls.BMI_LABELS,
                    include_lowest=True
                )
                created.append(new_col)
            except Exception:
                pass

        # Blood pressure binning
        bp_cols = cls._find_columns(df, ['blood_pressure', 'bp_sys', 'systolic', 'bloodpressure'])
        for col in bp_cols:
            new_col = f"{col}_category"
            try:
                df[new_col] = pd.cut(
                    df[col],
                    bins=cls.BP_SYSTOLIC_BINS,
                    labels=cls.BP_SYSTOLIC_LABELS,
                    include_lowest=True
                )
                created.append(new_col)
            except Exception:
                pass

        # Cholesterol binning
        chol_cols = cls._find_columns(df, ['cholesterol', 'chol', 'total_cholesterol'])
        for col in chol_cols:
            new_col = f"{col}_category"
            try:
                df[new_col] = pd.cut(
                    df[col],
                    bins=cls.CHOLESTEROL_BINS,
                    labels=cls.CHOLESTEROL_LABELS,
                    include_lowest=True
                )
                created.append(new_col)
            except Exception:
                pass

        # Glucose binning
        glucose_cols = cls._find_columns(df, ['glucose', 'blood_sugar', 'fasting_glucose'])
        for col in glucose_cols:
            new_col = f"{col}_category"
            try:
                df[new_col] = pd.cut(
                    df[col],
                    bins=cls.GLUCOSE_BINS,
                    labels=cls.GLUCOSE_LABELS,
                    include_lowest=True
                )
                created.append(new_col)
            except Exception:
                pass

        return df, created

    @classmethod
    def _create_interactions(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Create interaction features between related health metrics."""
        created = []

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)

        # Age-BMI interaction (important for health outcomes)
        age_cols = cls._find_columns(df, ['age'])
        bmi_cols = cls._find_columns(df, ['bmi'])
        for age_col in age_cols:
            for bmi_col in bmi_cols:
                new_col = f"{age_col}_x_{bmi_col}"
                try:
                    df[new_col] = df[age_col] * df[bmi_col]
                    created.append(new_col)
                except Exception:
                    pass

        # Blood pressure x Heart rate (cardiovascular stress indicator)
        bp_cols = cls._find_columns(df, ['blood_pressure', 'bp_sys', 'systolic'])
        hr_cols = cls._find_columns(df, ['heart_rate', 'hr', 'pulse'])
        for bp_col in bp_cols:
            for hr_col in hr_cols:
                new_col = f"{bp_col}_x_{hr_col}"
                try:
                    df[new_col] = df[bp_col] * df[hr_col]
                    created.append(new_col)
                except Exception:
                    pass

        # Glucose x BMI (diabetes risk indicator)
        glucose_cols = cls._find_columns(df, ['glucose', 'blood_sugar'])
        for glucose_col in glucose_cols:
            for bmi_col in bmi_cols:
                new_col = f"{glucose_col}_x_{bmi_col}"
                try:
                    df[new_col] = df[glucose_col] * df[bmi_col]
                    created.append(new_col)
                except Exception:
                    pass

        return df, created

    @classmethod
    def _create_ratios(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create ratio features from health metrics."""
        created = []

        # Pulse Pressure (Systolic - Diastolic)
        sys_cols = cls._find_columns(df, ['bp_sys', 'systolic', 'blood_pressure'])
        dia_cols = cls._find_columns(df, ['bp_dia', 'diastolic'])
        for sys_col in sys_cols:
            for dia_col in dia_cols:
                new_col = "pulse_pressure"
                try:
                    df[new_col] = df[sys_col] - df[dia_col]
                    created.append(new_col)
                except Exception:
                    pass

        # Mean Arterial Pressure (MAP)
        for sys_col in sys_cols:
            for dia_col in dia_cols:
                new_col = "mean_arterial_pressure"
                try:
                    df[new_col] = df[dia_col] + (df[sys_col] - df[dia_col]) / 3
                    created.append(new_col)
                except Exception:
                    pass

        # BMI to Age ratio
        age_cols = cls._find_columns(df, ['age'])
        bmi_cols = cls._find_columns(df, ['bmi'])
        for age_col in age_cols:
            for bmi_col in bmi_cols:
                new_col = f"{bmi_col}_per_age_decade"
                try:
                    df[new_col] = df[bmi_col] / (df[age_col] / 10 + 1)
                    created.append(new_col)
                except Exception:
                    pass

        return df, created

    @classmethod
    def _create_risk_scores(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create composite risk scores."""
        created = []

        # Cardiovascular Risk Score (simplified)
        cv_risk_components = []

        # Check for relevant columns
        age_cols = cls._find_columns(df, ['age'])
        bmi_cols = cls._find_columns(df, ['bmi'])
        bp_cols = cls._find_columns(df, ['blood_pressure', 'bp_sys', 'systolic'])
        chol_cols = cls._find_columns(df, ['cholesterol'])
        smoke_cols = cls._find_columns(df, ['smoker', 'smoking'])
        diabetes_cols = cls._find_columns(df, ['diabetes', 'diabetic'])

        if age_cols or bmi_cols or bp_cols:
            try:
                risk_score = pd.Series(0.0, index=df.index)

                # Age component (normalized)
                if age_cols:
                    age_normalized = (df[age_cols[0]] - df[age_cols[0]].min()) / (df[age_cols[0]].max() - df[age_cols[0]].min() + 1e-8)
                    risk_score += age_normalized * 0.2
                    cv_risk_components.append('age')

                # BMI component
                if bmi_cols:
                    bmi_risk = ((df[bmi_cols[0]] - 25).clip(lower=0) / 15).clip(upper=1)
                    risk_score += bmi_risk * 0.2
                    cv_risk_components.append('bmi')

                # Blood pressure component
                if bp_cols:
                    bp_risk = ((df[bp_cols[0]] - 120).clip(lower=0) / 60).clip(upper=1)
                    risk_score += bp_risk * 0.25
                    cv_risk_components.append('blood_pressure')

                # Cholesterol component
                if chol_cols:
                    chol_risk = ((df[chol_cols[0]] - 200).clip(lower=0) / 100).clip(upper=1)
                    risk_score += chol_risk * 0.2
                    cv_risk_components.append('cholesterol')

                # Smoking component
                if smoke_cols:
                    smoke_binary = df[smoke_cols[0]].apply(
                        lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'y'] else 0
                    )
                    risk_score += smoke_binary * 0.15
                    cv_risk_components.append('smoking')

                if cv_risk_components:
                    df['cardiovascular_risk_score'] = risk_score
                    created.append('cardiovascular_risk_score')

            except Exception:
                pass

        # Metabolic Risk Score
        metabolic_components = []
        try:
            metabolic_score = pd.Series(0.0, index=df.index)

            if bmi_cols:
                bmi_risk = ((df[bmi_cols[0]] - 25).clip(lower=0) / 15).clip(upper=1)
                metabolic_score += bmi_risk * 0.35
                metabolic_components.append('bmi')

            glucose_cols = cls._find_columns(df, ['glucose', 'blood_sugar'])
            if glucose_cols:
                glucose_risk = ((df[glucose_cols[0]] - 100).clip(lower=0) / 100).clip(upper=1)
                metabolic_score += glucose_risk * 0.35
                metabolic_components.append('glucose')

            if bp_cols:
                bp_risk = ((df[bp_cols[0]] - 120).clip(lower=0) / 60).clip(upper=1)
                metabolic_score += bp_risk * 0.3
                metabolic_components.append('blood_pressure')

            if metabolic_components:
                df['metabolic_risk_score'] = metabolic_score
                created.append('metabolic_risk_score')

        except Exception:
            pass

        # Overall Health Risk Index
        try:
            if 'cardiovascular_risk_score' in df.columns and 'metabolic_risk_score' in df.columns:
                df['overall_health_risk_index'] = (
                    df['cardiovascular_risk_score'] * 0.5 +
                    df['metabolic_risk_score'] * 0.5
                )
                created.append('overall_health_risk_index')
        except Exception:
            pass

        return df, created

    @classmethod
    def _create_aggregations(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create aggregated/summary features."""
        created = []

        # Count of risk factors
        try:
            risk_factor_cols = []

            # High BMI
            bmi_cols = cls._find_columns(df, ['bmi'])
            if bmi_cols:
                risk_factor_cols.append((df[bmi_cols[0]] > 30).astype(int))

            # High blood pressure
            bp_cols = cls._find_columns(df, ['blood_pressure', 'bp_sys', 'systolic'])
            if bp_cols:
                risk_factor_cols.append((df[bp_cols[0]] > 140).astype(int))

            # Smoking
            smoke_cols = cls._find_columns(df, ['smoker', 'smoking'])
            if smoke_cols:
                risk_factor_cols.append(
                    df[smoke_cols[0]].apply(
                        lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'y'] else 0
                    )
                )

            # High cholesterol
            chol_cols = cls._find_columns(df, ['cholesterol'])
            if chol_cols:
                risk_factor_cols.append((df[chol_cols[0]] > 240).astype(int))

            # Low physical activity
            activity_cols = cls._find_columns(df, ['physactivity', 'physical_activity', 'exercise'])
            if activity_cols:
                risk_factor_cols.append(
                    df[activity_cols[0]].apply(
                        lambda x: 1 if str(x).lower() in ['no', '0', 'false', 'n'] else 0
                    )
                )

            if risk_factor_cols:
                df['total_risk_factors'] = sum(risk_factor_cols)
                created.append('total_risk_factors')

        except Exception:
            pass

        # Count of healthy behaviors
        try:
            healthy_cols = []

            # Regular exercise
            if activity_cols:
                healthy_cols.append(
                    df[activity_cols[0]].apply(
                        lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'y'] else 0
                    )
                )

            # Non-smoker
            if smoke_cols:
                healthy_cols.append(
                    df[smoke_cols[0]].apply(
                        lambda x: 1 if str(x).lower() in ['no', '0', 'false', 'n'] else 0
                    )
                )

            # Eats fruits
            fruit_cols = cls._find_columns(df, ['fruits', 'fruit'])
            if fruit_cols:
                healthy_cols.append(
                    df[fruit_cols[0]].apply(
                        lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'y'] else 0
                    )
                )

            # Eats vegetables
            veg_cols = cls._find_columns(df, ['veggies', 'vegetables', 'veg'])
            if veg_cols:
                healthy_cols.append(
                    df[veg_cols[0]].apply(
                        lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'y'] else 0
                    )
                )

            if healthy_cols:
                df['healthy_behavior_count'] = sum(healthy_cols)
                created.append('healthy_behavior_count')

        except Exception:
            pass

        return df, created

    @classmethod
    def _find_columns(cls, df: pd.DataFrame, patterns: List[str]) -> List[str]:
        """Find columns matching any of the given patterns (case-insensitive)."""
        matched = []
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_').replace('-', '_')
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    matched.append(col)
                    break
        return matched

    @classmethod
    def create_polynomial_features(
        cls,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        degree: int = 2,
        include_bias: bool = False,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create polynomial features for specified columns.

        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features for
            degree: Polynomial degree (default 2)
            include_bias: Include bias term (default False)

        Returns:
            Tuple of (DataFrame with polynomial features, list of new column names)
        """
        from sklearn.preprocessing import PolynomialFeatures

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) == 0:
            return df, []

        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(df[columns].fillna(0))

        feature_names = poly.get_feature_names_out(columns)

        # Only add new features (not the original ones)
        new_features = []
        for i, name in enumerate(feature_names):
            if name not in columns:
                clean_name = name.replace(' ', '_')
                df[clean_name] = poly_features[:, i]
                new_features.append(clean_name)

        return df, new_features
