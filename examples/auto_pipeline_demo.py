"""
Comprehensive Auto-Preprocessing Pipeline Demo.

This example demonstrates the full automatic preprocessing pipeline
that handles everything:
- Duplicate removal
- Health metric validation
- Outlier detection and handling
- Missing value imputation (AI-powered)
- Target variable analysis
- Data visualizations

Just load your health dataset and let Smart Cleaner do everything!
"""

import pandas as pd
import numpy as np
from smart_cleaner import AutoPreprocessor, PipelineConfig


def create_realistic_health_dataset_with_issues():
    """
    Create a realistic health dataset with various data quality issues:
    - Missing values
    - Duplicates
    - Outliers
    - Invalid health metrics
    """
    np.random.seed(42)

    n_samples = 300

    # Create base data
    data = {
        'patient_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 85, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'bmi': np.random.normal(27, 5, n_samples),
        'blood_pressure': np.random.normal(125, 15, n_samples),
        'cholesterol': np.random.normal(200, 35, n_samples),
        'glucose': np.random.normal(105, 25, n_samples),
        'heart_rate': np.random.normal(75, 10, n_samples),
        'smoking': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'exercise_hours_per_week': np.random.randint(0, 15, n_samples),
        'has_heart_disease': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }

    df = pd.DataFrame(data)

    # Issue 1: Add some duplicate rows (5%)
    duplicate_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
    duplicate_rows = df.loc[duplicate_indices].copy()
    duplicate_rows['patient_id'] = range(n_samples + 1, n_samples + 1 + len(duplicate_rows))
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Issue 2: Add missing values
    # BMI: 15% missing
    df.loc[np.random.choice(df.index, size=int(0.15 * len(df)), replace=False), 'bmi'] = np.nan

    # Blood pressure: 10% missing
    df.loc[np.random.choice(df.index, size=int(0.10 * len(df)), replace=False), 'blood_pressure'] = np.nan

    # Cholesterol: 20% missing
    df.loc[np.random.choice(df.index, size=int(0.20 * len(df)), replace=False), 'cholesterol'] = np.nan

    # Gender: 5% missing
    df.loc[np.random.choice(df.index, size=int(0.05 * len(df)), replace=False), 'gender'] = np.nan

    # Issue 3: Add invalid health metric values
    # Invalid ages
    df.loc[np.random.choice(df.index, size=3, replace=False), 'age'] = [-5, 150, 200]

    # Invalid BMI
    invalid_bmi_indices = df['bmi'].dropna().sample(5).index
    df.loc[invalid_bmi_indices, 'bmi'] = [5, 80, 95, 3, 100]

    # Invalid blood pressure
    invalid_bp_indices = df['blood_pressure'].dropna().sample(5).index
    df.loc[invalid_bp_indices, 'blood_pressure'] = [30, 250, 300, 20, 280]

    # Issue 4: Add outliers (extreme but valid values)
    outlier_indices = df['heart_rate'].dropna().sample(10).index
    df.loc[outlier_indices, 'heart_rate'] = np.random.choice([40, 45, 180, 190, 200], 10)

    return df


def main():
    """Main demo of auto-preprocessing pipeline."""

    print("\n")
    print("=" * 80)
    print("SMART CLEANER - AUTOMATIC PREPROCESSING PIPELINE DEMO")
    print("=" * 80)
    print("\n")
    print("This demo shows the COMPLETE automatic data cleaning pipeline.")
    print("Just load your data and Smart Cleaner handles everything!")
    print("\n")

    # Create dataset with issues
    print("Creating synthetic health dataset with intentional data quality issues...")
    df = create_realistic_health_dataset_with_issues()

    print(f"\nDataset created:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicate rows: {df.duplicated().sum()}")
    print("\n")

    # Configure the pipeline
    print("Configuring automatic preprocessing pipeline...")
    print()

    config = PipelineConfig(
        # Duplicate handling
        remove_duplicates=True,
        duplicate_exclude_columns=['patient_id'],  # Don't check patient_id for duplicates

        # Health validation
        validate_health_metrics=True,
        health_fix_strategy='cap',  # Cap invalid values to valid ranges

        # Outlier handling
        handle_outliers=True,
        outlier_method='iqr',
        outlier_strategy='cap',
        outlier_exclude_columns=['patient_id'],

        # Missing value imputation
        impute_missing=True,
        use_ai_recommendations=True,  # Using Gemini AI
        # gemini_api_key is loaded from GEMINI_API_KEY environment variable

        # Visualization
        generate_visualizations=True,
        visualization_output_dir='./health_data_visualizations',

        # Target analysis
        target_column='has_heart_disease',  # Our prediction target
    )

    print("Configuration:")
    print(f"  Remove duplicates: {config.remove_duplicates}")
    print(f"  Validate health metrics: {config.validate_health_metrics}")
    print(f"  Handle outliers: {config.handle_outliers}")
    print(f"  Impute missing values: {config.impute_missing}")
    print(f"  AI-powered recommendations: {config.use_ai_recommendations}")
    print(f"  Target variable: {config.target_column}")
    print("\n")

    # Run the pipeline
    print("=" * 80)
    print("RUNNING AUTOMATIC PREPROCESSING PIPELINE")
    print("=" * 80)
    print("\n")

    preprocessor = AutoPreprocessor(config)
    cleaned_df, report = preprocessor.process(df)

    # Show summary
    print("\n")
    print(preprocessor.get_summary_report())

    # Show detailed results
    print("\n")
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print("\n")

    print("BEFORE vs AFTER:")
    print("-" * 80)
    print(f"Shape: {df.shape} → {cleaned_df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()} → {cleaned_df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()} → {cleaned_df.duplicated().sum()}")
    print("\n")

    # Show some examples of fixes
    print("EXAMPLE FIXES:")
    print("-" * 80)

    # Age fixes
    original_invalid_ages = df[df['age'] < 0]['age'].values
    if len(original_invalid_ages) > 0:
        print(f"Invalid ages: {original_invalid_ages} → Fixed to valid range")

    # BMI fixes
    original_invalid_bmi = df[(df['bmi'] < 10) | (df['bmi'] > 60)]['bmi'].dropna().values
    if len(original_invalid_bmi) > 0:
        print(f"Invalid BMI values: {original_invalid_bmi[:3]} → Capped to valid range [10-60]")

    # Blood pressure fixes
    original_invalid_bp = df[(df['blood_pressure'] < 70) | (df['blood_pressure'] > 200)]['blood_pressure'].dropna().values
    if len(original_invalid_bp) > 0:
        print(f"Invalid BP values: {original_invalid_bp[:3]} → Capped to valid range [70-200]")

    print("\n")

    # Show missing value handling
    print("MISSING VALUES HANDLED:")
    print("-" * 80)
    missing_before = df.isnull().sum()
    missing_after = cleaned_df.isnull().sum()

    for col in df.columns:
        if missing_before[col] > 0:
            print(f"{col}: {missing_before[col]} missing → {missing_after[col]} remaining")

    print("\n")

    # Save results
    output_file = "cleaned_health_data.csv"
    cleaned_df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
    print("\n")

    # Instructions for next steps
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n")
    print("1. Check the visualizations folder for data exploration plots")
    print(f"   Location: {config.visualization_output_dir}/")
    print("\n")
    print("2. To enable AI-powered imputation recommendations:")
    print("   - Set config.use_ai_recommendations = True")
    print("   - Provide config.anthropic_api_key = 'your-key'")
    print("\n")
    print("3. Your cleaned data is ready for machine learning!")
    print("   - No missing values (or properly imputed)")
    print("   - No invalid health metrics")
    print("   - Outliers handled")
    print("   - Duplicates removed")
    print("\n")

    print("=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n")

    # Show how to use with AI
    print("TIP: To use with AI-powered recommendations:")
    print("-" * 80)
    print("""
from smart_cleaner import AutoPreprocessor, PipelineConfig

config = PipelineConfig(
    gemini_api_key="your-api-key",  # Or set GEMINI_API_KEY env variable
    target_column="has_heart_disease",
    use_ai_recommendations=True,  # Enable AI
)

preprocessor = AutoPreprocessor(config)
cleaned_df, report = preprocessor.process(your_health_df)

# That's it! Everything is automatic.
    """)


if __name__ == "__main__":
    main()
