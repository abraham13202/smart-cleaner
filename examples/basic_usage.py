"""
Basic usage example for Smart Cleaner.

This example demonstrates how to use Smart Cleaner for data cleaning
with AI-powered recommendations using Ollama (local LLM).
"""

import pandas as pd
import numpy as np
from smart_cleaner.core.auto_pipeline import AutoPreprocessor, PipelineConfig
from smart_cleaner import EDAAnalyzer


def create_sample_healthcare_data():
    """Create sample healthcare dataset with missing values."""
    np.random.seed(42)

    # Create sample data
    n_samples = 200

    data = {
        'patient_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'bmi': np.random.normal(27, 5, n_samples),
        'blood_pressure': np.random.normal(120, 15, n_samples),
        'cholesterol': np.random.normal(200, 30, n_samples),
        'glucose': np.random.normal(100, 20, n_samples),
        'heart_rate': np.random.normal(75, 10, n_samples),
    }

    df = pd.DataFrame(data)

    # Introduce missing values randomly
    # BMI: 15% missing
    df.loc[np.random.choice(df.index, size=int(0.15 * n_samples), replace=False), 'bmi'] = np.nan

    # Blood pressure: 10% missing
    df.loc[np.random.choice(df.index, size=int(0.10 * n_samples), replace=False), 'blood_pressure'] = np.nan

    # Cholesterol: 20% missing
    df.loc[np.random.choice(df.index, size=int(0.20 * n_samples), replace=False), 'cholesterol'] = np.nan

    # Gender: 5% missing
    df.loc[np.random.choice(df.index, size=int(0.05 * n_samples), replace=False), 'gender'] = np.nan

    return df


def main():
    """Main example workflow."""

    print("=" * 60)
    print("Smart Cleaner - Basic Usage Example (Ollama)")
    print("=" * 60)
    print()

    # Create sample data
    print("Creating sample healthcare dataset...")
    df = create_sample_healthcare_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print()

    # Initialize EDA Analyzer
    print("-" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("-" * 60)
    eda = EDAAnalyzer()

    # Get data quality report
    quality_report = eda.data_quality_report(df)
    print(f"\nData Quality Report:")
    print(f"  Overall Quality Score: {quality_report['overall_quality_score']}")
    print(f"  Quality Level: {quality_report['quality_level']}")
    print(f"  Completeness: {quality_report['completeness_score']}%")
    print(f"  Columns with missing: {quality_report['columns_with_missing']}")

    # Analyze correlations
    correlations = eda.correlation_analysis(df, threshold=0.3)
    print(f"\nHigh Correlations Found: {correlations['high_correlations_count']}")
    for corr in correlations['correlations'][:3]:  # Top 3
        print(f"  {corr['column1']} <-> {corr['column2']}: {corr['correlation']:.3f} ({corr['strength']})")

    # AI-powered cleaning with Ollama
    print("\n" + "-" * 60)
    print("AI-POWERED DATA CLEANING (Ollama - Local)")
    print("-" * 60)

    try:
        # Configure pipeline with Ollama
        config = PipelineConfig(
            use_ai_recommendations=True,
            ai_provider="ollama",
            ollama_model="llama3.2",  # Change to your model: mistral, codellama, etc.
            remove_duplicates=True,
            handle_outliers=True,
            impute_missing=True,
            generate_visualizations=False,  # Set to True if you want plots
        )

        preprocessor = AutoPreprocessor(config)

        # Run the full pipeline
        print("\nRunning preprocessing pipeline with Ollama AI recommendations...")
        cleaned_df, report = preprocessor.process(df)

        # Show results
        print("\n" + "-" * 60)
        print("RESULTS")
        print("-" * 60)
        print(f"Original missing values: {df.isnull().sum().sum()}")
        print(f"Remaining missing values: {cleaned_df.isnull().sum().sum()}")

        # Compare before and after
        print("\n" + "-" * 60)
        print("BEFORE AND AFTER COMPARISON")
        print("-" * 60)

        for col in ['bmi', 'blood_pressure', 'cholesterol']:
            if col in df.columns:
                print(f"\n{col.upper()}:")
                print(f"  Before - Missing: {df[col].isnull().sum()}, Mean: {df[col].mean():.2f}")
                print(f"  After  - Missing: {cleaned_df[col].isnull().sum()}, Mean: {cleaned_df[col].mean():.2f}")

        # Print summary
        print("\n" + preprocessor.get_summary_report())

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Pull a model: ollama pull llama3.2")
        print("  3. Ollama should start automatically")
        print("  4. Install Python package: pip install ollama")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
