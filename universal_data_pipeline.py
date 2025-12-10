#!/usr/bin/env python3
"""
Universal AI-Powered Data Cleaning Pipeline (Ollama - Local LLM)

This pipeline processes ANY dataset using local AI for intelligent decisions.

Usage:
    python universal_data_pipeline.py <dataset.csv> [target_column]

Examples:
    python universal_data_pipeline.py data.csv
    python universal_data_pipeline.py patient_data.csv disease
    python universal_data_pipeline.py sales.csv revenue --model mistral
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import argparse

from smart_cleaner.core.auto_pipeline import AutoPreprocessor, PipelineConfig
from smart_cleaner.core.technical_documenter import TechnicalDocumenter
from smart_cleaner import EDAAnalyzer


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Universal Data Cleaning Pipeline with Ollama AI (Local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python universal_data_pipeline.py data.csv
    python universal_data_pipeline.py diabetes.csv Diabetes
    python universal_data_pipeline.py heart.csv target --model mistral
    python universal_data_pipeline.py data.csv --no-ai
        """
    )

    parser.add_argument("dataset", help="Path to CSV dataset")
    parser.add_argument("target", nargs="?", default=None,
                        help="Target column name (optional)")
    parser.add_argument("--model", default="llama3.2",
                        help="Ollama model to use (default: llama3.2)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV file path (default: <input>_cleaned.csv)")
    parser.add_argument("--no-ai", action="store_true",
                        help="Disable AI recommendations, use simple strategies")
    parser.add_argument("--no-duplicates", action="store_true",
                        help="Skip duplicate removal")
    parser.add_argument("--no-outliers", action="store_true",
                        help="Skip outlier handling")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip generating visualizations")

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.dataset):
        print(f"Error: File not found: {args.dataset}")
        sys.exit(1)

    start_time = datetime.now()

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print_section("UNIVERSAL DATA CLEANING PIPELINE (Ollama)")

    print(f"Loading dataset: {args.dataset}")
    try:
        df = pd.read_csv(args.dataset)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Initialize technical documenter
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    documenter = TechnicalDocumenter(model=args.model)
    documenter.start_documentation(df, dataset_name)

    # =========================================================================
    # STEP 2: EXPLORATORY DATA ANALYSIS
    # =========================================================================
    print_section("EXPLORATORY DATA ANALYSIS")

    eda = EDAAnalyzer()

    # Data quality report
    quality_report = eda.data_quality_report(df)
    print(f"Data Quality Report:")
    print(f"  Overall Quality Score: {quality_report['overall_quality_score']}")
    print(f"  Quality Level: {quality_report['quality_level']}")
    print(f"  Completeness: {quality_report['completeness_score']}%")
    print(f"  Columns with missing: {quality_report['columns_with_missing']}")

    # Missing values summary
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\nMissing Values:")
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")

    # Correlation analysis
    correlations = eda.correlation_analysis(df, threshold=0.3)
    if correlations['high_correlations_count'] > 0:
        print(f"\nHigh Correlations Found: {correlations['high_correlations_count']}")
        for corr in correlations['correlations'][:5]:
            print(f"  {corr['column1']} <-> {corr['column2']}: {corr['correlation']:.3f}")

    # =========================================================================
    # STEP 3: AI-POWERED DATA CLEANING
    # =========================================================================
    print_section("AI-POWERED DATA CLEANING")

    if args.no_ai:
        print("AI recommendations disabled. Using simple strategies.")
    else:
        print(f"Using Ollama model: {args.model}")
        print("Make sure Ollama is running with the model pulled.")

    # Configure pipeline
    config = PipelineConfig(
        use_ai_recommendations=not args.no_ai,
        ai_provider="ollama",
        ollama_model=args.model,
        target_column=args.target,
        remove_duplicates=not args.no_duplicates,
        handle_outliers=not args.no_outliers,
        impute_missing=True,
        generate_visualizations=not args.no_visualize,
        visualization_output_dir="./data_visualizations",
    )

    # Run pipeline
    preprocessor = AutoPreprocessor(config)
    cleaned_df, report = preprocessor.process(df)

    # =========================================================================
    # STEP 4: RESULTS SUMMARY
    # =========================================================================
    print_section("RESULTS SUMMARY")

    print(f"Original shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Cleaned shape: {cleaned_df.shape[0]:,} rows x {cleaned_df.shape[1]} columns")
    print(f"Rows removed: {df.shape[0] - cleaned_df.shape[0]:,}")

    print(f"\nMissing values:")
    print(f"  Before: {df.isnull().sum().sum():,}")
    print(f"  After: {cleaned_df.isnull().sum().sum():,}")

    # Data quality improvement
    quality_improvement = report.get('data_quality_improvement', {})
    if quality_improvement:
        print(f"\nData Quality Improvement:")
        print(f"  Completeness: {quality_improvement.get('completeness_before', 0):.1f}% -> {quality_improvement.get('completeness_after', 0):.1f}%")
        print(f"  Improvement: +{quality_improvement.get('completeness_improvement', 0):.1f}%")

    # =========================================================================
    # STEP 5: SAVE OUTPUT
    # =========================================================================
    print_section("SAVING OUTPUT")

    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.dataset)[0]
        output_path = f"{base_name}_cleaned.csv"

    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

    # Generate comprehensive RMD documentation
    print_section("GENERATING TECHNICAL DOCUMENTATION")
    rmd_path = f"{dataset_name}_documentation.Rmd"
    documenter.generate_rmd(cleaned_df, rmd_path, report)

    # Print visualization info
    if not args.no_visualize:
        print(f"\nVisualizations saved to: ./data_visualizations/")
        print("  - 01_missing_values.png")
        print("  - 02_correlation_matrix.png")
        print("  - 03_boxplots_outliers.png")
        print("  - 04_distributions.png")
        print("  - 05_categorical_distributions.png")
        print("  - 06_pairplot.png")
        print("  - 07_target_analysis.png")
        print("  - 08_data_quality_summary.png")
        print("  - 09_distribution_stats.png")

    # Print processing time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nTotal processing time: {duration:.1f} seconds")

    # Print summary report
    print("\n" + preprocessor.get_summary_report())

    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETE - DATA IS CLEAN AND READY FOR DATA SCIENTISTS")
    print("=" * 80)
    print("\nOutput files:")
    print(f"  1. Cleaned data: {output_path}")
    print(f"  2. Technical documentation: {rmd_path}")
    if not args.no_visualize:
        print(f"  3. Visualizations: ./data_visualizations/*.png (9 files)")
    print("\nThe RMD file contains EVERYTHING the data scientist needs:")
    print("  - Executive summary")
    print("  - Dataset overview and data dictionary")
    print("  - Methodology explanation")
    print("  - All imputation strategies with reasoning")
    print("  - Outlier treatment details")
    print("  - Before/after comparisons")
    print("  - Recommendations for next steps")
    print("  - R and Python code to load the data")
    print("\n")


if __name__ == "__main__":
    main()
