"""
Cohort-based imputation demonstration.

This example shows the difference between simple mean imputation
and smart cohort-based imputation using age groups for BMI data.
"""

import pandas as pd
import numpy as np
from smart_cleaner import ImputationEngine


def create_age_bmi_dataset():
    """
    Create realistic dataset where BMI varies by age group.

    Realistic BMI distributions by age:
    - 18-25: Lower BMI (mean ~23)
    - 26-35: Moderate BMI (mean ~25)
    - 36-50: Higher BMI (mean ~27)
    - 51-65: Highest BMI (mean ~28)
    - 65+: Slightly lower (mean ~26)
    """
    np.random.seed(42)

    data = []

    # Generate data for each age cohort with different BMI distributions
    cohorts = [
        (18, 25, 23, 3.5),  # (age_min, age_max, bmi_mean, bmi_std)
        (26, 35, 25, 4.0),
        (36, 50, 27, 4.5),
        (51, 65, 28, 5.0),
        (66, 80, 26, 4.0),
    ]

    for age_min, age_max, bmi_mean, bmi_std in cohorts:
        n_samples = 40  # 40 samples per cohort
        ages = np.random.randint(age_min, age_max + 1, n_samples)
        bmis = np.random.normal(bmi_mean, bmi_std, n_samples)

        for age, bmi in zip(ages, bmis):
            data.append({'age': age, 'bmi': bmi})

    df = pd.DataFrame(data)

    # Introduce missing values (30% missing)
    missing_indices = np.random.choice(
        df.index,
        size=int(0.3 * len(df)),
        replace=False
    )
    df.loc[missing_indices, 'bmi'] = np.nan

    return df


def compare_imputation_methods(df):
    """Compare simple mean vs cohort-based imputation."""

    print("=" * 70)
    print("COMPARISON: Simple Mean vs Smart Cohort-Based Imputation")
    print("=" * 70)
    print()

    # Show original data distribution
    print("Original Data (non-missing values):")
    print("-" * 70)
    print(f"Total samples: {len(df)}")
    print(f"Missing BMI values: {df['bmi'].isnull().sum()} ({df['bmi'].isnull().sum()/len(df)*100:.1f}%)")
    print()

    # Show BMI distribution by age cohort (original data)
    df['age_cohort'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 50, 65, 100],
        labels=['18-25', '26-35', '36-50', '51-65', '65+']
    )

    print("Original BMI by Age Cohort (non-missing only):")
    print(df.groupby('age_cohort')['bmi'].agg(['mean', 'count']))
    print()

    # Method 1: Simple mean imputation
    print("=" * 70)
    print("METHOD 1: Simple Mean Imputation")
    print("=" * 70)
    df_simple = df.copy()

    overall_mean = df_simple['bmi'].mean()
    print(f"Overall BMI mean: {overall_mean:.2f}")
    print("Imputing ALL missing values with overall mean...")

    df_simple['bmi'] = ImputationEngine.impute(df_simple, 'bmi', 'mean')

    print("\nBMI by Age Cohort (after simple mean imputation):")
    print(df_simple.groupby('age_cohort')['bmi'].agg(['mean', 'count']))
    print()

    # Method 2: Cohort-based mean imputation
    print("=" * 70)
    print("METHOD 2: Smart Cohort-Based Mean Imputation")
    print("=" * 70)
    df_cohort = df.copy()

    print("Imputing missing values based on age cohort...")
    print("Example: 24-year-old gets mean BMI of 18-25 age group")
    print("         55-year-old gets mean BMI of 51-65 age group")
    print()

    df_cohort['bmi'] = ImputationEngine.impute(
        df_cohort,
        'bmi',
        'cohort_mean',
        parameters={
            'cohort_column': 'age',
            'cohort_bins': [0, 25, 35, 50, 65, 100]
        }
    )

    print("BMI by Age Cohort (after cohort-based imputation):")
    print(df_cohort.groupby('age_cohort')['bmi'].agg(['mean', 'count']))
    print()

    # Compare the results
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()

    comparison_data = []
    for cohort in ['18-25', '26-35', '36-50', '51-65', '65+']:
        original_mean = df[df['age_cohort'] == cohort]['bmi'].mean()
        simple_mean = df_simple[df_simple['age_cohort'] == cohort]['bmi'].mean()
        cohort_mean = df_cohort[df_cohort['age_cohort'] == cohort]['bmi'].mean()

        comparison_data.append({
            'Age Cohort': cohort,
            'Original Mean': f"{original_mean:.2f}",
            'Simple Mean': f"{simple_mean:.2f}",
            'Cohort Mean': f"{cohort_mean:.2f}",
            'Simple Δ': f"{abs(simple_mean - original_mean):.2f}",
            'Cohort Δ': f"{abs(cohort_mean - original_mean):.2f}",
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()

    print("Key Insights:")
    print("-" * 70)
    print("1. Simple mean imputation pulls all cohorts toward the overall mean")
    print("2. This distorts the natural age-BMI relationship in the data")
    print("3. Cohort-based imputation preserves the age-specific BMI patterns")
    print("4. Cohort Δ (deviation) is smaller, meaning better preservation of")
    print("   original data distribution")
    print()

    # Show example cases
    print("=" * 70)
    print("EXAMPLE CASES")
    print("=" * 70)
    print()

    # Find some missing value cases from different age groups
    missing_mask = df['bmi'].isnull()

    example_ages = []
    for cohort_label, (min_age, max_age) in [
        ('18-25', (18, 25)),
        ('36-50', (36, 50)),
        ('65+', (66, 80))
    ]:
        cohort_missing = df[missing_mask & (df['age'] >= min_age) & (df['age'] <= max_age)]
        if len(cohort_missing) > 0:
            example_ages.append((cohort_label, cohort_missing.iloc[0]['age']))

    for cohort_label, age in example_ages:
        idx = df[(df['age'] == age) & missing_mask].index[0]

        simple_value = df_simple.loc[idx, 'bmi']
        cohort_value = df_cohort.loc[idx, 'bmi']

        print(f"Patient aged {int(age)} (cohort: {cohort_label}):")
        print(f"  Simple Mean Imputation: BMI = {simple_value:.2f}")
        print(f"  Cohort Mean Imputation: BMI = {cohort_value:.2f}")
        print(f"  Difference: {abs(cohort_value - simple_value):.2f}")
        print()

    print("This demonstrates why cohort-based imputation is superior for")
    print("healthcare data where age-related patterns are significant!")


def main():
    """Main execution."""
    # Create dataset
    df = create_age_bmi_dataset()

    # Run comparison
    compare_imputation_methods(df)

    print("\n" + "=" * 70)
    print("For AI-powered recommendations, use DataCleaner with your API key:")
    print("=" * 70)
    print("""
from smart_cleaner import DataCleaner

cleaner = DataCleaner(api_key="your-anthropic-api-key")
analysis = cleaner.analyze(df)

# AI will automatically detect age-BMI correlation and recommend
# cohort-based imputation with appropriate age bins!
cleaned_df = cleaner.clean(df, auto_apply=True)
    """)


if __name__ == "__main__":
    main()
