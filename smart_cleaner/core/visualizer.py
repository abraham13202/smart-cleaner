"""
Data visualization and exploration utilities.
Provides both visual plots and text-based summaries.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class DataVisualizer:
    """
    Comprehensive data visualization and exploration.
    Works with or without matplotlib/seaborn.
    """

    @staticmethod
    def has_plotting_libraries() -> Dict[str, bool]:
        """Check which plotting libraries are available."""
        libraries = {}

        try:
            import matplotlib.pyplot as plt
            libraries["matplotlib"] = True
        except ImportError:
            libraries["matplotlib"] = False

        try:
            import seaborn as sns
            libraries["seaborn"] = True
        except ImportError:
            libraries["seaborn"] = False

        return libraries

    @staticmethod
    def generate_data_profile(df: pd.DataFrame) -> str:
        """
        Generate comprehensive text-based data profile.

        Args:
            df: DataFrame to profile

        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("DATA PROFILE REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Basic info
        lines.append("BASIC INFORMATION")
        lines.append("-" * 80)
        lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        lines.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        lines.append("")

        # Data types
        lines.append("DATA TYPES")
        lines.append("-" * 80)
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            lines.append(f"  {dtype}: {count} columns")
        lines.append("")

        # Missing values
        lines.append("MISSING VALUES")
        lines.append("-" * 80)
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100)
        has_missing = missing[missing > 0]

        if len(has_missing) == 0:
            lines.append("  No missing values!")
        else:
            for col in has_missing.index:
                lines.append(f"  {col}: {missing[col]} ({missing_pct[col]:.1f}%)")
        lines.append("")

        # Duplicates
        lines.append("DUPLICATES")
        lines.append("-" * 80)
        dup_count = df.duplicated().sum()
        lines.append(f"  Duplicate rows: {dup_count} ({dup_count/len(df)*100:.1f}%)")
        lines.append("")

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            lines.append("NUMERIC COLUMNS SUMMARY")
            lines.append("-" * 80)

            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) > 0:
                    lines.append(f"\n{col}:")
                    lines.append(f"  Count: {len(series)}")
                    lines.append(f"  Mean: {series.mean():.2f}")
                    lines.append(f"  Median: {series.median():.2f}")
                    lines.append(f"  Std: {series.std():.2f}")
                    lines.append(f"  Min: {series.min():.2f}")
                    lines.append(f"  Max: {series.max():.2f}")

                    # Detect potential outliers
                    q1, q3 = series.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = series[(series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)]
                    if len(outliers) > 0:
                        lines.append(f"  Outliers: {len(outliers)} ({len(outliers)/len(series)*100:.1f}%)")
            lines.append("")

        # Categorical columns summary
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            lines.append("CATEGORICAL COLUMNS SUMMARY")
            lines.append("-" * 80)

            for col in cat_cols[:10]:  # Limit to first 10
                series = df[col].dropna()
                if len(series) > 0:
                    lines.append(f"\n{col}:")
                    lines.append(f"  Unique values: {series.nunique()}")
                    lines.append(f"  Most frequent: {series.mode()[0] if len(series.mode()) > 0 else 'N/A'}")

                    # Top 5 values
                    top_values = series.value_counts().head(5)
                    lines.append("  Top 5 values:")
                    for val, count in top_values.items():
                        lines.append(f"    {val}: {count} ({count/len(series)*100:.1f}%)")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    @staticmethod
    def analyze_correlations_text(
        df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> str:
        """
        Generate text-based correlation analysis.

        Args:
            df: DataFrame
            threshold: Minimum correlation to report

        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("CORRELATION ANALYSIS")
        lines.append("=" * 80)
        lines.append("")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            lines.append("Not enough numeric columns for correlation analysis")
            return "\n".join(lines)

        corr_matrix = df[numeric_cols].corr()

        # Find high correlations
        high_corr = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) >= threshold:
                    high_corr.append((col1, col2, corr_val))

        # Sort by absolute correlation
        high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

        if len(high_corr) == 0:
            lines.append(f"No correlations found above threshold {threshold}")
        else:
            lines.append(f"High Correlations (|r| >= {threshold}):")
            lines.append("-" * 80)

            for col1, col2, corr_val in high_corr:
                strength = "very strong" if abs(corr_val) >= 0.8 else "strong" if abs(corr_val) >= 0.6 else "moderate"
                lines.append(f"  {col1} <-> {col2}: {corr_val:.3f} ({strength})")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    @staticmethod
    def analyze_target_relationships(
        df: pd.DataFrame,
        target_column: str,
    ) -> str:
        """
        Analyze relationships between features and target variable.

        Args:
            df: DataFrame
            target_column: Name of target variable

        Returns:
            Formatted analysis report
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        lines = []
        lines.append("=" * 80)
        lines.append(f"TARGET ANALYSIS: {target_column}")
        lines.append("=" * 80)
        lines.append("")

        # Target variable info
        lines.append("TARGET VARIABLE INFORMATION")
        lines.append("-" * 80)

        target_series = df[target_column]
        lines.append(f"Type: {target_series.dtype}")
        lines.append(f"Missing values: {target_series.isnull().sum()} ({target_series.isnull().sum()/len(df)*100:.1f}%)")

        if pd.api.types.is_numeric_dtype(target_series):
            lines.append(f"Mean: {target_series.mean():.2f}")
            lines.append(f"Median: {target_series.median():.2f}")
            lines.append(f"Std: {target_series.std():.2f}")
            lines.append(f"Range: [{target_series.min():.2f}, {target_series.max():.2f}]")
        else:
            lines.append(f"Unique values: {target_series.nunique()}")
            value_counts = target_series.value_counts()
            lines.append("Distribution:")
            for val, count in value_counts.head(10).items():
                lines.append(f"  {val}: {count} ({count/len(target_series)*100:.1f}%)")

        lines.append("")

        # Correlation with numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_column]

        if len(numeric_cols) > 0 and pd.api.types.is_numeric_dtype(target_series):
            lines.append("CORRELATIONS WITH TARGET")
            lines.append("-" * 80)

            correlations = []
            for col in numeric_cols:
                corr = df[col].corr(target_series)
                if not np.isnan(corr):
                    correlations.append((col, corr))

            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            for col, corr in correlations[:15]:  # Top 15
                direction = "positive" if corr > 0 else "negative"
                strength = "very strong" if abs(corr) >= 0.8 else "strong" if abs(corr) >= 0.6 else "moderate" if abs(corr) >= 0.4 else "weak"
                lines.append(f"  {col}: {corr:.3f} ({strength} {direction})")

            lines.append("")

        # Categorical feature analysis
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        cat_cols = [col for col in cat_cols if col != target_column]

        if len(cat_cols) > 0:
            lines.append("CATEGORICAL FEATURES ANALYSIS")
            lines.append("-" * 80)

            for col in cat_cols[:10]:  # Limit to 10
                unique_count = df[col].nunique()
                lines.append(f"\n{col} (unique values: {unique_count}):")

                if pd.api.types.is_numeric_dtype(target_series):
                    # Show mean target per category
                    means = df.groupby(col)[target_column].mean().sort_values(ascending=False)
                    lines.append("  Mean target by category:")
                    for cat, mean in means.head(5).items():
                        lines.append(f"    {cat}: {mean:.2f}")
                else:
                    # Show distribution
                    lines.append("  Distribution:")
                    value_counts = df[col].value_counts().head(5)
                    for val, count in value_counts.items():
                        lines.append(f"    {val}: {count}")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    @staticmethod
    def create_visualizations(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        output_dir: str = "./visualizations",
    ) -> Dict[str, Any]:
        """
        Create comprehensive visualizations (if matplotlib/seaborn available).

        Args:
            df: DataFrame
            target_column: Optional target variable
            output_dir: Directory to save plots

        Returns:
            Dictionary with plot information
        """
        libraries = DataVisualizer.has_plotting_libraries()

        if not libraries["matplotlib"]:
            return {
                "status": "skipped",
                "reason": "matplotlib not installed",
                "recommendation": "Install with: pip install matplotlib seaborn",
            }

        import matplotlib.pyplot as plt
        if libraries["seaborn"]:
            import seaborn as sns
            sns.set_style("whitegrid")

        import os
        os.makedirs(output_dir, exist_ok=True)

        plots_created = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        print(f"    Creating visualizations for {len(numeric_cols)} numeric and {len(cat_cols)} categorical columns...")

        # 1. Missing values heatmap
        if df.isnull().sum().sum() > 0:
            plt.figure(figsize=(14, 6))
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            colors = ['#ff6b6b' if v > len(df)*0.3 else '#feca57' if v > len(df)*0.1 else '#48dbfb' for v in missing.values]
            missing.plot(kind='bar', color=colors)
            plt.title('Missing Values by Column (Red: >30%, Yellow: >10%, Blue: <10%)', fontsize=14)
            plt.xlabel('Column')
            plt.ylabel('Missing Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/01_missing_values.png", dpi=150)
            plt.close()
            plots_created.append("01_missing_values.png")

        # 2. Correlation heatmap
        if len(numeric_cols) >= 2:
            # Select top columns if too many
            cols_for_corr = numeric_cols[:20] if len(numeric_cols) > 20 else numeric_cols
            plt.figure(figsize=(14, 12))
            corr = df[cols_for_corr].corr()

            if libraries["seaborn"]:
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=len(cols_for_corr) <= 15, fmt='.2f',
                           cmap='RdYlBu_r', center=0, square=True,
                           linewidths=0.5, cbar_kws={"shrink": 0.8})
            else:
                plt.imshow(corr, cmap='coolwarm', aspect='auto')
                plt.colorbar()

            plt.title('Correlation Matrix Heatmap', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/02_correlation_matrix.png", dpi=150)
            plt.close()
            plots_created.append("02_correlation_matrix.png")

        # 3. Box plots for outlier detection
        if len(numeric_cols) > 0:
            cols_for_box = numeric_cols[:12] if len(numeric_cols) > 12 else numeric_cols
            n_box = len(cols_for_box)
            n_rows = (n_box + 3) // 4
            fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_box == 1 else axes.flatten()

            for idx, col in enumerate(cols_for_box):
                if libraries["seaborn"]:
                    sns.boxplot(data=df, y=col, ax=axes[idx], color='#74b9ff')
                else:
                    axes[idx].boxplot(df[col].dropna())
                axes[idx].set_title(f'{col}', fontsize=10)
                axes[idx].set_ylabel('')

            for idx in range(n_box, len(axes)):
                axes[idx].axis('off')

            plt.suptitle('Box Plots - Outlier Detection', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/03_boxplots_outliers.png", dpi=150)
            plt.close()
            plots_created.append("03_boxplots_outliers.png")

        # 4. Distribution plots (histograms with KDE)
        if len(numeric_cols) > 0:
            cols_for_dist = numeric_cols[:12] if len(numeric_cols) > 12 else numeric_cols
            n_dist = len(cols_for_dist)
            n_rows = (n_dist + 3) // 4
            fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_dist == 1 else axes.flatten()

            for idx, col in enumerate(cols_for_dist):
                data = df[col].dropna()
                if libraries["seaborn"]:
                    sns.histplot(data, kde=True, ax=axes[idx], color='#0984e3')
                else:
                    axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{col}', fontsize=10)
                axes[idx].set_xlabel('')

            for idx in range(n_dist, len(axes)):
                axes[idx].axis('off')

            plt.suptitle('Distribution Plots with Density', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/04_distributions.png", dpi=150)
            plt.close()
            plots_created.append("04_distributions.png")

        # 5. Categorical columns distribution
        if len(cat_cols) > 0:
            cols_for_cat = [c for c in cat_cols if df[c].nunique() <= 20][:8]
            if len(cols_for_cat) > 0:
                n_cat = len(cols_for_cat)
                n_rows = (n_cat + 1) // 2
                fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4*n_rows))
                axes = axes.flatten() if n_rows > 1 else [axes] if n_cat == 1 else axes

                for idx, col in enumerate(cols_for_cat):
                    value_counts = df[col].value_counts().head(10)
                    if libraries["seaborn"]:
                        sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[idx], palette='viridis')
                    else:
                        axes[idx].barh(value_counts.index, value_counts.values)
                    axes[idx].set_title(f'{col}', fontsize=10)
                    axes[idx].set_xlabel('Count')

                for idx in range(n_cat, len(axes)):
                    axes[idx].axis('off')

                plt.suptitle('Categorical Variables Distribution', fontsize=14, y=1.02)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/05_categorical_distributions.png", dpi=150)
                plt.close()
                plots_created.append("05_categorical_distributions.png")

        # 6. Pairplot for top correlated features
        if len(numeric_cols) >= 3 and libraries["seaborn"]:
            # Select top 5 most variable numeric columns
            variances = df[numeric_cols].var().sort_values(ascending=False)
            top_cols = variances.head(5).index.tolist()
            if target_column and target_column in numeric_cols and target_column not in top_cols:
                top_cols = top_cols[:4] + [target_column]

            plt.figure(figsize=(12, 12))
            pair_df = df[top_cols].dropna()
            if len(pair_df) > 1000:
                pair_df = pair_df.sample(1000, random_state=42)

            g = sns.pairplot(pair_df, diag_kind='kde', corner=True,
                           plot_kws={'alpha': 0.5, 's': 20})
            g.fig.suptitle('Pair Plot - Feature Relationships', y=1.02, fontsize=14)
            plt.savefig(f"{output_dir}/06_pairplot.png", dpi=150)
            plt.close()
            plots_created.append("06_pairplot.png")

        # 7. Target analysis plots
        if target_column and target_column in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            if pd.api.types.is_numeric_dtype(df[target_column]):
                # Distribution of target
                if libraries["seaborn"]:
                    sns.histplot(df[target_column].dropna(), kde=True, ax=axes[0], color='#e17055')
                else:
                    axes[0].hist(df[target_column].dropna(), bins=30, edgecolor='black')
                axes[0].set_title(f'{target_column} Distribution', fontsize=12)
                axes[0].set_xlabel(target_column)

                # Feature correlations with target
                correlations = df[numeric_cols].corrwith(df[target_column]).abs().sort_values(ascending=True)
                correlations = correlations[correlations.index != target_column].tail(15)
                correlations.plot(kind='barh', ax=axes[1], color='#00b894')
                axes[1].set_title(f'Top Feature Correlations with {target_column}', fontsize=12)
                axes[1].set_xlabel('Absolute Correlation')
            else:
                # Categorical target
                value_counts = df[target_column].value_counts()
                if libraries["seaborn"]:
                    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[0], palette='Set2')
                else:
                    axes[0].bar(value_counts.index, value_counts.values)
                axes[0].set_title(f'{target_column} Class Distribution', fontsize=12)
                axes[0].tick_params(axis='x', rotation=45)

                # Pie chart
                axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                           colors=plt.cm.Set2.colors[:len(value_counts)])
                axes[1].set_title(f'{target_column} Class Proportions', fontsize=12)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/07_target_analysis.png", dpi=150)
            plt.close()
            plots_created.append("07_target_analysis.png")

        # 8. Data quality summary visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Missing values percentage
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(15)
        if missing_pct.sum() > 0:
            missing_pct.plot(kind='barh', ax=axes[0], color='#fd79a8')
            axes[0].set_title('Top 15 Missing Value %', fontsize=12)
            axes[0].set_xlabel('Missing %')
        else:
            axes[0].text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', fontsize=14)
            axes[0].set_title('Missing Values', fontsize=12)
            axes[0].axis('off')

        # Data types distribution
        dtype_counts = df.dtypes.value_counts()
        axes[1].pie(dtype_counts.values, labels=[str(d) for d in dtype_counts.index],
                   autopct='%1.0f%%', colors=plt.cm.Pastel1.colors[:len(dtype_counts)])
        axes[1].set_title('Data Types Distribution', fontsize=12)

        # Memory usage by column type
        memory_by_type = df.memory_usage(deep=True).groupby(df.dtypes).sum() / 1024 / 1024
        memory_by_type.plot(kind='bar', ax=axes[2], color='#a29bfe')
        axes[2].set_title('Memory Usage by Type (MB)', fontsize=12)
        axes[2].set_xlabel('Data Type')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/08_data_quality_summary.png", dpi=150)
        plt.close()
        plots_created.append("08_data_quality_summary.png")

        # 9. Numeric statistics summary
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Skewness
            skewness = df[numeric_cols].skew().sort_values()
            colors = ['#ff6b6b' if abs(v) > 1 else '#feca57' if abs(v) > 0.5 else '#48dbfb' for v in skewness.values]
            skewness.plot(kind='barh', ax=axes[0], color=colors)
            axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            axes[0].set_title('Skewness (Red: High, Yellow: Moderate, Blue: Low)', fontsize=11)
            axes[0].set_xlabel('Skewness')

            # Kurtosis
            kurtosis = df[numeric_cols].kurtosis().sort_values()
            colors = ['#ff6b6b' if abs(v) > 3 else '#feca57' if abs(v) > 1 else '#48dbfb' for v in kurtosis.values]
            kurtosis.plot(kind='barh', ax=axes[1], color=colors)
            axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            axes[1].set_title('Kurtosis (Red: High, Yellow: Moderate, Blue: Normal)', fontsize=11)
            axes[1].set_xlabel('Kurtosis')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/09_distribution_stats.png", dpi=150)
            plt.close()
            plots_created.append("09_distribution_stats.png")

        print(f"    Created {len(plots_created)} visualizations in {output_dir}/")

        return {
            "status": "success",
            "plots_created": plots_created,
            "output_directory": output_dir,
        }
