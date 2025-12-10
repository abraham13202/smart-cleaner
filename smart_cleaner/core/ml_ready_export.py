"""
ML-Ready Export module.
Prepares data for machine learning with train/test splits, class balancing, and multiple export formats.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os


class MLReadyExport:
    """
    Prepares and exports ML-ready datasets.
    Handles train/test splits, class imbalance, and multiple export formats.
    """

    def __init__(self):
        """Initialize ML export handler."""
        self.export_report = {}
        self.splits = {}

    def prepare_ml_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
        random_state: int = 42,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Prepare ML-ready train/val/test splits.

        Args:
            df: Input DataFrame
            target_column: Target variable name
            test_size: Proportion for test set
            val_size: Proportion for validation set (from training)
            stratify: Whether to stratify splits (for classification)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (splits dict, preparation report)
        """
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        report = {
            "original_shape": df.shape,
            "target_column": target_column,
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state,
        }

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Determine if classification
        is_classification = y.nunique() <= 20
        report["task_type"] = "classification" if is_classification else "regression"

        # Stratify only for classification
        stratify_col = y if (stratify and is_classification) else None

        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_col,
            random_state=random_state
        )

        # Second split: train vs val
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)
            stratify_trainval = y_trainval if (stratify and is_classification) else None

            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval,
                test_size=val_ratio,
                stratify=stratify_trainval,
                random_state=random_state
            )

            splits = {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            }

            report["split_sizes"] = {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            }
        else:
            splits = {
                "X_train": X_trainval,
                "X_test": X_test,
                "y_train": y_trainval,
                "y_test": y_test,
            }

            report["split_sizes"] = {
                "train": len(X_trainval),
                "test": len(X_test),
            }

        # Class distribution analysis
        if is_classification:
            report["class_distribution"] = self._analyze_class_distribution(
                y, y_train if val_size == 0 else splits["y_train"], y_test
            )

        # Feature summary
        report["n_features"] = len(X.columns)
        report["feature_names"] = list(X.columns)
        report["numeric_features"] = list(X.select_dtypes(include=[np.number]).columns)
        report["categorical_features"] = list(X.select_dtypes(include=['object', 'category']).columns)

        self.splits = splits
        self.export_report = report
        return splits, report

    def _analyze_class_distribution(
        self,
        y_full: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Analyze class distribution across splits."""
        def get_distribution(y):
            counts = y.value_counts()
            return {
                "counts": counts.to_dict(),
                "percentages": (counts / len(y) * 100).round(2).to_dict(),
            }

        full_dist = get_distribution(y_full)
        train_dist = get_distribution(y_train)
        test_dist = get_distribution(y_test)

        # Calculate imbalance metrics
        full_counts = y_full.value_counts()
        majority = full_counts.iloc[0]
        minority = full_counts.iloc[-1]

        return {
            "full_dataset": full_dist,
            "train_set": train_dist,
            "test_set": test_dist,
            "n_classes": len(full_counts),
            "majority_class": str(full_counts.index[0]),
            "minority_class": str(full_counts.index[-1]),
            "imbalance_ratio": round(majority / minority, 2),
            "is_imbalanced": (majority / minority) > 3,
        }

    def handle_class_imbalance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = 'smote',
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Handle class imbalance using various methods.

        Args:
            X_train: Training features
            y_train: Training target
            method: 'smote', 'random_oversample', 'random_undersample', 'class_weight'
            random_state: Random seed

        Returns:
            Tuple of (balanced X, balanced y, balancing report)
        """
        report = {
            "method": method,
            "original_shape": X_train.shape,
            "original_distribution": y_train.value_counts().to_dict(),
        }

        if method == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=random_state)
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
            except ImportError:
                raise ImportError("imbalanced-learn required for SMOTE. Install with: pip install imbalanced-learn")

        elif method == 'random_oversample':
            try:
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=random_state)
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
            except ImportError:
                raise ImportError("imbalanced-learn required. Install with: pip install imbalanced-learn")

        elif method == 'random_undersample':
            try:
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=random_state)
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
            except ImportError:
                raise ImportError("imbalanced-learn required. Install with: pip install imbalanced-learn")

        elif method == 'class_weight':
            # Return original data with computed class weights
            class_counts = y_train.value_counts()
            total = len(y_train)
            n_classes = len(class_counts)
            class_weights = {
                cls: total / (n_classes * count)
                for cls, count in class_counts.items()
            }
            report["class_weights"] = class_weights
            return X_train, y_train, report

        else:
            raise ValueError(f"Unknown method: {method}")

        report["balanced_shape"] = X_balanced.shape
        report["balanced_distribution"] = pd.Series(y_balanced).value_counts().to_dict()

        return pd.DataFrame(X_balanced, columns=X_train.columns), pd.Series(y_balanced), report

    def get_class_weights(self, y: pd.Series) -> Dict:
        """Calculate class weights for imbalanced classification."""
        class_counts = y.value_counts()
        total = len(y)
        n_classes = len(class_counts)

        # Balanced class weights
        weights = {
            cls: total / (n_classes * count)
            for cls, count in class_counts.items()
        }

        return weights

    def export_splits(
        self,
        output_dir: str,
        format: str = 'csv',
        include_combined: bool = True,
    ) -> Dict[str, str]:
        """
        Export train/val/test splits to files.

        Args:
            output_dir: Output directory
            format: 'csv', 'parquet', or 'pickle'
            include_combined: Also save combined train+target files

        Returns:
            Dictionary of exported file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        exported_files = {}

        for name, data in self.splits.items():
            if format == 'csv':
                filepath = os.path.join(output_dir, f"{name}.csv")
                data.to_csv(filepath, index=False)
            elif format == 'parquet':
                filepath = os.path.join(output_dir, f"{name}.parquet")
                data.to_parquet(filepath, index=False)
            elif format == 'pickle':
                filepath = os.path.join(output_dir, f"{name}.pkl")
                data.to_pickle(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

            exported_files[name] = filepath

        # Export combined files
        if include_combined:
            if "X_train" in self.splits and "y_train" in self.splits:
                train_combined = self.splits["X_train"].copy()
                train_combined['target'] = self.splits["y_train"].values

                if format == 'csv':
                    filepath = os.path.join(output_dir, "train_combined.csv")
                    train_combined.to_csv(filepath, index=False)
                elif format == 'parquet':
                    filepath = os.path.join(output_dir, "train_combined.parquet")
                    train_combined.to_parquet(filepath, index=False)

                exported_files["train_combined"] = filepath

            if "X_test" in self.splits and "y_test" in self.splits:
                test_combined = self.splits["X_test"].copy()
                test_combined['target'] = self.splits["y_test"].values

                if format == 'csv':
                    filepath = os.path.join(output_dir, "test_combined.csv")
                    test_combined.to_csv(filepath, index=False)
                elif format == 'parquet':
                    filepath = os.path.join(output_dir, "test_combined.parquet")
                    test_combined.to_parquet(filepath, index=False)

                exported_files["test_combined"] = filepath

        # Save metadata
        metadata = {
            "export_format": format,
            "files": exported_files,
            "report": self.export_report,
        }

        metadata_path = os.path.join(output_dir, "ml_data_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        exported_files["metadata"] = metadata_path

        return exported_files

    def get_ml_summary(self) -> str:
        """Get human-readable ML data summary."""
        lines = []
        r = self.export_report

        lines.append("=" * 60)
        lines.append("ML-READY DATA SUMMARY")
        lines.append("=" * 60)

        lines.append(f"\nTask Type: {r.get('task_type', 'Unknown').upper()}")
        lines.append(f"Target Column: {r.get('target_column', 'N/A')}")
        lines.append(f"Original Shape: {r.get('original_shape', 'N/A')}")

        lines.append("\nSplit Sizes:")
        for split_name, size in r.get("split_sizes", {}).items():
            lines.append(f"  - {split_name}: {size:,} samples")

        lines.append(f"\nFeatures: {r.get('n_features', 'N/A')}")
        lines.append(f"  - Numeric: {len(r.get('numeric_features', []))}")
        lines.append(f"  - Categorical: {len(r.get('categorical_features', []))}")

        if "class_distribution" in r:
            cd = r["class_distribution"]
            lines.append(f"\nClass Distribution:")
            lines.append(f"  - Number of classes: {cd.get('n_classes', 'N/A')}")
            lines.append(f"  - Majority class: {cd.get('majority_class', 'N/A')}")
            lines.append(f"  - Minority class: {cd.get('minority_class', 'N/A')}")
            lines.append(f"  - Imbalance ratio: {cd.get('imbalance_ratio', 'N/A')}:1")
            lines.append(f"  - Is imbalanced: {cd.get('is_imbalanced', 'N/A')}")

            if cd.get('is_imbalanced'):
                lines.append("\n  ⚠️  RECOMMENDATION: Consider using class weights or resampling")
                lines.append("     Options: SMOTE, RandomOverSampler, class_weight='balanced'")

        lines.append("\nFeature Names:")
        features = r.get("feature_names", [])
        for i, feat in enumerate(features[:20], 1):
            lines.append(f"  {i}. {feat}")
        if len(features) > 20:
            lines.append(f"  ... and {len(features) - 20} more")

        return "\n".join(lines)

    def generate_baseline_code(self, output_path: Optional[str] = None) -> str:
        """
        Generate Python code for baseline ML models.

        Args:
            output_path: Path to save the code file

        Returns:
            Generated Python code as string
        """
        r = self.export_report
        is_classification = r.get("task_type") == "classification"

        code = '''"""
Baseline ML Model Training Script
Auto-generated by Smart Cleaner
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
X_train = pd.read_csv("ml_ready_data/X_train.csv")
X_test = pd.read_csv("ml_ready_data/X_test.csv")
y_train = pd.read_csv("ml_ready_data/y_train.csv").values.ravel()
y_test = pd.read_csv("ml_ready_data/y_test.csv").values.ravel()

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")

'''

        if is_classification:
            code += '''
# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

'''
            if r.get("class_distribution", {}).get("is_imbalanced"):
                code += '''
# Note: Data is imbalanced. Consider using class_weight='balanced'
models["Logistic Regression (Balanced)"] = LogisticRegression(
    max_iter=1000, class_weight='balanced', random_state=42
)
models["Random Forest (Balanced)"] = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42
)

'''

            code += '''
print("\\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

results = {}
for name, model in models.items():
    print(f"\\nTraining {name}...")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    # Fit and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
    }

    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")

# Best model
best_model = max(results.items(), key=lambda x: x[1]["test_f1"])
print(f"\\n{'='*60}")
print(f"BEST MODEL: {best_model[0]}")
print(f"Test F1-Score: {best_model[1]['test_f1']:.4f}")
print(f"{'='*60}")

# Detailed report for best model
print("\\nDetailed Classification Report:")
model = models[best_model[0]]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
'''

        else:
            code += '''
# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("\\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

results = {}
for name, model in models.items():
    print(f"\\nTraining {name}...")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    # Fit and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_r2": test_r2,
        "test_rmse": test_rmse,
    }

    print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")

# Best model
best_model = max(results.items(), key=lambda x: x[1]["test_r2"])
print(f"\\n{'='*60}")
print(f"BEST MODEL: {best_model[0]}")
print(f"Test R²: {best_model[1]['test_r2']:.4f}")
print(f"{'='*60}")
'''

        code += '''

# Save results
import json
with open("ml_ready_data/model_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\\nResults saved to ml_ready_data/model_results.json")
'''

        if output_path:
            with open(output_path, 'w') as f:
                f.write(code)

        return code
