"""
Class Balancer.
Handles class imbalance in classification datasets.

Class imbalance can severely affect model performance, causing:
- Bias toward majority class
- Poor minority class recall
- Misleading accuracy metrics

Methods implemented:
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Over-sampling
- Random Under-sampling
- SMOTE + Tomek Links
- Class Weight Calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings


class ClassBalancer:
    """
    Comprehensive class imbalance handling for classification problems.

    This is CRITICAL for datasets where one class significantly
    outnumbers others (e.g., fraud detection, disease diagnosis).
    """

    def __init__(self):
        """Initialize the balancer."""
        self.balance_report = {}
        self.original_distribution = {}

    def analyze_imbalance(
        self,
        y: Union[pd.Series, np.ndarray],
        imbalance_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Analyze class distribution and detect imbalance.

        Args:
            y: Target variable
            imbalance_threshold: Ratio below which a class is considered minority

        Returns:
            Imbalance analysis report
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        value_counts = y.value_counts()
        total = len(y)

        # Calculate percentages
        percentages = (value_counts / total * 100).round(2)

        # Calculate imbalance ratio
        majority_count = value_counts.max()
        minority_count = value_counts.min()
        imbalance_ratio = round(majority_count / minority_count, 2) if minority_count > 0 else float('inf')

        # Determine severity
        if imbalance_ratio < 2:
            severity = "LOW"
            recommendation = "Dataset is relatively balanced. No action needed."
        elif imbalance_ratio < 5:
            severity = "MODERATE"
            recommendation = "Consider using class weights or light oversampling."
        elif imbalance_ratio < 10:
            severity = "HIGH"
            recommendation = "Use SMOTE or significant oversampling. Consider undersampling majority."
        else:
            severity = "SEVERE"
            recommendation = "Aggressive balancing needed. Use SMOTE + undersampling combination."

        # Store original distribution
        self.original_distribution = {
            "counts": value_counts.to_dict(),
            "percentages": percentages.to_dict(),
            "imbalance_ratio": imbalance_ratio
        }

        return {
            "n_classes": len(value_counts),
            "class_distribution": {
                str(k): {
                    "count": int(v),
                    "percentage": float(percentages[k])
                }
                for k, v in value_counts.items()
            },
            "majority_class": str(value_counts.idxmax()),
            "minority_class": str(value_counts.idxmin()),
            "majority_count": int(majority_count),
            "minority_count": int(minority_count),
            "imbalance_ratio": imbalance_ratio,
            "severity": severity,
            "recommendation": recommendation,
            "is_imbalanced": imbalance_ratio > 2,
        }

    def balance(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        method: str = 'auto',
        target_ratio: float = 1.0,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Balance the dataset using specified method.

        Args:
            X: Feature DataFrame
            y: Target variable
            method: Balancing method ('auto', 'smote', 'oversample', 'undersample', 'smote_tomek')
            target_ratio: Target ratio of minority to majority (1.0 = equal)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (balanced_X, balanced_y, balance_report)
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='target')

        # Analyze current imbalance
        analysis = self.analyze_imbalance(y)

        if method == 'auto':
            method = self._select_method(analysis)

        report = {
            "method_used": method,
            "original_distribution": analysis["class_distribution"],
            "original_size": len(X),
        }

        # Apply balancing
        if method == 'smote':
            X_balanced, y_balanced = self._apply_smote(X, y, target_ratio, random_state)
        elif method == 'oversample':
            X_balanced, y_balanced = self._random_oversample(X, y, target_ratio, random_state)
        elif method == 'undersample':
            X_balanced, y_balanced = self._random_undersample(X, y, target_ratio, random_state)
        elif method == 'smote_tomek':
            X_balanced, y_balanced = self._apply_smote_tomek(X, y, random_state)
        elif method == 'none':
            X_balanced, y_balanced = X.copy(), y.copy()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Analyze new distribution
        new_analysis = self.analyze_imbalance(y_balanced)

        report.update({
            "new_distribution": new_analysis["class_distribution"],
            "new_size": len(X_balanced),
            "size_change": len(X_balanced) - len(X),
            "new_imbalance_ratio": new_analysis["imbalance_ratio"],
        })

        self.balance_report = report
        return X_balanced, y_balanced, report

    def _select_method(self, analysis: Dict) -> str:
        """Auto-select the best balancing method."""
        severity = analysis["severity"]
        minority_count = analysis["minority_count"]

        if severity == "LOW":
            return "none"
        elif severity == "MODERATE":
            return "oversample"
        elif minority_count < 100:
            # Very few samples - SMOTE might not work well
            return "oversample"
        else:
            return "smote"

    def _apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_ratio: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique).

        SMOTE creates synthetic samples by:
        1. Selecting a minority sample
        2. Finding its k nearest neighbors
        3. Creating new sample along the line between them
        """
        try:
            from imblearn.over_sampling import SMOTE

            # Calculate sampling strategy
            value_counts = y.value_counts()
            majority_count = value_counts.max()
            sampling_strategy = {
                cls: int(majority_count * target_ratio)
                for cls, count in value_counts.items()
                if count < majority_count * target_ratio
            }

            if not sampling_strategy:
                return X.copy(), y.copy()

            # Determine k_neighbors based on minority class size
            min_samples = min(value_counts[cls] for cls in sampling_strategy.keys())
            k_neighbors = min(5, min_samples - 1)

            if k_neighbors < 1:
                # Fall back to random oversampling if SMOTE not possible
                return self._random_oversample(X, y, target_ratio, random_state)

            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=k_neighbors
            )

            X_resampled, y_resampled = smote.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)

        except ImportError:
            warnings.warn("imbalanced-learn not installed. Using random oversampling instead.")
            return self._random_oversample(X, y, target_ratio, random_state)
        except Exception as e:
            warnings.warn(f"SMOTE failed: {e}. Using random oversampling instead.")
            return self._random_oversample(X, y, target_ratio, random_state)

    def _apply_smote_tomek(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE + Tomek Links.

        This combines:
        1. SMOTE for oversampling minority
        2. Tomek links removal for cleaning decision boundary
        """
        try:
            from imblearn.combine import SMOTETomek

            smote_tomek = SMOTETomek(random_state=random_state)
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)

        except ImportError:
            warnings.warn("imbalanced-learn not installed. Using SMOTE only.")
            return self._apply_smote(X, y, 1.0, random_state)
        except Exception as e:
            warnings.warn(f"SMOTE-Tomek failed: {e}. Using random oversampling.")
            return self._random_oversample(X, y, 1.0, random_state)

    def _random_oversample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_ratio: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Random oversampling of minority classes.

        Simply duplicates minority class samples randomly.
        """
        np.random.seed(random_state)

        value_counts = y.value_counts()
        majority_count = value_counts.max()
        target_count = int(majority_count * target_ratio)

        X_list = [X]
        y_list = [y]

        for cls, count in value_counts.items():
            if count < target_count:
                # Get indices of this class
                class_indices = y[y == cls].index

                # How many samples to add
                n_to_add = target_count - count

                # Random sample with replacement
                sampled_indices = np.random.choice(class_indices, size=n_to_add, replace=True)

                X_list.append(X.loc[sampled_indices])
                y_list.append(y.loc[sampled_indices])

        X_resampled = pd.concat(X_list, ignore_index=True)
        y_resampled = pd.concat(y_list, ignore_index=True)

        return X_resampled, y_resampled

    def _random_undersample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_ratio: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Random undersampling of majority class.

        Randomly removes samples from majority class.
        Warning: Can lose important information!
        """
        np.random.seed(random_state)

        value_counts = y.value_counts()
        minority_count = value_counts.min()
        target_count = int(minority_count / target_ratio) if target_ratio > 0 else minority_count

        X_list = []
        y_list = []

        for cls, count in value_counts.items():
            class_indices = y[y == cls].index

            if count > target_count:
                # Undersample this class
                sampled_indices = np.random.choice(class_indices, size=target_count, replace=False)
            else:
                sampled_indices = class_indices

            X_list.append(X.loc[sampled_indices])
            y_list.append(y.loc[sampled_indices])

        X_resampled = pd.concat(X_list, ignore_index=True)
        y_resampled = pd.concat(y_list, ignore_index=True)

        return X_resampled, y_resampled

    def calculate_class_weights(
        self,
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[Any, float]:
        """
        Calculate class weights for use in model training.

        This is an alternative to resampling - let the model handle imbalance.

        Formula: weight = n_samples / (n_classes * n_samples_for_class)
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        value_counts = y.value_counts()
        n_samples = len(y)
        n_classes = len(value_counts)

        weights = {}
        for cls, count in value_counts.items():
            weights[cls] = n_samples / (n_classes * count)

        return weights

    def get_sample_weights(
        self,
        y: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Get sample weights array for training.

        Returns array of weights where each sample's weight
        is based on its class weight.
        """
        class_weights = self.calculate_class_weights(y)

        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        return np.array([class_weights[cls] for cls in y])

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate balancing report."""
        if not self.balance_report:
            return "No balancing performed. Run balance() first."

        r = self.balance_report
        lines = []

        lines.append("=" * 80)
        lines.append("CLASS BALANCING REPORT")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"Method Used: {r['method_used']}")
        lines.append(f"Original Size: {r['original_size']:,}")
        lines.append(f"New Size: {r['new_size']:,}")
        lines.append(f"Change: {r['size_change']:+,} samples")
        lines.append("")

        lines.append("Original Distribution:")
        for cls, info in r['original_distribution'].items():
            lines.append(f"  {cls}: {info['count']:,} ({info['percentage']:.1f}%)")
        lines.append("")

        lines.append("New Distribution:")
        for cls, info in r['new_distribution'].items():
            lines.append(f"  {cls}: {info['count']:,} ({info['percentage']:.1f}%)")
        lines.append("")

        lines.append(f"New Imbalance Ratio: {r['new_imbalance_ratio']}")
        lines.append("=" * 80)

        report_text = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text
