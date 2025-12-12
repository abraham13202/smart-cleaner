"""Core modules for Smart Cleaner."""

from .ai_advisor import AIAdvisor
from .data_cleaner import DataCleaner
from .imputation import (
    ImputationEngine,
    ImputationStrategy,
    MeanImputation,
    MedianImputation,
    ModeImputation,
    CohortMeanImputation,
    KNNImputation,
)
from .eda import EDAAnalyzer
from .outliers import OutlierHandler
from .health_validator import HealthValidator
from .duplicates import DuplicateHandler
from .visualizer import DataVisualizer
from .auto_pipeline import AutoPreprocessor, PipelineConfig
from .report_generator import ReportGenerator

# New modules for comprehensive data analysis
from .feature_engineer import FeatureEngineer
from .feature_encoder import FeatureEncoder
from .feature_scaler import FeatureScaler
from .feature_selector import FeatureSelector
from .advanced_eda import AdvancedEDA
from .data_dictionary import DataDictionary
from .ml_ready_export import MLReadyExport

# AI-powered strategic modules
from .ai_strategy_advisor import AIStrategyAdvisor
from .technical_documenter import TechnicalDocumenter
from .smart_imputer import SmartImputer

# Quality assurance and validation modules
from .data_quality_auditor import DataQualityAuditor
from .data_leakage_detector import DataLeakageDetector
from .multicollinearity_checker import MulticollinearityChecker
from .class_balancer import ClassBalancer
from .final_quality_checker import FinalQualityChecker

# New v0.5 modules
from .profiler import DataProfiler
from .batch_processor import BatchProcessor
from .validators import DataValidator
from .audit import AuditTrail, AuditedDataFrame

__all__ = [
    "AIAdvisor",
    "DataCleaner",
    "ImputationEngine",
    "ImputationStrategy",
    "MeanImputation",
    "MedianImputation",
    "ModeImputation",
    "CohortMeanImputation",
    "KNNImputation",
    "EDAAnalyzer",
    "OutlierHandler",
    "HealthValidator",
    "DuplicateHandler",
    "DataVisualizer",
    "AutoPreprocessor",
    "PipelineConfig",
    "ReportGenerator",
    # New modules
    "FeatureEngineer",
    "FeatureEncoder",
    "FeatureScaler",
    "FeatureSelector",
    "AdvancedEDA",
    "DataDictionary",
    "MLReadyExport",
    # AI-powered strategic modules
    "AIStrategyAdvisor",
    "TechnicalDocumenter",
    "SmartImputer",
    # Quality assurance and validation modules
    "DataQualityAuditor",
    "DataLeakageDetector",
    "MulticollinearityChecker",
    "ClassBalancer",
    "FinalQualityChecker",
    # New v0.5 modules
    "DataProfiler",
    "BatchProcessor",
    "DataValidator",
    "AuditTrail",
    "AuditedDataFrame",
]
