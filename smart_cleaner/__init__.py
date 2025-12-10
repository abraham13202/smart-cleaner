"""
Smart Cleaner - AI-Powered Data Cleaning & Analysis Platform

A comprehensive data cleaning and analysis platform that replaces manual data analysis.
Includes AI-powered recommendations, feature engineering, encoding, scaling, and ML-ready exports.

Example usage:
    from smart_cleaner import AutoPreprocessor, PipelineConfig

    # Full automatic preprocessing
    config = PipelineConfig(
        gemini_api_key="your-key",  # Or set GEMINI_API_KEY env variable
        target_column="disease",
        use_ai_recommendations=True,
    )
    preprocessor = AutoPreprocessor(config)
    cleaned_df, report = preprocessor.process(df)

    # For comprehensive analysis pipeline
    from smart_cleaner import (
        FeatureEngineer,
        FeatureEncoder,
        FeatureScaler,
        FeatureSelector,
        AdvancedEDA,
        DataDictionary,
        MLReadyExport,
    )
"""

from .core import (
    DataCleaner,
    AIAdvisor,
    EDAAnalyzer,
    ImputationEngine,
    AutoPreprocessor,
    PipelineConfig,
    OutlierHandler,
    HealthValidator,
    DuplicateHandler,
    DataVisualizer,
    ReportGenerator,
    # Comprehensive analysis modules
    FeatureEngineer,
    FeatureEncoder,
    FeatureScaler,
    FeatureSelector,
    AdvancedEDA,
    DataDictionary,
    MLReadyExport,
    # AI-powered strategic modules
    AIStrategyAdvisor,
    TechnicalDocumenter,
    SmartImputer,
    # Quality assurance and validation modules
    DataQualityAuditor,
    DataLeakageDetector,
    MulticollinearityChecker,
    ClassBalancer,
    FinalQualityChecker,
)
from .utils import Config

__version__ = "0.4.0"
__author__ = "Smart Cleaner Team"

__all__ = [
    # Core cleaning
    "DataCleaner",
    "AIAdvisor",
    "EDAAnalyzer",
    "ImputationEngine",
    "AutoPreprocessor",
    "PipelineConfig",
    "OutlierHandler",
    "HealthValidator",
    "DuplicateHandler",
    "DataVisualizer",
    "ReportGenerator",
    "Config",
    # Comprehensive analysis
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
]
