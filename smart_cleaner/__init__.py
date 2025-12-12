"""
Smart Cleaner - AI-Powered Data Cleaning & Analysis Platform

A comprehensive data cleaning and analysis platform using local LLMs (Ollama).
100% free, 100% private - your data never leaves your machine.

Example usage:
    from smart_cleaner import AutoPreprocessor, PipelineConfig

    # Full automatic preprocessing with Ollama
    config = PipelineConfig(
        use_ai_recommendations=True,
        ai_provider="ollama",
        ollama_model="llama3.2",
        target_column="salary",
    )
    preprocessor = AutoPreprocessor(config)
    cleaned_df, report = preprocessor.process(df)

    # Data profiling
    from smart_cleaner import DataProfiler
    profiler = DataProfiler()
    profile = profiler.profile(df)
    profiler.print_summary(profile)

    # Batch processing
    from smart_cleaner import BatchProcessor
    processor = BatchProcessor(config)
    report = processor.process_directory("data/raw/", "data/cleaned/")

    # Validation rules
    from smart_cleaner import DataValidator
    validator = DataValidator()
    validator.add_not_null("id")
    validator.add_range("age", min_val=0, max_val=150)
    report = validator.validate(df)
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
    # New modules
    DataProfiler,
    BatchProcessor,
    DataValidator,
    AuditTrail,
    AuditedDataFrame,
)
from .utils import Config, YAMLPipelineConfig, load_config

__version__ = "0.5.0"
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
    # New modules
    "DataProfiler",
    "BatchProcessor",
    "DataValidator",
    "AuditTrail",
    "AuditedDataFrame",
    "YAMLPipelineConfig",
    "load_config",
]
