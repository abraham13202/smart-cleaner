"""
Batch Processing Module for Smart Cleaner.
Process multiple files with the same pipeline configuration.
"""

import os
import glob
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json

from .auto_pipeline import AutoPreprocessor
from ..utils.config import PipelineConfig
from ..utils.progress import ProgressBar, StepProgress


@dataclass
class BatchResult:
    """Result for a single file in batch processing."""
    file_path: str
    success: bool
    output_path: Optional[str] = None
    rows_before: int = 0
    rows_after: int = 0
    missing_before: int = 0
    missing_after: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None


@dataclass
class BatchReport:
    """Complete batch processing report."""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    total_rows_processed: int = 0
    total_missing_fixed: int = 0
    total_time: float = 0.0
    results: List[BatchResult] = field(default_factory=list)


class BatchProcessor:
    """
    Process multiple files with consistent configuration.

    Usage:
        processor = BatchProcessor(config)
        report = processor.process_directory("data/raw/", "data/cleaned/")
    """

    def __init__(
        self,
        config: PipelineConfig = None,
        output_format: str = "csv",
        output_suffix: str = "_cleaned",
        parallel: bool = False,
        max_workers: int = 4,
        progress_callback: Callable[[int, int, str], None] = None,
    ):
        """
        Initialize batch processor.

        Args:
            config: Pipeline configuration
            output_format: Output file format (csv, parquet, excel)
            output_suffix: Suffix to add to output files
            parallel: Whether to process files in parallel
            max_workers: Number of parallel workers
            progress_callback: Optional callback for progress updates
        """
        self.config = config or PipelineConfig()
        self.output_format = output_format
        self.output_suffix = output_suffix
        self.parallel = parallel
        self.max_workers = max_workers
        self.progress_callback = progress_callback

    def process_directory(
        self,
        input_dir: str,
        output_dir: str = None,
        pattern: str = "*.csv",
        recursive: bool = False,
    ) -> BatchReport:
        """
        Process all matching files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory (defaults to input_dir)
            pattern: Glob pattern for file matching
            recursive: Whether to search recursively

        Returns:
            BatchReport with results
        """
        # Find files
        if recursive:
            search_pattern = os.path.join(input_dir, "**", pattern)
            files = glob.glob(search_pattern, recursive=True)
        else:
            search_pattern = os.path.join(input_dir, pattern)
            files = glob.glob(search_pattern)

        if not files:
            print(f"No files matching '{pattern}' found in {input_dir}")
            return BatchReport()

        # Set output directory
        if output_dir is None:
            output_dir = input_dir
        os.makedirs(output_dir, exist_ok=True)

        return self.process_files(files, output_dir)

    def process_files(
        self,
        files: List[str],
        output_dir: str,
    ) -> BatchReport:
        """
        Process a list of files.

        Args:
            files: List of file paths
            output_dir: Output directory

        Returns:
            BatchReport with results
        """
        report = BatchReport(total_files=len(files))
        start_time = datetime.now()

        print(f"\n{'=' * 60}")
        print(f"BATCH PROCESSING: {len(files)} files")
        print(f"{'=' * 60}\n")

        if self.parallel:
            results = self._process_parallel(files, output_dir)
        else:
            results = self._process_sequential(files, output_dir)

        # Compile report
        for result in results:
            report.results.append(result)
            if result.success:
                report.successful += 1
                report.total_rows_processed += result.rows_after
                report.total_missing_fixed += (result.missing_before - result.missing_after)
            else:
                report.failed += 1

        report.completed_at = datetime.now().isoformat()
        report.total_time = (datetime.now() - start_time).total_seconds()

        self._print_report(report)

        return report

    def _process_sequential(
        self,
        files: List[str],
        output_dir: str,
    ) -> List[BatchResult]:
        """Process files sequentially."""
        results = []

        with ProgressBar(total=len(files), desc="Processing files") as pbar:
            for file_path in files:
                result = self._process_single_file(file_path, output_dir)
                results.append(result)
                pbar.update(1)

                if self.progress_callback:
                    self.progress_callback(
                        len(results),
                        len(files),
                        f"Processed {os.path.basename(file_path)}"
                    )

        return results

    def _process_parallel(
        self,
        files: List[str],
        output_dir: str,
    ) -> List[BatchResult]:
        """Process files in parallel using threads."""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_file, f, output_dir): f
                for f in files
            }

            with ProgressBar(total=len(files), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

                    if self.progress_callback:
                        self.progress_callback(
                            len(results),
                            len(files),
                            f"Processed {os.path.basename(result.file_path)}"
                        )

        return results

    def _process_single_file(
        self,
        file_path: str,
        output_dir: str,
    ) -> BatchResult:
        """Process a single file."""
        import time
        start_time = time.time()

        result = BatchResult(file_path=file_path, success=False)

        try:
            # Load data
            df = self._load_file(file_path)
            result.rows_before = len(df)
            result.missing_before = df.isnull().sum().sum()

            # Process
            preprocessor = AutoPreprocessor(self.config)
            cleaned_df, report = preprocessor.process(df)

            result.rows_after = len(cleaned_df)
            result.missing_after = cleaned_df.isnull().sum().sum()

            # Save output
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(
                output_dir,
                f"{base_name}{self.output_suffix}.{self.output_format}"
            )

            self._save_file(cleaned_df, output_path)
            result.output_path = output_path
            result.success = True

        except Exception as e:
            result.error = str(e)

        result.processing_time = time.time() - start_time
        return result

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load a data file."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif ext == '.parquet':
            return pd.read_parquet(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _save_file(self, df: pd.DataFrame, output_path: str):
        """Save a data file."""
        ext = self.output_format.lower()

        if ext == 'csv':
            df.to_csv(output_path, index=False)
        elif ext == 'parquet':
            df.to_parquet(output_path, index=False)
        elif ext in ['xlsx', 'excel']:
            df.to_excel(output_path, index=False)
        elif ext == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {ext}")

    def _print_report(self, report: BatchReport):
        """Print batch report summary."""
        print(f"\n{'=' * 60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"\n📊 Summary:")
        print(f"   Total files: {report.total_files}")
        print(f"   Successful: {report.successful} ✓")
        print(f"   Failed: {report.failed} ✗")
        print(f"   Total rows processed: {report.total_rows_processed:,}")
        print(f"   Missing values fixed: {report.total_missing_fixed:,}")
        print(f"   Total time: {report.total_time:.1f}s")

        if report.failed > 0:
            print(f"\n❌ Failed files:")
            for result in report.results:
                if not result.success:
                    print(f"   • {result.file_path}: {result.error}")

        print(f"\n{'=' * 60}\n")

    def save_report(self, report: BatchReport, output_path: str):
        """Save batch report to JSON."""
        report_dict = {
            "started_at": report.started_at,
            "completed_at": report.completed_at,
            "summary": {
                "total_files": report.total_files,
                "successful": report.successful,
                "failed": report.failed,
                "total_rows_processed": report.total_rows_processed,
                "total_missing_fixed": report.total_missing_fixed,
                "total_time_seconds": report.total_time,
            },
            "results": [
                {
                    "file": r.file_path,
                    "success": r.success,
                    "output": r.output_path,
                    "rows_before": r.rows_before,
                    "rows_after": r.rows_after,
                    "missing_before": r.missing_before,
                    "missing_after": r.missing_after,
                    "time_seconds": r.processing_time,
                    "error": r.error,
                }
                for r in report.results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)


def batch_process(
    input_path: str,
    output_dir: str = None,
    pattern: str = "*.csv",
    config: PipelineConfig = None,
    output_format: str = "csv",
    parallel: bool = False,
) -> BatchReport:
    """
    Convenience function for batch processing.

    Args:
        input_path: Directory or single file path
        output_dir: Output directory
        pattern: File pattern for directory processing
        config: Pipeline configuration
        output_format: Output format
        parallel: Enable parallel processing

    Returns:
        BatchReport
    """
    processor = BatchProcessor(
        config=config,
        output_format=output_format,
        parallel=parallel,
    )

    if os.path.isdir(input_path):
        return processor.process_directory(input_path, output_dir, pattern)
    elif os.path.isfile(input_path):
        output_dir = output_dir or os.path.dirname(input_path)
        return processor.process_files([input_path], output_dir)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
