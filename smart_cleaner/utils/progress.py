"""
Progress tracking utilities for Smart Cleaner.
Provides visual progress feedback for long-running operations.
"""

import sys
import time
from typing import Optional, Callable, Iterator, Any
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class ProgressState:
    """Current progress state."""
    current: int = 0
    total: int = 100
    message: str = ""
    start_time: float = 0.0


class ProgressBar:
    """
    Terminal progress bar for CLI operations.

    Usage:
        with ProgressBar(total=100, desc="Processing") as pbar:
            for i in range(100):
                # do work
                pbar.update(1)
    """

    def __init__(
        self,
        total: int = 100,
        desc: str = "",
        width: int = 40,
        fill_char: str = "█",
        empty_char: str = "░",
        show_percentage: bool = True,
        show_time: bool = True,
        disable: bool = False,
    ):
        self.total = total
        self.desc = desc
        self.width = width
        self.fill_char = fill_char
        self.empty_char = empty_char
        self.show_percentage = show_percentage
        self.show_time = show_time
        self.disable = disable

        self.current = 0
        self.start_time = None
        self._last_update = 0

    def __enter__(self):
        self.start_time = time.time()
        self._display()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.current = self.total
        self._display()
        print()  # New line after completion
        return False

    def update(self, n: int = 1, message: str = None):
        """Update progress by n steps."""
        if self.disable:
            return

        self.current = min(self.current + n, self.total)
        if message:
            self.desc = message

        # Throttle updates to avoid flicker
        now = time.time()
        if now - self._last_update > 0.05 or self.current == self.total:
            self._display()
            self._last_update = now

    def set_description(self, desc: str):
        """Update the description."""
        self.desc = desc
        self._display()

    def _display(self):
        """Render the progress bar."""
        if self.disable:
            return

        # Calculate percentage
        pct = (self.current / self.total) * 100 if self.total > 0 else 0

        # Build progress bar
        filled = int(self.width * self.current / self.total) if self.total > 0 else 0
        bar = self.fill_char * filled + self.empty_char * (self.width - filled)

        # Build time string
        time_str = ""
        if self.show_time and self.start_time:
            elapsed = time.time() - self.start_time
            if self.current > 0 and self.current < self.total:
                eta = (elapsed / self.current) * (self.total - self.current)
                time_str = f" [{self._format_time(elapsed)}<{self._format_time(eta)}]"
            else:
                time_str = f" [{self._format_time(elapsed)}]"

        # Build percentage string
        pct_str = f" {pct:5.1f}%" if self.show_percentage else ""

        # Compose final string
        desc_str = f"{self.desc}: " if self.desc else ""
        counter_str = f" {self.current}/{self.total}"

        line = f"\r{desc_str}|{bar}|{pct_str}{counter_str}{time_str}"

        # Write to stderr (so stdout remains clean for piping)
        sys.stderr.write(line)
        sys.stderr.flush()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m{seconds % 60:.0f}s"
        else:
            h = int(seconds / 3600)
            m = int((seconds % 3600) / 60)
            return f"{h}h{m}m"


class StepProgress:
    """
    Step-based progress tracker for multi-phase operations.

    Usage:
        progress = StepProgress([
            "Loading data",
            "Analyzing structure",
            "Getting AI recommendations",
            "Applying transformations",
            "Generating report"
        ])

        with progress:
            # Step 1
            progress.next_step()
            load_data()

            # Step 2
            progress.next_step()
            analyze()
            ...
    """

    def __init__(self, steps: list, show_substeps: bool = True):
        self.steps = steps
        self.total_steps = len(steps)
        self.current_step = 0
        self.show_substeps = show_substeps
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"\n✓ Completed all {self.total_steps} steps in {elapsed:.1f}s")
        return False

    def next_step(self, custom_message: str = None):
        """Move to the next step."""
        self.current_step += 1
        step_name = custom_message or self.steps[self.current_step - 1]
        prefix = f"[{self.current_step}/{self.total_steps}]"
        print(f"\n{prefix} {step_name}...")

    def substep(self, message: str):
        """Show a substep message."""
        if self.show_substeps:
            print(f"    → {message}")

    def info(self, message: str):
        """Show an info message."""
        print(f"    ℹ {message}")

    def success(self, message: str):
        """Show a success message."""
        print(f"    ✓ {message}")

    def warning(self, message: str):
        """Show a warning message."""
        print(f"    ⚠ {message}")


def progress_iterator(
    iterable: Iterator,
    total: int = None,
    desc: str = "",
    disable: bool = False
) -> Iterator:
    """
    Wrap an iterator with a progress bar.

    Usage:
        for item in progress_iterator(items, desc="Processing"):
            process(item)
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = 0

    with ProgressBar(total=total, desc=desc, disable=disable) as pbar:
        for item in iterable:
            yield item
            pbar.update(1)


@contextmanager
def timed_operation(name: str, show_start: bool = True):
    """
    Context manager that times an operation.

    Usage:
        with timed_operation("Data loading"):
            df = pd.read_csv(...)
    """
    if show_start:
        print(f"⏳ {name}...", end=" ", flush=True)

    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"done ({elapsed:.2f}s)")


class CallbackProgress:
    """
    Progress tracker that accepts callbacks for integration with GUIs.

    Usage:
        def update_gui(current, total, message):
            gui.set_progress(current / total)
            gui.set_status(message)

        progress = CallbackProgress(callback=update_gui, total=100)
        progress.update(10, "Loading...")
    """

    def __init__(
        self,
        callback: Callable[[int, int, str], None],
        total: int = 100
    ):
        self.callback = callback
        self.total = total
        self.current = 0

    def update(self, n: int = 1, message: str = ""):
        """Update progress and trigger callback."""
        self.current = min(self.current + n, self.total)
        if self.callback:
            self.callback(self.current, self.total, message)

    def set_total(self, total: int):
        """Update total count."""
        self.total = total

    def reset(self):
        """Reset progress."""
        self.current = 0
