"""Tests for src/core/logging_setup.py."""
import logging
import tempfile
import unittest
from pathlib import Path

from src.core.logging_setup import configure_run_logging


class TestConfigureRunLogging(unittest.TestCase):
    def tearDown(self) -> None:
        """Close + remove any FileHandlers added during the test to avoid handler accumulation and unclosed file handles."""
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        for h in file_handlers:
            h.close()
        root.handlers = [h for h in root.handlers if not isinstance(h, logging.FileHandler)]

    def test_creates_nested_directory(self) -> None:
        """Log file is created inside a nested directory that does not yet exist."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "a" / "b" / "c" / "run.log"
            configure_run_logging(log_path)
            self.assertTrue(log_path.parent.exists(), "Parent directories were not created")
            self.assertTrue(log_path.exists(), "Log file was not created")

    def test_debug_message_written_to_file(self) -> None:
        """A DEBUG message sent to the root logger appears in the log file."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "sub" / "run.log"
            root = configure_run_logging(log_path)
            root.setLevel(logging.DEBUG)
            root.debug("unique-marker-abc123")
            for h in root.handlers:
                h.flush()
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("unique-marker-abc123", content)

    def test_calling_twice_adds_second_handler(self) -> None:
        """Calling configure_run_logging twice with different paths adds a second handler without crashing."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path_1 = Path(tmp) / "run1.log"
            log_path_2 = Path(tmp) / "run2.log"
            configure_run_logging(log_path_1)
            configure_run_logging(log_path_2)
            root = logging.getLogger()
            file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
            self.assertGreaterEqual(len(file_handlers), 2)
            self.assertTrue(log_path_1.exists())
            self.assertTrue(log_path_2.exists())

    def test_log_format_contains_timestamp_level_and_message(self) -> None:
        """Log lines include a timestamp, the level name, and the message text."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "fmt_test" / "run.log"
            root = configure_run_logging(log_path)
            root.setLevel(logging.DEBUG)
            root.warning("format-check-xyz")
            for h in root.handlers:
                h.flush()
            content = log_path.read_text(encoding="utf-8")
            # Timestamp pattern: starts with a digit (e.g. "2026-")
            self.assertTrue(
                any(line and line[0].isdigit() for line in content.splitlines()),
                "No timestamp found in log output",
            )
            self.assertIn("WARNING", content)
            self.assertIn("format-check-xyz", content)


if __name__ == "__main__":
    unittest.main()
