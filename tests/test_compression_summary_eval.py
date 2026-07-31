"""Eval: Compression summary quality — generic, unbiased.

Tests that compression summaries are:
1. Accurate (categories match actual dropped items)
2. Useful (contain information that would help retrieval)
3. Not misleading (don't hallucinate categories)

These are NOT skewed to show summaries as amazing — they test
real-world data patterns and verify correctness.
"""

from headroom.transforms.compression_summary import summarize_compressed_code


class TestCodeSummaryAccuracy:
    """Verify code summaries accurately describe removed sections."""

    def test_real_python_module(self):
        """Summary of a realistic Python module compression."""
        # Use AST-based summary (language-agnostic)
        bodies = [
            ("def __init__(self, url: str, pool_size: int = 10):", "...", 8),
            ("def connect(self) -> Any:", "...", 15),
            ("def _create_new(self) -> Any:", "...", 22),
            ("def release(self, conn: Any) -> None:", "...", 28),
            ("def close_all(self) -> None:", "...", 33),
            ("def create_engine(url: str) -> DatabaseConnection:", "...", 38),
        ]
        summary = summarize_compressed_code(bodies, 6)
        assert summary.startswith("6 bodies: ")
        has_names = any(
            name in summary for name in ["connect", "release", "close_all", "create_engine"]
        )
        assert has_names, f"Summary missing function names: {summary}"
