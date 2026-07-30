"""Binary spreadsheet ingestion: ``.xlsx`` / ``.xls`` → tabular text.

The compression pipeline is text-only, so binary spreadsheets enter through this
adapter at the SDK boundary. Each sheet is rendered to CSV text, which then flows
through the normal tabular detection → SmartCrusher path like any other table.

Parsers are optional dependencies (``pip install headroom-ai[spreadsheet]``) and
are imported lazily; a missing dependency fails loudly with an actionable
message rather than silently degrading.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

__all__ = ["load_spreadsheet"]


def _is_blank(cell: object) -> bool:
    """True for cells that carry no data (``None`` or whitespace-only text).

    ``0`` and ``False`` are real values, so only ``None`` and blank strings
    count — an ``is None`` check alone would keep ``""`` padding cells.
    """
    if cell is None:
        return True
    return isinstance(cell, str) and not cell.strip()


def _bounding_box(rows: list[list[object]]) -> tuple[int, int]:
    """Return ``(n_rows, n_cols)`` of the populated region of ``rows``.

    openpyxl's read-only mode trusts the sheet's declared dimensions, which
    editors routinely over-report — a 12×4 table can arrive as 800×26, and every
    phantom row renders as a ``,,,,`` line. Trailing empty rows *and* trailing
    empty columns are trimmed so the CSV covers only real data.
    """
    last_row = 0
    last_col = 0
    for r_idx, row in enumerate(rows, start=1):
        row_last_col = 0
        for c_idx, cell in enumerate(row, start=1):
            if not _is_blank(cell):
                row_last_col = c_idx
        if row_last_col:
            last_row = r_idx
            last_col = max(last_col, row_last_col)
    return last_row, last_col


def _rows_to_csv(rows: list[list[object]]) -> str:
    """Render rows to CSV text, dropping fully empty trailing rows and columns.

    The writer is pinned to ``\\n`` line endings: csv's default ``excel``
    dialect emits ``\\r\\n``, which costs a token per row and leaves a dangling
    ``\\r`` on the final line after stripping.
    """
    n_rows, n_cols = _bounding_box(rows)
    if not n_rows or not n_cols:
        return ""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in rows[:n_rows]:
        trimmed = list(row[:n_cols])
        if len(trimmed) < n_cols:
            trimmed.extend([None] * (n_cols - len(trimmed)))
        writer.writerow(["" if cell is None else cell for cell in trimmed])
    return buf.getvalue().strip("\n")


def _load_xlsx(path: Path) -> dict[str, str]:
    try:
        import openpyxl
    except ImportError as e:  # pragma: no cover - openpyxl ships in [dev]; defensive guard
        raise ImportError(
            "Reading .xlsx files requires openpyxl. "
            "Install it with: pip install headroom-ai[spreadsheet]"
        ) from e

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets: dict[str, str] = {}
    try:
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            text = _rows_to_csv(rows)
            if text.strip():
                sheets[ws.title] = text
    finally:
        wb.close()
    return sheets


def _load_xls(
    path: Path,
) -> dict[str, str]:  # pragma: no cover - legacy .xls; needs optional xlrd + binary fixture
    try:
        import xlrd
    except ImportError as e:
        raise ImportError(
            "Reading legacy .xls files requires xlrd. "
            "Install it with: pip install headroom-ai[spreadsheet]"
        ) from e

    book = xlrd.open_workbook(str(path))
    sheets: dict[str, str] = {}
    for sheet in book.sheets():
        rows = [sheet.row_values(i) for i in range(sheet.nrows)]
        text = _rows_to_csv(rows)
        if text.strip():
            sheets[sheet.name] = text
    return sheets


def load_spreadsheet(path: str | Path) -> dict[str, str]:
    """Load a spreadsheet file into ``{sheet_name: csv_text}``.

    Args:
        path: Path to a ``.xlsx`` or ``.xls`` file.

    Returns:
        Mapping of sheet name to CSV-rendered text (empty sheets omitted).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file extension is unsupported.
        ImportError: If the required parser dependency is not installed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Spreadsheet not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".xlsx":
        return _load_xlsx(p)
    if suffix == ".xls":
        return _load_xls(p)  # pragma: no cover - legacy .xls path, see _load_xls
    raise ValueError(f"Unsupported spreadsheet format '{suffix}'. Supported: .xlsx, .xls")
