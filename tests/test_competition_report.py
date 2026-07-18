from __future__ import annotations

import pytest

pytest.importorskip("reportlab")

from scripts.build_competition_report import Benchmark, _overall_status, html_report


def benchmark(passed: bool) -> Benchmark:
    return Benchmark(
        label="fixture",
        directory="fixture",
        evaluation={"pass": passed, "metrics": {}},
        results=[],
    )


def test_overall_status_preserves_failed_generalization_gate() -> None:
    assert _overall_status([benchmark(True), benchmark(False)]) == (
        "MIXED - GENERALIZATION GATES NOT MET"
    )


def test_html_table_has_one_header_row_and_data_cells() -> None:
    document = html_report(
        "# Report\n\n| Split | Status |\n| --- | --- |\n| Holdout | FAIL |\n"
    )
    assert document.count("<th>") == 2
    assert document.count("<td>") == 2
    assert "<td>FAIL</td>" in document
