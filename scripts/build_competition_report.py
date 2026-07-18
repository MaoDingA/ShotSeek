#!/usr/bin/env python3
"""Build the auditable ShotSeek competition evaluation report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SPECS = [
    ("黄金回归", "regression-v1"),
    ("Development v1", "development-v1"),
    ("Holdout v1", "holdout-v1"),
    ("Longform v1", "longform-v1"),
    ("Holdout v2", "holdout-v2-first"),
]


@dataclass(frozen=True)
class Benchmark:
    label: str
    directory: str
    evaluation: dict[str, Any]
    results: list[dict[str, Any]]

    @property
    def metrics(self) -> dict[str, Any]:
        return self.evaluation["metrics"]

    @property
    def status(self) -> str:
        return "PASS" if self.evaluation["pass"] else "FAIL"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def collect(benchmark_root: Path) -> list[Benchmark]:
    benchmarks: list[Benchmark] = []
    for label, directory in BENCHMARK_SPECS:
        source = benchmark_root / directory
        benchmarks.append(
            Benchmark(
                label=label,
                directory=directory,
                evaluation=_load_json(source / "evaluation.json"),
                results=_load_json(source / "results.json"),
            )
        )
    return benchmarks


def verify_frozen_inputs(manifest_path: Path) -> list[str]:
    manifest = _load_json(manifest_path)
    checked: list[str] = []
    for split in manifest["splits"].values():
        if "query_path" in split:
            paths = [split["query_path"]]
            hashes = {split["query_path"]: split["query_file_sha256"]}
        else:
            paths = split.get("query_paths", [])
            hashes = split.get("query_file_sha256", {})
        for relative in paths:
            path = ROOT / relative
            expected = hashes[relative] if isinstance(hashes, dict) else hashes
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(f"frozen benchmark hash mismatch: {relative}")
            checked.append(f"{relative}: {actual}")
    return checked


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _ms(value: float) -> str:
    return f"{value:.1f} ms"


def _misses(benchmark: Benchmark) -> list[str]:
    return [
        f'{item["query_id"]}: {item["text"]}'
        for item in benchmark.results
        if item["positive"] and not item["correct_at_3"]
    ]


def _overall_status(benchmarks: list[Benchmark]) -> str:
    return (
        "PASS"
        if all(item.evaluation["pass"] for item in benchmarks)
        else "MIXED - GENERALIZATION GATES NOT MET"
    )


def markdown_report(
    benchmarks: list[Benchmark],
    checked_hashes: list[str],
    longform_runtime: dict[str, Any],
    holdout_runtime: dict[str, Any],
) -> str:
    lines = [
        "# ShotSeek 参赛评测与运行时报告",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        f"总体状态：**{_overall_status(benchmarks)}**",
        "",
        "ShotSeek 的 Production Runtime 已完成真实 StepFun 长视频闭环，"
        "但独立 Holdout 的泛化门禁尚未全部通过。本报告将稳定性、产品可运行性"
        "与泛化能力分开陈述，不用黄金样片成绩替代真实泛化证据。",
        "",
        "## 关键结论",
        "",
        "- M0-M3 真实链路已闭合：上传、任务、进度、代理、镜头、StepFun 视觉、"
        "StepAudio ASR、场景、索引、检索、证据、播放器与导出。",
        f'- 36:58 长片运行到 READY，用时 {longform_runtime["elapsed_s"]:.1f}s，'
        f'RTF {longform_runtime["elapsed_s"] / (longform_runtime["video"]["duration_ms"] / 1000):.3f}。',
        f'- 独立 146 秒样片运行到 READY，用时 {holdout_runtime["elapsed_s"]:.1f}s，'
        f'RTF {holdout_runtime["elapsed_s"] / (holdout_runtime["video"]["duration_ms"] / 1000):.3f}。',
        "- 黄金回归证明系统稳定；Holdout v1/v2 与 Longform v1 失败证明当前"
        "precision-first rule verifier 的跨素材召回仍是明确风险。",
        "",
        "## Benchmark",
        "",
        "| Split | 状态 | Query | R@1 | R@3 | Verifier P | P95 | 负例误报 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in benchmarks:
        m = item.metrics
        lines.append(
            f'| {item.label} | {item.status} | {m["query_count"]} | '
            f'{_percent(m["recall_at_1"])} | {_percent(m["recall_at_3"])} | '
            f'{_percent(m["verifier_precision"])} | '
            f'{_ms(m["latency_ms"]["p95"])} | '
            f'{m["negative_false_positive_query_count"]} |'
        )
    lines.extend(
        [
            "",
            "## 失败 Case",
            "",
        ]
    )
    for item in benchmarks:
        misses = _misses(item)
        if misses:
            lines.append(f"### {item.label}")
            lines.extend(f"- {entry}" for entry in misses)
            lines.append("")
    lines.extend(
        [
            "## 真实 Runtime",
            "",
            "| 运行 | 状态 | 片长 | Scene | StepFun 视觉 | StepAudio ASR | 重试 |",
            "| --- | --- | ---: | ---: | --- | --- | ---: |",
            (
                f'| Longform v1 | {longform_runtime["status"]} | 36:58.672 | '
                f'{longform_runtime["video"]["scene_count"]} | LIVE | LIVE | '
                f'{longform_runtime["retry_count"]} |'
            ),
            (
                f'| Holdout v2 | {holdout_runtime["status"]} | 02:26.000 | '
                f'{holdout_runtime["video"]["scene_count"]} | LIVE | LIVE | '
                f'{holdout_runtime["retry_count"]} |'
            ),
            "",
            "## 验收判断",
            "",
            "- Production Runtime：PASS",
            "- 证据完整性与负例控制：PASS",
            "- 黄金回归稳定性：PASS",
            "- 独立素材泛化：FAIL",
            "- 长视频 Recall@3 门禁：FAIL",
            "- 当前可参赛演示：PASS，必须如实披露泛化边界",
            "- 当前可宣称通用影视检索已达标：NO",
            "",
            "## 可复现性",
            "",
        ]
    )
    lines.extend(f"- {entry}" for entry in checked_hashes)
    lines.extend(
        [
            "",
            "所有 Benchmark 使用冻结 JSONL、固定 SQLite、rule planner、"
            "rule verifier、Top 3 与确定性回放。运行产物和媒体不提交 Git，"
            "公开仓库保留来源、SHA-256、契约、代码和生成脚本。",
            "",
        ]
    )
    return "\n".join(lines)


def html_report(markdown: str) -> str:
    rows: list[str] = []
    in_list = False
    in_table = False
    table_row = 0
    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("|"):
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            if not in_table:
                rows.append("<table>")
                in_table = True
                table_row = 0
            tag = "th" if table_row == 0 else "td"
            rows.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
            table_row += 1
            continue
        if in_table:
            rows.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                rows.append("<ul>")
                in_list = True
            rows.append(f"<li>{html.escape(line[2:])}</li>")
            continue
        if in_list:
            rows.append("</ul>")
            in_list = False
        if not line:
            continue
        if line.startswith("# "):
            rows.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            rows.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            rows.append(f"<h3>{html.escape(line[4:])}</h3>")
        else:
            rows.append(f"<p>{html.escape(line).replace('**', '')}</p>")
    if in_table:
        rows.append("</table>")
    if in_list:
        rows.append("</ul>")
    body = "\n".join(rows)
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ShotSeek 参赛评测与运行时报告</title>
<style>
body{{font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;max-width:1100px;margin:0 auto;padding:56px;color:#172033;line-height:1.65}}
h1{{font-size:42px;line-height:1.15;margin-bottom:12px}} h2{{margin-top:42px;border-top:1px solid #d9e1ec;padding-top:24px}}
table{{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}} th,td{{padding:10px;border-bottom:1px solid #d9e1ec;text-align:left}} th{{background:#eef5ff}}
code{{background:#eef2f7;padding:2px 5px}} li{{margin:7px 0}}
</style><body>{body}</body></html>"""


def _register_font() -> str:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def _chart(benchmarks: list[Benchmark], font: str) -> Drawing:
    drawing = Drawing(168 * mm, 78 * mm)
    chart = VerticalBarChart()
    chart.x = 12 * mm
    chart.y = 13 * mm
    chart.width = 148 * mm
    chart.height = 54 * mm
    chart.data = [
        [item.metrics["recall_at_1"] * 100 for item in benchmarks],
        [item.metrics["recall_at_3"] * 100 for item in benchmarks],
    ]
    chart.categoryAxis.categoryNames = [
        "Golden", "Dev", "Holdout 1", "Longform", "Holdout 2"
    ]
    chart.categoryAxis.labels.fontName = font
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 100
    chart.valueAxis.valueStep = 20
    chart.valueAxis.labels.fontName = font
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = colors.HexColor("#3D8DFF")
    chart.bars[1].fillColor = colors.HexColor("#6DCBF4")
    chart.barSpacing = 2
    chart.groupSpacing = 8
    drawing.add(chart)
    drawing.add(String(10, 6, "R@1", fontName=font, fontSize=8, fillColor=colors.HexColor("#3D8DFF")))
    drawing.add(String(48, 6, "R@3", fontName=font, fontSize=8, fillColor=colors.HexColor("#278EBB")))
    return drawing


def pdf_report(
    path: Path,
    benchmarks: list[Benchmark],
    checked_hashes: list[str],
    longform_runtime: dict[str, Any],
    holdout_runtime: dict[str, Any],
) -> None:
    font = _register_font()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCN", parent=styles["Title"], fontName=font, fontSize=30, leading=38, textColor=colors.HexColor("#14213D"), spaceAfter=10)
    h1 = ParagraphStyle("H1CN", parent=styles["Heading1"], fontName=font, fontSize=20, leading=28, textColor=colors.HexColor("#14213D"), spaceAfter=12)
    h2 = ParagraphStyle("H2CN", parent=styles["Heading2"], fontName=font, fontSize=13, leading=18, textColor=colors.HexColor("#3D8DFF"), spaceBefore=8, spaceAfter=6)
    body = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName=font, fontSize=9.5, leading=15, textColor=colors.HexColor("#27354A"), spaceAfter=7)
    small = ParagraphStyle("SmallCN", parent=body, fontSize=7.2, leading=11, textColor=colors.HexColor("#56657A"))
    status = ParagraphStyle("StatusCN", parent=body, fontSize=12, leading=18, textColor=colors.HexColor("#B42318"), alignment=TA_CENTER)
    center = ParagraphStyle("CenterCN", parent=body, alignment=TA_CENTER)

    def page(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9E1EC"))
        canvas.line(20 * mm, 282 * mm, 190 * mm, 282 * mm)
        canvas.setFont(font, 7)
        canvas.setFillColor(colors.HexColor("#718096"))
        canvas.drawString(20 * mm, 12 * mm, "ShotSeek - Evidence-Aligned Timeline Retrieval")
        canvas.drawRightString(190 * mm, 12 * mm, str(doc.page))
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="ShotSeek 参赛评测与运行时报告",
        author="ShotSeek",
    )
    doc.addPageTemplates(
        PageTemplate(
            id="main",
            frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")],
            onPage=page,
        )
    )

    story: list[Any] = [
        Spacer(1, 28 * mm),
        Paragraph("ShotSeek", title),
        Paragraph("参赛评测与 Production Runtime 证据报告", h1),
        Spacer(1, 8 * mm),
        Paragraph(_overall_status(benchmarks), status),
        Spacer(1, 10 * mm),
        Paragraph(
            "这不是一份只展示最好数字的宣传页。它把真实可运行性、黄金回归稳定性与独立素材泛化分开报告：Runtime 已闭环，泛化门禁仍未全部通过。",
            center,
        ),
        Spacer(1, 38 * mm),
        Paragraph(f"生成日期  {date.today().isoformat()}", small),
        Paragraph("代码、数据集哈希、SQLite 哈希与逐条失败 Case 均可审计", small),
        PageBreak(),
        Paragraph("1. 结论：产品闭环已成立，通用准确率仍需继续提升", h1),
    ]

    conclusion_data = [
        ["验收面", "状态", "证据"],
        ["Production Runtime", "PASS", "36:58 与 02:26 素材均真实 READY"],
        ["StepFun 接入", "PASS", "视觉与 ASR 产物均标记 LIVE"],
        ["证据与负例", "PASS", "直接证据率达标，负例误报为 0"],
        ["黄金回归", "PASS", "40 条查询 R@1/R@3 = 100%"],
        ["独立泛化", "FAIL", "Holdout v1/v2 未同时达到 R@1/R@3 门槛"],
        ["长视频门禁", "FAIL", "Longform R@1 73.3%，R@3 73.3%"],
    ]
    story.extend([
        Table(conclusion_data, colWidths=[42 * mm, 25 * mm, 101 * mm], repeatRows=1, style=[
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3FF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#14213D")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E1EC")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]),
        Spacer(1, 8 * mm),
        Paragraph("比赛叙事应强调“证据对齐时间线检索”与真实 Runtime，同时主动披露 rule verifier 对跨素材同义表达过于保守。失败结果不是隐藏项，而是下一阶段模型验证器的明确入口。", body),
        PageBreak(),
        Paragraph("2. DGX Spark 上的真实 Production Runtime", h1),
    ])

    runtime_rows = [
        ["运行", "片长", "场景", "耗时", "RTF", "视觉", "ASR", "重试"],
        [
            "Longform v1", "36:58.672",
            str(longform_runtime["video"]["scene_count"]),
            f'{longform_runtime["elapsed_s"]:.1f}s',
            f'{longform_runtime["elapsed_s"] / (longform_runtime["video"]["duration_ms"] / 1000):.3f}',
            "LIVE", "LIVE", str(longform_runtime["retry_count"]),
        ],
        [
            "Holdout v2", "02:26.000",
            str(holdout_runtime["video"]["scene_count"]),
            f'{holdout_runtime["elapsed_s"]:.1f}s',
            f'{holdout_runtime["elapsed_s"] / (holdout_runtime["video"]["duration_ms"] / 1000):.3f}',
            "LIVE", "LIVE", str(holdout_runtime["retry_count"]),
        ],
    ]
    story.extend([
        Paragraph("上传 -> Job -> Progress -> 代理 -> 镜头 -> StepFun -> 时间线 -> Scene -> SQLite -> Search -> Evidence -> Export 已在同一 Runtime 内闭合。", body),
        Table(runtime_rows, colWidths=[31 * mm, 25 * mm, 16 * mm, 20 * mm, 16 * mm, 18 * mm, 18 * mm, 13 * mm], repeatRows=1, style=[
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3FF")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E1EC")),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]),
        Spacer(1, 7 * mm),
        Paragraph("Holdout v2 首次运行还验证了失败恢复能力：AAC padding 令容器比视频流长 48ms，旧探测器误判 CFR。修复后仍以帧数和视频流时长严格校验，并补回归测试；同一媒体随后 0 重试 READY。", body),
        Paragraph("Longform 主要耗时来自真实 StepFun 视觉分析；所有网络产物明确区分 LIVE、CACHED、PARTIAL 与 FAILED，缓存不会冒充实时结果。", body),
        PageBreak(),
        Paragraph("3. Benchmark：稳定性与泛化必须分开看", h1),
        _chart(benchmarks, font),
    ])

    benchmark_rows = [["Split", "状态", "Q", "R@1", "R@3", "P", "P95", "负例"]]
    for item in benchmarks:
        m = item.metrics
        benchmark_rows.append([
            item.label, item.status, str(m["query_count"]),
            _percent(m["recall_at_1"]), _percent(m["recall_at_3"]),
            _percent(m["verifier_precision"]), _ms(m["latency_ms"]["p95"]),
            str(m["negative_false_positive_query_count"]),
        ])
    story.extend([
        Table(benchmark_rows, colWidths=[35 * mm, 17 * mm, 10 * mm, 18 * mm, 18 * mm, 18 * mm, 29 * mm, 14 * mm], repeatRows=1, style=[
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3FF")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E1EC")),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]),
        Spacer(1, 6 * mm),
        Paragraph("门槛：R@1 >= 65%，R@3 >= 80%，证据率 >= 85%，P95 <= 3s，负例误报 = 0。Holdout v2 只返回 2 个结果且全部正确，说明系统当前是 precision-first：不乱答，但会漏掉大量正确场景。", body),
        PageBreak(),
        Paragraph("4. 失败画像：Verifier 过严，而不是搜索链路失控", h1),
    ])

    for item in benchmarks:
        misses = _misses(item)
        if not misses:
            continue
        story.append(Paragraph(f"{item.label} - {len(misses)} 条未命中", h2))
        story.append(
            Paragraph("<br/>".join(html.escape(entry) for entry in misses), small)
        )
    story.extend([
        Spacer(1, 4 * mm),
        Paragraph("Holdout v2 的 llama / animal / goat-like animal 指代变化、复合动作和序数约束被 rule verifier 严格词项交集拒绝。由于 v2 在首次运行前已冻结，本报告不根据结果改写查询、不补接受 Scene，也不把 v2 重新命名为通过。", body),
        Paragraph("下一轮正确做法：把 v2 降级为 Development v2，用它设计跨实体别名和基于模型的 Top-K 证据复核；随后再用完全未见的 Holdout v3 验证。", body),
        PageBreak(),
        Paragraph("5. 可复现证据与提交口径", h1),
        Paragraph("冻结输入校验", h2),
    ])
    story.extend(Paragraph(html.escape(entry), small) for entry in checked_hashes)
    story.extend([
        Spacer(1, 5 * mm),
        Paragraph("可对外宣称", h2),
        Paragraph("ShotSeek 已在 DGX Spark 上跑通真实 StepFun 长视频 Production Runtime；每次命中具备画面、对白或镜头边界证据，并能导出 JSON、SRT、XML 与 CMX3600 EDL。", body),
        Paragraph("不可对外宣称", h2),
        Paragraph("不能把黄金样片 100% Recall 描述为通用影视检索准确率；不能声称所有 Holdout 门禁已通过；不能把缓存结果冒充 LIVE。", body),
        Paragraph("最终判断", h2),
        Paragraph("作品已达到可演示、可评审、可交付的参赛形态。若要把“通用长视频检索达到目标准确率”作为核心结论，仍需完成 Development v2 -> Holdout v3 的独立验证。", body),
    ])
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--longform-runtime", type=Path, default=Path("runs/m3-longform-live-v1/runtime/run_report.json"))
    parser.add_argument("--holdout-runtime", type=Path, default=Path("runs/m3-holdout-v2/runtime/run_report.json"))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args()

    benchmark_root = ROOT / args.benchmark_root
    output_root = ROOT / args.output_root
    reports = output_root / "reports"
    pdfs = output_root / "pdf"
    reports.mkdir(parents=True, exist_ok=True)
    pdfs.mkdir(parents=True, exist_ok=True)

    benchmarks = collect(benchmark_root)
    checked_hashes = verify_frozen_inputs(ROOT / "eval/benchmark/manifest.json")
    longform_runtime = _load_json(ROOT / args.longform_runtime)
    holdout_runtime = _load_json(ROOT / args.holdout_runtime)
    markdown = markdown_report(
        benchmarks, checked_hashes, longform_runtime, holdout_runtime
    )
    (reports / "shotseek-competition-evaluation.md").write_text(
        markdown + "\n", encoding="utf-8"
    )
    (reports / "shotseek-competition-evaluation.html").write_text(
        html_report(markdown), encoding="utf-8"
    )
    pdf_report(
        pdfs / "shotseek-competition-evaluation.pdf",
        benchmarks,
        checked_hashes,
        longform_runtime,
        holdout_runtime,
    )
    print(
        json.dumps(
            {
                "status": _overall_status(benchmarks),
                "markdown": str((reports / "shotseek-competition-evaluation.md").relative_to(ROOT)),
                "html": str((reports / "shotseek-competition-evaluation.html").relative_to(ROOT)),
                "pdf": str((pdfs / "shotseek-competition-evaluation.pdf").relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
