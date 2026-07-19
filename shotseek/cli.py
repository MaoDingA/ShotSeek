"""ShotSeek command-line interface."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Sequence

from shotseek.doctor import DoctorConfig, ShotSeekDoctor, format_terminal


def _package_version() -> str:
    try:
        return importlib.metadata.version("shotseek")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shotseek",
        description="ShotSeek evidence-aligned long-video retrieval tools",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="run read-only deployment diagnostics",
        description=(
            "Run read-only ShotSeek diagnostics. Default mode is offline and never "
            "starts, stops, kills, repairs, or downloads anything."
        ),
    )
    doctor.add_argument("--project-root", type=Path, default=Path.cwd())
    doctor.add_argument("--runtime-root", type=Path)
    doctor.add_argument(
        "--runtime-port",
        type=int,
        default=_env_int("SHOTSEEK_PORT", 8000),
    )
    doctor.add_argument(
        "--frontend-port",
        type=int,
        default=_env_int("SHOTSEEK_FRONTEND_PORT", 5173),
    )
    doctor.add_argument(
        "--debug-port",
        dest="debug_ports",
        type=int,
        action="append",
        default=[],
        help="optional debug port to inspect; may be repeated",
    )
    doctor.add_argument(
        "--disk-warn-gb",
        type=float,
        default=_env_float("SHOTSEEK_DOCTOR_DISK_WARN_GB", 20.0),
    )
    doctor.add_argument(
        "--disk-fail-gb",
        type=float,
        default=_env_float("SHOTSEEK_DOCTOR_DISK_FAIL_GB", 5.0),
    )
    doctor.add_argument(
        "--timeout-seconds",
        type=float,
        default=_env_float("SHOTSEEK_DOCTOR_TIMEOUT_SECONDS", 10.0),
    )
    doctor.add_argument("--verbose", action="store_true")
    doctor.add_argument("--json", dest="json_output", action="store_true")
    doctor.add_argument(
        "--deep",
        action="store_true",
        help="run one project-local 1-second NVENC encode and remove it",
    )
    doctor.add_argument(
        "--live",
        action="store_true",
        help="make one low-cost StepFun text request; never uploads media or starts ASR",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "doctor":
        parser.error(f"unknown command: {args.command}")
    try:
        config = DoctorConfig(
            project_root=args.project_root,
            runtime_root=args.runtime_root,
            runtime_port=args.runtime_port,
            frontend_port=args.frontend_port,
            debug_ports=tuple(args.debug_ports),
            disk_warn_gb=args.disk_warn_gb,
            disk_fail_gb=args.disk_fail_gb,
            timeout_s=args.timeout_seconds,
            deep=args.deep,
            live=args.live,
        )
    except ValueError as error:
        parser.error(str(error))
    report = ShotSeekDoctor(config).run()
    if args.json_output:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_terminal(report, verbose=args.verbose))
    return 1 if report.status == "fail" else 0


def main() -> None:
    raise SystemExit(run())
