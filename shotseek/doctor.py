"""Read-only deployment diagnostics for ShotSeek.

The default doctor never reaches the public internet and never creates, removes,
starts, stops, or repairs anything.  ``--deep`` opts into one bounded NVENC
encode inside the project-local temporary directory.  ``--live`` opts into one
small StepFun text request; it never uploads media or starts ASR.
"""

from __future__ import annotations

import errno
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Mapping, Sequence


DOCTOR_SCHEMA_VERSION = "shotseek-doctor-v1"
DEFAULT_CHAT_BASE_URL = "https://api.stepfun.com/step_plan/v1"
DEFAULT_STEPFUN_MODEL = "step-3.7-flash"
TERMINAL_JOB_STATES = frozenset({"READY", "PARTIAL", "FAILED", "CANCELLED"})
KNOWN_JOB_STATES = frozenset(
    {
        "CREATED",
        "QUEUED",
        "PROBING",
        "TRANSCODING",
        "EXTRACTING_AUDIO",
        "DETECTING_SHOTS",
        "CHUNKING",
        "ANALYZING_VISUAL",
        "ANALYZING_ASR",
        "ALIGNING",
        "BUILDING_SCENES",
        "INDEXING",
        "RETRYING",
        *TERMINAL_JOB_STATES,
    }
)
REQUIRED_REGISTRY_TABLES = frozenset({"video", "job", "job_event", "artifact"})
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(authorization|api[_-]?key|token|secret|password)"
    r"(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}")
REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
LISTENER_RE = re.compile(r'users:\(\(\"(?P<process>[^\"]+)\",pid=(?P<pid>\d+)')


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class DoctorCheck:
    check_id: str
    status: CheckStatus
    message: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", self.check_id):
            raise ValueError(f"invalid doctor check_id: {self.check_id!r}")
        if not self.message.strip():
            raise ValueError("doctor check message cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class DoctorReport:
    status: str
    checks: tuple[DoctorCheck, ...]
    summary: Mapping[str, int]
    project_root: str
    deep: bool
    live: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": DOCTOR_SCHEMA_VERSION,
            "status": self.status,
            "project_root": self.project_root,
            "mode": {"deep": self.deep, "live": self.live},
            "checks": [check.to_dict() for check in self.checks],
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class DoctorConfig:
    project_root: Path
    runtime_root: Path | None = None
    runtime_port: int = 8000
    frontend_port: int = 5173
    debug_ports: tuple[int, ...] = ()
    disk_warn_gb: float = 20.0
    disk_fail_gb: float = 5.0
    timeout_s: float = 10.0
    deep: bool = False
    live: bool = False

    def __post_init__(self) -> None:
        root = self.project_root.resolve()
        runtime = (self.runtime_root or root / "data" / "runtime").resolve()
        if not runtime.is_relative_to(root):
            raise ValueError("runtime_root must stay inside project_root")
        for port in (self.runtime_port, self.frontend_port, *self.debug_ports):
            if not 1 <= port <= 65535:
                raise ValueError(f"invalid TCP port: {port}")
        if self.disk_fail_gb < 0 or self.disk_warn_gb < 0:
            raise ValueError("disk thresholds must be non-negative")
        if self.disk_fail_gb >= self.disk_warn_gb:
            raise ValueError("disk_fail_gb must be lower than disk_warn_gb")
        if not 0 < self.timeout_s <= 60:
            raise ValueError("timeout_s must be in (0, 60]")
        object.__setattr__(self, "project_root", root)
        object.__setattr__(self, "runtime_root", runtime)
        object.__setattr__(self, "debug_ports", tuple(dict.fromkeys(self.debug_ports)))


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class HttpResult:
    status_code: int
    body: bytes


@dataclass(frozen=True)
class LiveProbeResult:
    status_code: int
    latency_ms: float
    response_valid: bool


@dataclass(frozen=True)
class PortInspection:
    available: bool
    pid: int | None = None
    process: str | None = None
    error: str | None = None


CommandRunner = Callable[[Sequence[str], float], CommandResult]
BinaryFinder = Callable[[str], str | None]
LocalHttpGetter = Callable[[str, float], HttpResult]
StepFunProbe = Callable[[str, str, str, float], LiveProbeResult]
PortInspector = Callable[[str, int], PortInspection]


def _safe_command_environment() -> dict[str, str]:
    """Pass only runtime essentials to diagnostic subprocesses, never secrets."""
    allowed = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "TMPDIR",
        "XDG_RUNTIME_DIR",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def run_command(argv: Sequence[str], timeout_s: float) -> CommandResult:
    completed = subprocess.run(
        list(argv),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env=_safe_command_environment(),
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def local_http_get(url: str, timeout_s: float) -> HttpResult:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("default doctor HTTP checks are restricted to localhost")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"User-Agent": "ShotSeek-Doctor/1"})
    with opener.open(request, timeout=timeout_s) as response:
        return HttpResult(int(response.status), response.read(1_048_576))


def stepfun_live_probe(
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: float,
) -> LiveProbeResult:
    parsed = urllib.parse.urlsplit(base_url)
    loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme not in ({"https"} if not loopback else {"http", "https"}):
        raise ValueError("StepFun endpoint must use HTTPS unless it is loopback")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("StepFun endpoint must not contain credentials, query, or fragment")
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply exactly: OK"}],
            "stream": False,
            "reasoning_effort": "low",
            "max_tokens": 8,
            "temperature": 0,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ShotSeek-Doctor/1",
        },
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read(65_536)
        status_code = int(response.status)
    latency_ms = (time.perf_counter() - started) * 1000
    try:
        decoded = json.loads(body)
        valid = (
            isinstance(decoded, dict)
            and isinstance(decoded.get("choices"), list)
            and bool(decoded["choices"])
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        valid = False
    return LiveProbeResult(status_code, latency_ms, valid)


def _redact(value: object, secrets: Sequence[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "<redacted>")
    text = SECRET_ASSIGNMENT_RE.sub(r"\1\2<redacted>", text)
    return BEARER_RE.sub("Bearer <redacted>", text)


def _version_line(output: str) -> str:
    return next((line.strip() for line in output.splitlines() if line.strip()), "unknown")


def _nearest_existing(path: Path) -> Path | None:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() else None


def _sqlite_uri(path: Path) -> str:
    quoted = urllib.parse.quote(str(path.resolve()), safe="/")
    return f"file:{quoted}?mode=ro&immutable=1"


def _read_meminfo() -> tuple[int, int] | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        match = re.search(r"(\d+)", rest)
        if match:
            values[key] = int(match.group(1)) * 1024
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None
    return total, available


def inspect_port(host: str, port: int) -> PortInspection:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            return PortInspection(available=False)
        return PortInspection(available=False, error=f"{type(error).__name__}: {error}")
    finally:
        sock.close()
    return PortInspection(available=True)


class _AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        key = "src" if tag in {"script", "img"} else "href" if tag == "link" else None
        if key and values.get(key):
            self.references.append(str(values[key]))


class ShotSeekDoctor:
    def __init__(
        self,
        config: DoctorConfig,
        *,
        environ: Mapping[str, str] | None = None,
        runner: CommandRunner = run_command,
        which: BinaryFinder = shutil.which,
        local_get: LocalHttpGetter = local_http_get,
        live_probe: StepFunProbe = stepfun_live_probe,
        port_inspector: PortInspector = inspect_port,
    ) -> None:
        self.config = config
        self.environ = dict(os.environ if environ is None else environ)
        self.runner = runner
        self.which = which
        self.local_get = local_get
        self.live_probe = live_probe
        self.port_inspector = port_inspector
        self._checks: list[DoctorCheck] = []
        self._check_ids: set[str] = set()
        self._gpu_rows: list[dict[str, object]] = []
        self._runtime_is_healthy = False
        self._runtime_health_payload: dict[str, object] = {}
        self._api_key: str | None = None

    @property
    def project_root(self) -> Path:
        return self.config.project_root

    @property
    def runtime_root(self) -> Path:
        assert self.config.runtime_root is not None
        return self.config.runtime_root

    def _record(self, check_id: str, callback: Callable[[], DoctorCheck]) -> None:
        if check_id in self._check_ids:
            raise RuntimeError(f"duplicate doctor check_id: {check_id}")
        self._check_ids.add(check_id)
        secrets = tuple(
            value
            for key, value in self.environ.items()
            if value and any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        )
        try:
            check = callback()
            if check.check_id != check_id:
                raise ValueError(
                    f"check returned {check.check_id!r}; expected {check_id!r}"
                )
        except Exception as error:  # one broken probe must not crash the doctor
            check = DoctorCheck(
                check_id,
                CheckStatus.FAIL,
                f"Check raised {type(error).__name__}",
                {"error": _redact(error, secrets)},
            )
        self._checks.append(check)

    def _check_project(self) -> DoctorCheck:
        pyproject = self.project_root / "pyproject.toml"
        if not pyproject.is_file():
            return DoctorCheck(
                "project.metadata",
                CheckStatus.FAIL,
                "pyproject.toml is missing; use --project-root",
            )
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            name = str(data.get("project", {}).get("name", ""))
        except (OSError, tomllib.TOMLDecodeError) as error:
            return DoctorCheck(
                "project.metadata",
                CheckStatus.FAIL,
                "pyproject.toml is unreadable",
                {"error": _redact(error)},
            )
        if name != "shotseek":
            return DoctorCheck(
                "project.metadata",
                CheckStatus.FAIL,
                f"Unexpected project name: {name or '<empty>'}",
            )
        return DoctorCheck("project.metadata", CheckStatus.PASS, "ShotSeek project detected")

    def _check_python(self) -> DoctorCheck:
        version = platform.python_version()
        status = CheckStatus.PASS if sys.version_info >= (3, 11) else CheckStatus.FAIL
        return DoctorCheck(
            "python.version",
            status,
            f"Python {version}" + (" detected" if status == CheckStatus.PASS else " is below 3.11"),
            {"implementation": platform.python_implementation()},
        )

    def _check_binary(self, name: str, *version_args: str, required: bool = True) -> DoctorCheck:
        check_id = f"binary.{name}"
        executable = self.which(name)
        if not executable:
            return DoctorCheck(
                check_id,
                CheckStatus.FAIL if required else CheckStatus.WARN,
                f"{name} is not available",
            )
        result = self.runner([executable, *version_args], min(self.config.timeout_s, 10))
        output = result.stdout or result.stderr
        if result.returncode != 0:
            return DoctorCheck(
                check_id,
                CheckStatus.FAIL if required else CheckStatus.WARN,
                f"{name} version probe failed",
                {"returncode": result.returncode},
            )
        return DoctorCheck(
            check_id,
            CheckStatus.PASS,
            _version_line(output),
            {"executable": executable},
        )

    def _check_dependencies(self) -> DoctorCheck:
        pyproject = self.project_root / "pyproject.toml"
        if not pyproject.is_file():
            return DoctorCheck(
                "python.dependencies", CheckStatus.SKIP, "Project metadata unavailable"
            )
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        requirements = list(data.get("project", {}).get("dependencies", []))
        installed: dict[str, str] = {}
        missing: list[str] = []
        for requirement in requirements:
            match = REQUIREMENT_NAME_RE.match(str(requirement))
            if not match:
                continue
            name = match.group(1)
            try:
                installed[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(name)
        if missing:
            return DoctorCheck(
                "python.dependencies",
                CheckStatus.FAIL,
                f"Missing Python dependencies: {', '.join(sorted(missing))}",
                {"installed": installed, "missing": sorted(missing)},
            )
        return DoctorCheck(
            "python.dependencies",
            CheckStatus.PASS,
            f"{len(installed)} required Python dependencies installed",
            {"installed": installed},
        )

    def _check_chromium(self) -> DoctorCheck:
        candidates = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")
        found = next(((name, self.which(name)) for name in candidates if self.which(name)), None)
        if found is None:
            return DoctorCheck(
                "browser.chromium",
                CheckStatus.WARN,
                "Chromium/Chrome not found; browser E2E is unavailable",
            )
        name, executable = found
        return DoctorCheck(
            "browser.chromium",
            CheckStatus.PASS,
            f"{name} available",
            {"executable": executable},
        )

    def _check_playwright(self) -> DoctorCheck:
        python_available = importlib.util.find_spec("playwright") is not None
        node_paths = (
            self.project_root / "apps" / "web" / "node_modules" / "playwright",
            self.project_root / "apps" / "web" / "node_modules" / "@playwright" / "test",
        )
        node_available = any(path.exists() for path in node_paths)
        if not python_available and not node_available:
            return DoctorCheck(
                "browser.playwright",
                CheckStatus.WARN,
                "Playwright is not installed; browser E2E is unavailable",
            )
        sources = [
            source
            for source, available in (("python", python_available), ("node", node_available))
            if available
        ]
        return DoctorCheck(
            "browser.playwright",
            CheckStatus.PASS,
            f"Playwright available ({', '.join(sources)})",
        )

    def _check_nvidia_smi(self) -> DoctorCheck:
        if platform.system() == "Darwin":
            return DoctorCheck(
                "nvidia.smi", CheckStatus.SKIP, "NVIDIA checks are not applicable on macOS"
            )
        executable = self.which("nvidia-smi")
        if not executable:
            return DoctorCheck(
                "nvidia.smi", CheckStatus.FAIL, "nvidia-smi is not available on Linux"
            )
        query = "name,driver_version,memory.total,memory.free"
        result = self.runner(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            min(self.config.timeout_s, 10),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return DoctorCheck(
                "nvidia.smi",
                CheckStatus.FAIL,
                "nvidia-smi could not query the GPU",
                {"returncode": result.returncode},
            )
        rows: list[dict[str, object]] = []
        for raw in result.stdout.splitlines():
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) < 4:
                continue
            total = float(parts[2]) if re.fullmatch(r"\d+(?:\.\d+)?", parts[2]) else None
            free = float(parts[3]) if re.fullmatch(r"\d+(?:\.\d+)?", parts[3]) else None
            rows.append(
                {
                    "name": parts[0],
                    "driver": parts[1],
                    "memory_total_mib": total,
                    "memory_free_mib": free,
                }
            )
        if not rows:
            return DoctorCheck(
                "nvidia.smi", CheckStatus.FAIL, "nvidia-smi returned an unknown format"
            )
        self._gpu_rows = rows
        first = rows[0]
        return DoctorCheck(
            "nvidia.smi",
            CheckStatus.PASS,
            f"NVIDIA GPU detected: {first['name']}",
            {"gpus": rows},
        )

    def _check_cuda_visibility(self) -> DoctorCheck:
        if platform.system() == "Darwin":
            return DoctorCheck(
                "nvidia.cuda_visibility", CheckStatus.SKIP, "CUDA is not applicable on macOS"
            )
        if not self._gpu_rows:
            return DoctorCheck(
                "nvidia.cuda_visibility", CheckStatus.SKIP, "GPU detection did not pass"
            )
        if "CUDA_VISIBLE_DEVICES" not in self.environ:
            return DoctorCheck(
                "nvidia.cuda_visibility",
                CheckStatus.PASS,
                "CUDA visibility is not restricted by the process environment",
            )
        value = self.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if value in {"", "-1", "none", "None"}:
            return DoctorCheck(
                "nvidia.cuda_visibility",
                CheckStatus.FAIL,
                "CUDA_VISIBLE_DEVICES hides all GPUs (value redacted)",
            )
        return DoctorCheck(
            "nvidia.cuda_visibility",
            CheckStatus.PASS,
            "CUDA visibility is explicitly configured (value redacted)",
        )

    def _ffmpeg_listing(self, flag: str) -> tuple[str | None, CommandResult | None]:
        executable = self.which("ffmpeg")
        if not executable:
            return None, None
        return executable, self.runner(
            [executable, "-hide_banner", flag], min(self.config.timeout_s, 10)
        )

    def _check_nvdec(self) -> DoctorCheck:
        if platform.system() == "Darwin":
            return DoctorCheck("nvidia.nvdec", CheckStatus.SKIP, "NVDEC is not applicable on macOS")
        executable, decoders = self._ffmpeg_listing("-decoders")
        if executable is None or decoders is None:
            return DoctorCheck("nvidia.nvdec", CheckStatus.SKIP, "FFmpeg is unavailable")
        hwaccels = self.runner(
            [executable, "-hide_banner", "-hwaccels"], min(self.config.timeout_s, 10)
        )
        text = f"{decoders.stdout}\n{decoders.stderr}"
        codecs = sorted(set(re.findall(r"\b([a-z0-9]+_cuvid)\b", text)))
        cuda_listed = "cuda" in hwaccels.stdout.lower()
        if decoders.returncode == 0 and hwaccels.returncode == 0 and codecs and cuda_listed:
            return DoctorCheck(
                "nvidia.nvdec",
                CheckStatus.PASS,
                f"NVDEC capabilities enumerated ({len(codecs)} CUVID decoders)",
                {"decoders": codecs, "cuda_hwaccel": True},
            )
        return DoctorCheck(
            "nvidia.nvdec",
            CheckStatus.WARN,
            "FFmpeg did not enumerate CUDA/NVDEC capabilities",
            {"decoders": codecs, "cuda_hwaccel": cuda_listed},
        )

    def _check_nvenc(self) -> DoctorCheck:
        if platform.system() == "Darwin":
            return DoctorCheck("nvidia.nvenc", CheckStatus.SKIP, "NVENC is not applicable on macOS")
        _, encoders = self._ffmpeg_listing("-encoders")
        if encoders is None:
            return DoctorCheck("nvidia.nvenc", CheckStatus.SKIP, "FFmpeg is unavailable")
        text = f"{encoders.stdout}\n{encoders.stderr}"
        names = sorted(set(re.findall(r"\b([a-z0-9]+_nvenc)\b", text)))
        if encoders.returncode == 0 and names:
            return DoctorCheck(
                "nvidia.nvenc",
                CheckStatus.PASS,
                f"NVENC encoders enumerated: {', '.join(names)}",
                {"encoders": names, "actual_encode_tested": False},
            )
        return DoctorCheck(
            "nvidia.nvenc",
            CheckStatus.WARN,
            "FFmpeg did not enumerate an NVENC encoder",
            {"encoders": names, "actual_encode_tested": False},
        )

    def _check_unified_memory(self) -> DoctorCheck:
        if platform.system() == "Darwin":
            return DoctorCheck(
                "nvidia.unified_memory", CheckStatus.SKIP, "DGX Spark memory check is not applicable on macOS"
            )
        if not self._gpu_rows:
            return DoctorCheck(
                "nvidia.unified_memory", CheckStatus.SKIP, "GPU detection did not pass"
            )
        is_spark = any(
            "GB10" in str(row.get("name", "")).upper()
            or "DGX SPARK" in str(row.get("name", "")).upper()
            for row in self._gpu_rows
        )
        if not is_spark:
            return DoctorCheck(
                "nvidia.unified_memory",
                CheckStatus.SKIP,
                "GPU detected, but DGX Spark unified memory was not identified",
            )
        memory = _read_meminfo()
        if memory is None:
            return DoctorCheck(
                "nvidia.unified_memory",
                CheckStatus.WARN,
                "DGX Spark detected but system memory could not be read",
            )
        total, available = memory
        return DoctorCheck(
            "nvidia.unified_memory",
            CheckStatus.PASS,
            f"Unified memory available: {available / 1024**3:.1f} GiB / {total / 1024**3:.1f} GiB",
            {"total_bytes": total, "available_bytes": available},
        )

    def _check_nvenc_deep(self) -> DoctorCheck:
        if not self.config.deep:
            return DoctorCheck(
                "nvidia.nvenc_probe",
                CheckStatus.SKIP,
                "Actual NVENC encode not run; use --deep",
            )
        if platform.system() == "Darwin":
            return DoctorCheck(
                "nvidia.nvenc_probe", CheckStatus.SKIP, "NVENC is not applicable on macOS"
            )
        executable = self.which("ffmpeg")
        if not executable:
            return DoctorCheck(
                "nvidia.nvenc_probe", CheckStatus.FAIL, "FFmpeg is unavailable"
            )
        temporary_root = (self.project_root / "tmp").resolve()
        if not temporary_root.is_relative_to(self.project_root):
            return DoctorCheck(
                "nvidia.nvenc_probe", CheckStatus.FAIL, "Temporary path escaped project root"
            )
        root_created = False
        if not temporary_root.exists():
            temporary_root.mkdir(parents=True)
            root_created = True
        try:
            with tempfile.TemporaryDirectory(prefix="doctor-nvenc-", dir=temporary_root) as raw:
                output = Path(raw) / "probe.mp4"
                result = self.runner(
                    [
                        executable,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=black:s=640x360:r=30:d=1",
                        "-c:v",
                        "h264_nvenc",
                        "-preset",
                        "p1",
                        "-t",
                        "1",
                        str(output),
                    ],
                    max(self.config.timeout_s, 20),
                )
                if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
                    return DoctorCheck(
                        "nvidia.nvenc_probe",
                        CheckStatus.FAIL,
                        "One-second NVENC synthetic encode failed",
                        {
                            "returncode": result.returncode,
                            "error": _redact(_version_line(result.stderr)),
                        },
                    )
                size = output.stat().st_size
                return DoctorCheck(
                    "nvidia.nvenc_probe",
                    CheckStatus.PASS,
                    "One-second NVENC synthetic encode succeeded",
                    {"bytes": size, "temporary_output_removed": True},
                )
        finally:
            if root_created:
                try:
                    temporary_root.rmdir()
                except OSError:
                    pass

    def _check_writable(self, check_id: str, path: Path, label: str) -> DoctorCheck:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.project_root):
            return DoctorCheck(check_id, CheckStatus.FAIL, f"{label} escapes project root")
        existing = _nearest_existing(resolved)
        if existing is None:
            return DoctorCheck(check_id, CheckStatus.FAIL, f"{label} has no existing parent")
        if resolved.exists() and not resolved.is_dir():
            return DoctorCheck(check_id, CheckStatus.FAIL, f"{label} is not a directory")
        writable = os.access(existing, os.W_OK | os.X_OK)
        return DoctorCheck(
            check_id,
            CheckStatus.PASS if writable else CheckStatus.FAIL,
            f"{label} is {'writable' if writable else 'not writable'}",
            {"path": str(resolved), "checked_existing_parent": str(existing)},
        )

    def _check_disk(self) -> DoctorCheck:
        usage = shutil.disk_usage(self.project_root)
        free_gb = usage.free / 1024**3
        if free_gb < self.config.disk_fail_gb:
            status = CheckStatus.FAIL
        elif free_gb < self.config.disk_warn_gb:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.PASS
        return DoctorCheck(
            "storage.free_disk",
            status,
            f"Free disk: {free_gb:.1f} GiB",
            {
                "free_bytes": usage.free,
                "warn_below_gb": self.config.disk_warn_gb,
                "fail_below_gb": self.config.disk_fail_gb,
            },
        )

    def _listener_details(self, port: int) -> tuple[int | None, str | None]:
        ss = self.which("ss")
        if ss:
            result = self.runner(
                [ss, "-ltnpH", f"sport = :{port}"], min(self.config.timeout_s, 5)
            )
            match = LISTENER_RE.search(result.stdout)
            if match:
                return int(match.group("pid")), match.group("process")
        lsof = self.which("lsof")
        if lsof:
            result = self.runner(
                [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-FpPc"],
                min(self.config.timeout_s, 5),
            )
            pid_match = re.search(r"(?m)^p(\d+)$", result.stdout)
            process_match = re.search(r"(?m)^c(.+)$", result.stdout)
            if pid_match:
                return int(pid_match.group(1)), process_match.group(1) if process_match else None
        return None, None

    def _check_port(self, role: str, port: int) -> DoctorCheck:
        check_id = f"port.{role}.{port}"
        inspection = self.port_inspector("127.0.0.1", port)
        if inspection.available:
            return DoctorCheck(check_id, CheckStatus.PASS, f"Port {port} is available")
        pid, process = inspection.pid, inspection.process
        if pid is None and inspection.error is None:
            pid, process = self._listener_details(port)
        details: dict[str, object] = {"port": port, "occupied": True}
        if pid is not None:
            details["pid"] = pid
        if process:
            details["process"] = process
        if inspection.error:
            details["error"] = _redact(inspection.error)
        suffix = ""
        if pid is not None:
            suffix += f", pid={pid}"
        if process:
            suffix += f", process={process}"
        return DoctorCheck(
            check_id,
            CheckStatus.WARN,
            f"Port {port} is occupied{suffix}",
            details,
        )

    def _check_runtime_health(self) -> DoctorCheck:
        url = f"http://127.0.0.1:{self.config.runtime_port}/health"
        try:
            response = self.local_get(url, min(self.config.timeout_s, 3))
        except urllib.error.HTTPError as error:
            return DoctorCheck(
                "runtime.health",
                CheckStatus.WARN,
                f"Runtime health endpoint returned HTTP {error.code}",
            )
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return DoctorCheck(
                "runtime.health",
                CheckStatus.SKIP,
                "Runtime service is not running",
            )
        if response.status_code != 200:
            return DoctorCheck(
                "runtime.health",
                CheckStatus.WARN,
                f"Runtime health returned HTTP {response.status_code}",
            )
        try:
            payload = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return DoctorCheck(
                "runtime.health", CheckStatus.WARN, "Port responds but health JSON is invalid"
            )
        if payload.get("service") != "shotseek-runtime":
            return DoctorCheck(
                "runtime.health", CheckStatus.WARN, "Port responds but is not ShotSeek Runtime"
            )
        status = str(payload.get("status", "")).lower()
        self._runtime_is_healthy = status == "ok"
        self._runtime_health_payload = payload
        return DoctorCheck(
            "runtime.health",
            CheckStatus.PASS if self._runtime_is_healthy else CheckStatus.FAIL,
            f"Runtime health: {status or 'unknown'}",
            {
                "schema_version": payload.get("schema_version"),
                "worker_enabled": payload.get("worker_enabled"),
            },
        )

    def _registry_path(self) -> Path:
        return self.runtime_root / "runtime.sqlite3"

    def _open_registry(self) -> sqlite3.Connection:
        connection = sqlite3.connect(_sqlite_uri(self._registry_path()), uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _check_registry_integrity(self) -> DoctorCheck:
        path = self._registry_path()
        if not path.is_file():
            return DoctorCheck(
                "runtime.registry_integrity", CheckStatus.SKIP, "Runtime registry does not exist"
            )
        with self._open_registry() as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
            integrity = str(result[0]) if result else "missing"
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        missing = sorted(REQUIRED_REGISTRY_TABLES - tables)
        if integrity.lower() != "ok" or missing:
            return DoctorCheck(
                "runtime.registry_integrity",
                CheckStatus.FAIL,
                "Runtime registry integrity failed",
                {"integrity_check": integrity, "missing_tables": missing},
            )
        return DoctorCheck(
            "runtime.registry_integrity",
            CheckStatus.PASS,
            "SQLite registry integrity: ok",
            {"path": str(path), "required_tables": sorted(REQUIRED_REGISTRY_TABLES)},
        )

    def _runtime_job_counts(self) -> tuple[dict[str, int], int] | None:
        if not self._runtime_is_healthy:
            return None
        url = f"http://127.0.0.1:{self.config.runtime_port}/api/v1/jobs"
        try:
            response = self.local_get(url, min(self.config.timeout_s, 3))
            if response.status_code != 200:
                return None
            payload = json.loads(response.body)
            items = payload.get("items")
            if not isinstance(items, list):
                return None
            states: list[str] = []
            resume_count = 0
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("state"), str):
                    return None
                states.append(item["state"])
                if item.get("resume_state") is not None:
                    resume_count += 1
            return dict(sorted(Counter(states).items())), resume_count
        except (
            urllib.error.URLError,
            ConnectionError,
            TimeoutError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            return None

    def _check_registry_jobs(self) -> DoctorCheck:
        path = self._registry_path()
        if not path.is_file():
            return DoctorCheck(
                "runtime.registry_jobs", CheckStatus.SKIP, "Runtime registry does not exist"
            )
        runtime_counts = self._runtime_job_counts()
        source = "runtime_api" if runtime_counts is not None else "read_only_sqlite_snapshot"
        with self._open_registry() as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "job" not in tables:
                return DoctorCheck(
                    "runtime.registry_jobs", CheckStatus.FAIL, "Runtime job table is missing"
                )
            rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM job GROUP BY state ORDER BY state"
            ).fetchall()
            resume_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM job WHERE resume_state IS NOT NULL"
                ).fetchone()[0]
            )
        if runtime_counts is None:
            counts = {str(row["state"]): int(row["count"]) for row in rows}
        else:
            counts, resume_count = runtime_counts
        unknown = sorted(set(counts) - KNOWN_JOB_STATES)
        active = sum(count for state, count in counts.items() if state not in TERMINAL_JOB_STATES)
        retrying = counts.get("RETRYING", 0)
        if unknown:
            return DoctorCheck(
                "runtime.registry_jobs",
                CheckStatus.FAIL,
                f"Unknown job states: {', '.join(unknown)}",
                {"source": source, "state_counts": counts, "resume_state_count": resume_count},
            )
        status = CheckStatus.PASS
        message = f"Job registry readable ({sum(counts.values())} jobs)"
        if retrying or resume_count:
            status = CheckStatus.WARN
            message = f"Job registry has retry/recovery state ({retrying} retrying, {resume_count} resumable)"
        elif active and not self._runtime_is_healthy:
            status = CheckStatus.WARN
            message = f"{active} non-terminal jobs exist while Runtime health is unavailable"
        return DoctorCheck(
            "runtime.registry_jobs",
            status,
            message,
            {
                "source": source,
                "state_counts": counts,
                "active_count": active,
                "resume_state_count": resume_count,
            },
        )

    def _check_scene_databases(self) -> DoctorCheck:
        registry = self._registry_path()
        if not registry.is_file():
            return DoctorCheck(
                "runtime.scene_databases", CheckStatus.SKIP, "Runtime registry does not exist"
            )
        with self._open_registry() as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "video" not in tables:
                return DoctorCheck(
                    "runtime.scene_databases", CheckStatus.FAIL, "Runtime video table is missing"
                )
            rows = connection.execute(
                "SELECT video_id, search_db_path FROM video WHERE search_db_path IS NOT NULL"
            ).fetchall()
        if not rows:
            return DoctorCheck(
                "runtime.scene_databases", CheckStatus.SKIP, "No Scene databases are registered"
            )
        failures: list[str] = []
        checked: list[str] = []
        for row in rows:
            video_id = str(row["video_id"])
            path = (self.project_root / str(row["search_db_path"])).resolve()
            if not path.is_relative_to(self.project_root):
                failures.append(f"{video_id}:path_escape")
                continue
            if not path.is_file():
                failures.append(f"{video_id}:missing")
                continue
            try:
                with sqlite3.connect(_sqlite_uri(path), uri=True) as database:
                    database.execute("PRAGMA query_only=ON")
                    result = database.execute("PRAGMA quick_check").fetchone()
                if not result or str(result[0]).lower() != "ok":
                    failures.append(f"{video_id}:integrity")
                else:
                    checked.append(video_id)
            except sqlite3.Error:
                failures.append(f"{video_id}:unreadable")
        if failures:
            return DoctorCheck(
                "runtime.scene_databases",
                CheckStatus.FAIL,
                f"Scene database checks failed for {len(failures)} video(s)",
                {"checked_video_ids": checked, "failures": failures},
            )
        return DoctorCheck(
            "runtime.scene_databases",
            CheckStatus.PASS,
            f"{len(checked)} Scene database(s) readable",
            {"checked_video_ids": checked},
        )

    def _static_root(self) -> Path:
        return self.project_root / "shotseek" / "runtime" / "static"

    def _check_frontend_build(self) -> DoctorCheck:
        root = self._static_root()
        index = root / "index.html"
        assets = root / "assets"
        missing = [str(path.relative_to(self.project_root)) for path in (index, assets) if not path.exists()]
        if missing:
            return DoctorCheck(
                "frontend.build",
                CheckStatus.FAIL,
                "Frontend static build is incomplete",
                {"missing": missing},
            )
        if not index.is_file() or not assets.is_dir():
            return DoctorCheck(
                "frontend.build", CheckStatus.FAIL, "Frontend static build has invalid file types"
            )
        return DoctorCheck(
            "frontend.build",
            CheckStatus.PASS,
            "Frontend static build exists",
            {"root": str(root)},
        )

    def _check_frontend_assets(self) -> DoctorCheck:
        root = self._static_root().resolve()
        index = root / "index.html"
        if not index.is_file():
            return DoctorCheck(
                "frontend.assets", CheckStatus.SKIP, "Frontend index.html is unavailable"
            )
        parser = _AssetParser()
        parser.feed(index.read_text(encoding="utf-8"))
        checked: list[str] = []
        missing: list[str] = []
        escaped: list[str] = []
        for reference in parser.references:
            path_part = urllib.parse.urlsplit(reference).path
            if not path_part.startswith(("/assets/", "assets/", "/favicon")):
                continue
            target = (root / path_part.lstrip("/")).resolve()
            if not target.is_relative_to(root):
                escaped.append(reference)
            elif not target.is_file():
                missing.append(reference)
            else:
                checked.append(reference)
        if escaped or missing or not checked:
            return DoctorCheck(
                "frontend.assets",
                CheckStatus.FAIL,
                "Frontend asset references are incomplete",
                {"checked": checked, "missing": missing, "escaped": escaped},
            )
        return DoctorCheck(
            "frontend.assets",
            CheckStatus.PASS,
            f"{len(checked)} frontend asset reference(s) resolved",
            {"checked": checked},
        )

    def _check_frontend_served(self) -> DoctorCheck:
        if not self._runtime_is_healthy:
            return DoctorCheck(
                "frontend.runtime_serve", CheckStatus.SKIP, "Runtime service is not running"
            )
        url = f"http://127.0.0.1:{self.config.runtime_port}/"
        try:
            response = self.local_get(url, min(self.config.timeout_s, 3))
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return DoctorCheck(
                "frontend.runtime_serve", CheckStatus.FAIL, "Runtime could not serve the frontend"
            )
        body = response.body.decode("utf-8", errors="replace")
        valid = response.status_code == 200 and "ShotSeek" in body
        return DoctorCheck(
            "frontend.runtime_serve",
            CheckStatus.PASS if valid else CheckStatus.FAIL,
            "Runtime serves the ShotSeek frontend" if valid else "Runtime frontend response is invalid",
            {"http_status": response.status_code},
        )

    def _check_stepfun_credential(self) -> DoctorCheck:
        primary_present = "STEPFUN_API_KEY" in self.environ
        primary = self.environ.get("STEPFUN_API_KEY", "").strip()
        legacy_present = "STEP_API_KEY" in self.environ
        legacy = self.environ.get("STEP_API_KEY", "").strip()
        env_file_present = (self.project_root / ".env").is_file()
        if primary:
            self._api_key = primary
            return DoctorCheck(
                "stepfun.credential",
                CheckStatus.PASS,
                "StepFun credential configured in process environment (value redacted)",
                {"configured": True, "source": "process_environment", "env_file_present": env_file_present},
            )
        if legacy:
            self._api_key = legacy
            return DoctorCheck(
                "stepfun.credential",
                CheckStatus.WARN,
                "Legacy STEP_API_KEY is configured; prefer STEPFUN_API_KEY (value redacted)",
                {"configured": True, "source": "legacy_process_environment", "env_file_present": env_file_present},
            )
        if primary_present or legacy_present:
            return DoctorCheck(
                "stepfun.credential",
                CheckStatus.WARN,
                "StepFun credential variable is present but empty",
                {"configured": False, "source": "process_environment", "env_file_present": env_file_present},
            )
        return DoctorCheck(
            "stepfun.credential",
            CheckStatus.WARN,
            "StepFun credential is not configured in the process environment",
            {
                "configured": False,
                "source": "not_configured",
                "env_file_present": env_file_present,
                "env_file_values_read": False,
            },
        )

    def _check_stepfun_connectivity(self) -> DoctorCheck:
        if not self.config.live:
            return DoctorCheck(
                "stepfun.connectivity",
                CheckStatus.SKIP,
                "StepFun connectivity not tested; use --live",
                {"provider_status": "SKIP"},
            )
        if not self._api_key:
            return DoctorCheck(
                "stepfun.connectivity",
                CheckStatus.FAIL,
                "StepFun live check requires a non-empty process credential",
                {"provider_status": "FAILED"},
            )
        base_url = self.environ.get("STEPFUN_CHAT_BASE_URL", DEFAULT_CHAT_BASE_URL).strip()
        model = self.environ.get("STEPFUN_VISION_MODEL", DEFAULT_STEPFUN_MODEL).strip()
        if not base_url or not model:
            return DoctorCheck(
                "stepfun.connectivity",
                CheckStatus.FAIL,
                "StepFun endpoint or model configuration is empty",
                {"provider_status": "FAILED"},
            )
        try:
            result = self.live_probe(base_url, self._api_key, model, self.config.timeout_s)
        except urllib.error.HTTPError as error:
            return DoctorCheck(
                "stepfun.connectivity",
                CheckStatus.FAIL,
                f"StepFun live check failed with HTTP {error.code}; offline checks remain valid",
                {"provider_status": "FAILED", "http_status": error.code, "model": model},
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            return DoctorCheck(
                "stepfun.connectivity",
                CheckStatus.FAIL,
                f"StepFun live check failed ({type(error).__name__}); offline checks remain valid",
                {"provider_status": "FAILED", "model": model},
            )
        if result.status_code != 200 or not result.response_valid:
            return DoctorCheck(
                "stepfun.connectivity",
                CheckStatus.FAIL,
                "StepFun live check returned an invalid response; offline checks remain valid",
                {
                    "provider_status": "FAILED",
                    "http_status": result.status_code,
                    "latency_ms": round(result.latency_ms, 1),
                    "model": model,
                },
            )
        return DoctorCheck(
            "stepfun.connectivity",
            CheckStatus.PASS,
            f"StepFun connectivity LIVE ({result.latency_ms:.1f} ms)",
            {
                "provider_status": "LIVE",
                "http_status": result.status_code,
                "latency_ms": round(result.latency_ms, 1),
                "model": model,
                "media_uploaded": False,
                "asr_started": False,
            },
        )

    def run(self) -> DoctorReport:
        self._record("project.metadata", self._check_project)
        self._record("python.version", self._check_python)
        self._record("binary.ffmpeg", lambda: self._check_binary("ffmpeg", "-version"))
        self._record("binary.ffprobe", lambda: self._check_binary("ffprobe", "-version"))
        self._record("binary.node", lambda: self._check_binary("node", "--version", required=False))
        self._record("binary.npm", lambda: self._check_binary("npm", "--version", required=False))
        self._record("python.dependencies", self._check_dependencies)
        self._record("browser.chromium", self._check_chromium)
        self._record("browser.playwright", self._check_playwright)

        self._record("nvidia.smi", self._check_nvidia_smi)
        self._record("nvidia.cuda_visibility", self._check_cuda_visibility)
        self._record("nvidia.nvdec", self._check_nvdec)
        self._record("nvidia.nvenc", self._check_nvenc)
        self._record("nvidia.unified_memory", self._check_unified_memory)
        self._record("nvidia.nvenc_probe", self._check_nvenc_deep)

        self._record(
            "storage.project",
            lambda: self._check_writable("storage.project", self.project_root, "Project directory"),
        )
        self._record(
            "storage.runtime",
            lambda: self._check_writable("storage.runtime", self.runtime_root, "Runtime directory"),
        )
        self._record(
            "storage.temp",
            lambda: self._check_writable("storage.temp", self.project_root / "tmp", "Temporary directory"),
        )
        self._record(
            "storage.sqlite",
            lambda: self._check_writable("storage.sqlite", self._registry_path().parent, "SQLite directory"),
        )
        self._record(
            "storage.uploads",
            lambda: self._check_writable("storage.uploads", self.runtime_root / "uploads", "Upload directory"),
        )
        self._record("storage.free_disk", self._check_disk)

        runtime_port_id = f"port.runtime.{self.config.runtime_port}"
        self._record(runtime_port_id, lambda: self._check_port("runtime", self.config.runtime_port))
        frontend_port_id = f"port.frontend.{self.config.frontend_port}"
        self._record(frontend_port_id, lambda: self._check_port("frontend", self.config.frontend_port))
        for port in self.config.debug_ports:
            check_id = f"port.debug.{port}"
            self._record(check_id, lambda port=port: self._check_port("debug", port))

        self._record("runtime.health", self._check_runtime_health)
        self._record("runtime.registry_integrity", self._check_registry_integrity)
        self._record("runtime.registry_jobs", self._check_registry_jobs)
        self._record("runtime.scene_databases", self._check_scene_databases)

        self._record("stepfun.credential", self._check_stepfun_credential)
        self._record("stepfun.connectivity", self._check_stepfun_connectivity)

        self._record("frontend.build", self._check_frontend_build)
        self._record("frontend.assets", self._check_frontend_assets)
        self._record("frontend.runtime_serve", self._check_frontend_served)

        summary = {
            status.value.lower(): sum(check.status == status for check in self._checks)
            for status in CheckStatus
        }
        if summary["fail"]:
            status = "fail"
        elif summary["warn"]:
            status = "pass_with_warnings"
        else:
            status = "pass"
        return DoctorReport(
            status=status,
            checks=tuple(self._checks),
            summary=summary,
            project_root=str(self.project_root),
            deep=self.config.deep,
            live=self.config.live,
        )


def format_terminal(report: DoctorReport, *, verbose: bool = False) -> str:
    lines = ["ShotSeek Doctor", ""]
    for check in report.checks:
        lines.append(f"[{check.status.value}] {check.message}")
        if verbose and check.details:
            details = json.dumps(check.details, ensure_ascii=False, sort_keys=True)
            lines.append(f"       {check.check_id}: {details}")
    label = {
        "pass": "PASS",
        "pass_with_warnings": "PASS WITH WARNINGS",
        "fail": "FAIL",
    }[report.status]
    lines.extend(
        [
            "",
            f"Result: {label}",
            (
                "Summary: "
                f"PASS={report.summary['pass']} "
                f"WARN={report.summary['warn']} "
                f"FAIL={report.summary['fail']} "
                f"SKIP={report.summary['skip']}"
            ),
        ]
    )
    return "\n".join(lines)
