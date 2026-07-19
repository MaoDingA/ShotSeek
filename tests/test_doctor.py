from __future__ import annotations

import json
import sqlite3
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from shotseek.cli import run as run_cli

from shotseek.doctor import (
    DOCTOR_SCHEMA_VERSION,
    CheckStatus,
    CommandResult,
    DoctorCheck,
    DoctorConfig,
    DoctorReport,
    HttpResult,
    LiveProbeResult,
    PortInspection,
    ShotSeekDoctor,
    format_terminal,
    local_http_get,
    stepfun_live_probe,
)


def make_project(root: Path) -> Path:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "shotseek"\nversion = "0.0.0"\ndependencies = []\n',
        encoding="utf-8",
    )
    static = root / "shotseek" / "runtime" / "static"
    (static / "assets").mkdir(parents=True)
    (static / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    (static / "assets" / "app.css").write_text("body{}", encoding="utf-8")
    (static / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (static / "index.html").write_text(
        '<title>ShotSeek</title><link rel="stylesheet" href="/assets/app.css">'
        '<link rel="icon" href="/favicon.svg"><script src="/assets/app.js"></script>',
        encoding="utf-8",
    )
    return root


class FakeRunner:
    def __init__(self, *, create_nvenc_output: bool = False) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.create_nvenc_output = create_nvenc_output

    def __call__(self, argv, timeout_s: float) -> CommandResult:
        command = tuple(str(item) for item in argv)
        self.commands.append(command)
        executable = Path(command[0]).name
        if executable == "nvidia-smi":
            return CommandResult(0, "NVIDIA GB10, 590.00, 131072, 65536\n")
        if executable == "ffmpeg" and "-decoders" in command:
            return CommandResult(0, " V..... h264_cuvid H.264 CUVID\n V..... hevc_cuvid HEVC CUVID\n")
        if executable == "ffmpeg" and "-encoders" in command:
            return CommandResult(0, " V....D h264_nvenc NVIDIA NVENC H.264 encoder\n")
        if executable == "ffmpeg" and "-hwaccels" in command:
            return CommandResult(0, "Hardware acceleration methods:\ncuda\n")
        if executable == "ffmpeg" and "h264_nvenc" in command:
            if self.create_nvenc_output:
                Path(command[-1]).write_bytes(b"synthetic-nvenc")
                return CommandResult(0)
            return CommandResult(1, stderr="encode failed")
        if executable == "ffmpeg":
            return CommandResult(0, "ffmpeg version test")
        if executable == "ffprobe":
            return CommandResult(0, "ffprobe version test")
        if executable == "node":
            return CommandResult(0, "v22.0.0")
        if executable == "npm":
            return CommandResult(0, "10.0.0")
        if executable in {"ss", "lsof"}:
            return CommandResult(0, "")
        return CommandResult(0, f"{executable} test")


def fake_which(name: str) -> str:
    return f"/usr/bin/{name}"


def runtime_offline(url: str, timeout_s: float) -> HttpResult:
    raise urllib.error.URLError("offline")


def ports_available(host: str, port: int) -> PortInspection:
    return PortInspection(True)


def check_by_id(report: DoctorReport, check_id: str) -> DoctorCheck:
    return next(check for check in report.checks if check.check_id == check_id)


def test_doctor_check_contract_and_report_schema(tmp_path: Path) -> None:
    check = DoctorCheck("sample.check", CheckStatus.PASS, "ok", {"value": 1})
    report = DoctorReport(
        status="pass",
        checks=(check,),
        summary={"pass": 1, "warn": 0, "fail": 0, "skip": 0},
        project_root=str(tmp_path),
        deep=False,
        live=False,
    )
    payload = report.to_dict()
    assert list(payload) == [
        "schema_version",
        "status",
        "project_root",
        "mode",
        "checks",
        "summary",
    ]
    assert payload["schema_version"] == DOCTOR_SCHEMA_VERSION
    assert payload["checks"][0]["check_id"] == "sample.check"
    with pytest.raises(ValueError, match="invalid doctor check_id"):
        DoctorCheck("Bad ID", CheckStatus.PASS, "bad")


def test_doctor_config_rejects_unsafe_or_invalid_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside project_root"):
        DoctorConfig(project_root=tmp_path, runtime_root=tmp_path.parent / "outside")
    with pytest.raises(ValueError, match="lower than"):
        DoctorConfig(project_root=tmp_path, disk_warn_gb=5, disk_fail_gb=5)
    with pytest.raises(ValueError, match="invalid TCP port"):
        DoctorConfig(project_root=tmp_path, runtime_port=0)


def test_default_run_is_offline_read_only_and_has_unique_ids(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    (root / ".env").write_text("STEPFUN_API_KEY=must-not-be-read\n", encoding="utf-8")
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    live_calls: list[tuple[str, str, str, float]] = []

    def forbidden_live(base_url: str, api_key: str, model: str, timeout_s: float):
        live_calls.append((base_url, api_key, model, timeout_s))
        raise AssertionError("default doctor reached StepFun")

    doctor = ShotSeekDoctor(
        DoctorConfig(project_root=root),
        environ={},
        runner=FakeRunner(),
        which=fake_which,
        local_get=runtime_offline,
        live_probe=forbidden_live,
        port_inspector=ports_available,
    )
    report = doctor.run()
    after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    assert before == after
    assert not live_calls
    assert report.summary["fail"] == 0
    ids = [check.check_id for check in report.checks]
    assert len(ids) == len(set(ids))
    credential = check_by_id(report, "stepfun.credential")
    assert credential.status == CheckStatus.WARN
    assert credential.details["env_file_present"] is True
    assert credential.details["env_file_values_read"] is False
    assert "must-not-be-read" not in json.dumps(report.to_dict())


def test_single_check_exception_is_redacted_and_does_not_abort(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    secret = "top-secret-stepfun-token"
    doctor = ShotSeekDoctor(
        DoctorConfig(project_root=root),
        environ={"STEPFUN_API_KEY": secret},
        runner=FakeRunner(),
        which=fake_which,
        local_get=runtime_offline,
        port_inspector=ports_available,
    )

    def broken_python() -> DoctorCheck:
        raise RuntimeError(f"authorization=Bearer {secret}")

    doctor._check_python = broken_python  # type: ignore[method-assign]
    report = doctor.run()
    failed = check_by_id(report, "python.version")
    assert failed.status == CheckStatus.FAIL
    raw = json.dumps(report.to_dict())
    assert secret not in raw
    assert "<redacted>" in raw
    assert check_by_id(report, "frontend.assets").status == CheckStatus.PASS


def test_stepfun_live_probe_is_explicit_and_reports_live_without_media(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    secret = "secret-that-must-not-appear"
    calls: list[tuple[str, str, str, float]] = []

    def live_probe(base_url: str, api_key: str, model: str, timeout_s: float) -> LiveProbeResult:
        calls.append((base_url, api_key, model, timeout_s))
        return LiveProbeResult(200, 12.5, True)

    doctor = ShotSeekDoctor(
        DoctorConfig(project_root=root, live=True),
        environ={"STEPFUN_API_KEY": secret},
        runner=FakeRunner(),
        which=fake_which,
        local_get=runtime_offline,
        live_probe=live_probe,
        port_inspector=ports_available,
    )
    report = doctor.run()
    assert len(calls) == 1
    live = check_by_id(report, "stepfun.connectivity")
    assert live.status == CheckStatus.PASS
    assert live.details["provider_status"] == "LIVE"
    assert live.details["media_uploaded"] is False
    assert live.details["asr_started"] is False
    assert secret not in json.dumps(report.to_dict())


def test_deep_nvenc_probe_is_project_local_and_cleans_up(tmp_path: Path, monkeypatch) -> None:
    root = make_project(tmp_path)
    monkeypatch.setattr("shotseek.doctor.platform.system", lambda: "Linux")
    runner = FakeRunner(create_nvenc_output=True)
    doctor = ShotSeekDoctor(
        DoctorConfig(project_root=root, deep=True),
        environ={},
        runner=runner,
        which=fake_which,
        local_get=runtime_offline,
        port_inspector=ports_available,
    )
    check = doctor._check_nvenc_deep()
    assert check.status == CheckStatus.PASS
    assert check.details["temporary_output_removed"] is True
    assert not (root / "tmp").exists()
    command = next(command for command in runner.commands if "h264_nvenc" in command)
    output = Path(command[-1])
    assert output.is_relative_to(root)
    assert not output.exists()


def test_macos_skips_nvidia_checks(tmp_path: Path, monkeypatch) -> None:
    root = make_project(tmp_path)
    monkeypatch.setattr("shotseek.doctor.platform.system", lambda: "Darwin")
    doctor = ShotSeekDoctor(DoctorConfig(project_root=root, deep=True), environ={})
    assert doctor._check_nvidia_smi().status == CheckStatus.SKIP
    assert doctor._check_cuda_visibility().status == CheckStatus.SKIP
    assert doctor._check_nvdec().status == CheckStatus.SKIP
    assert doctor._check_nvenc().status == CheckStatus.SKIP
    assert doctor._check_unified_memory().status == CheckStatus.SKIP
    assert doctor._check_nvenc_deep().status == CheckStatus.SKIP


def test_disk_thresholds_are_configurable(tmp_path: Path, monkeypatch) -> None:
    root = make_project(tmp_path)
    doctor = ShotSeekDoctor(
        DoctorConfig(project_root=root, disk_warn_gb=20, disk_fail_gb=5),
        environ={},
    )
    monkeypatch.setattr(
        "shotseek.doctor.shutil.disk_usage",
        lambda path: SimpleNamespace(total=100 * 1024**3, used=94 * 1024**3, free=6 * 1024**3),
    )
    assert doctor._check_disk().status == CheckStatus.WARN
    monkeypatch.setattr(
        "shotseek.doctor.shutil.disk_usage",
        lambda path: SimpleNamespace(total=100 * 1024**3, used=96 * 1024**3, free=4 * 1024**3),
    )
    assert doctor._check_disk().status == CheckStatus.FAIL


def test_occupied_port_reports_process_without_killing(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    doctor = ShotSeekDoctor(
        DoctorConfig(project_root=root),
        environ={},
        port_inspector=lambda host, port: PortInspection(False, pid=1234, process="python"),
    )
    check = doctor._check_port("runtime", 8000)
    assert check.status == CheckStatus.WARN
    assert check.details["pid"] == 1234
    assert check.details["process"] == "python"
    assert "pid=1234" in check.message


def create_registry(root: Path) -> tuple[Path, Path]:
    runtime = root / "data" / "runtime"
    runtime.mkdir(parents=True)
    scene = runtime / "videos" / "video_a" / "search.sqlite3"
    scene.parent.mkdir(parents=True)
    with sqlite3.connect(scene) as connection:
        connection.execute("CREATE TABLE scene(scene_id TEXT PRIMARY KEY)")
    registry = runtime / "runtime.sqlite3"
    with sqlite3.connect(registry) as connection:
        connection.executescript(
            """
            CREATE TABLE video(video_id TEXT, search_db_path TEXT);
            CREATE TABLE job(state TEXT, resume_state TEXT);
            CREATE TABLE job_event(event_id INTEGER);
            CREATE TABLE artifact(artifact_id TEXT);
            """
        )
        connection.execute(
            "INSERT INTO video VALUES (?, ?)",
            ("video_a", str(scene.relative_to(root))),
        )
        connection.execute("INSERT INTO job VALUES ('RETRYING', 'ANALYZING_ASR')")
    return registry, scene


def test_registry_and_scene_checks_are_read_only(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    registry, scene = create_registry(root)
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (registry, scene)
    }
    doctor = ShotSeekDoctor(DoctorConfig(project_root=root), environ={})
    assert doctor._check_registry_integrity().status == CheckStatus.PASS
    jobs = doctor._check_registry_jobs()
    assert jobs.status == CheckStatus.WARN
    assert jobs.details["resume_state_count"] == 1
    assert doctor._check_scene_databases().status == CheckStatus.PASS
    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (registry, scene)
    }
    assert before == after
    assert not list(root.rglob("*-wal"))
    assert not list(root.rglob("*-shm"))


def test_runtime_health_skip_and_frontend_local_serve(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    offline = ShotSeekDoctor(
        DoctorConfig(project_root=root), environ={}, local_get=runtime_offline
    )
    assert offline._check_runtime_health().status == CheckStatus.SKIP

    def local_get(url: str, timeout_s: float) -> HttpResult:
        if url.endswith("/health"):
            return HttpResult(
                200,
                json.dumps(
                    {
                        "status": "ok",
                        "service": "shotseek-runtime",
                        "schema_version": "m3-runtime-api-v1",
                        "worker_enabled": True,
                    }
                ).encode(),
            )
        if url.endswith("/api/v1/jobs"):
            return HttpResult(
                200,
                b'{"items":[{"state":"QUEUED","resume_state":"PROBING"}]}',
            )
        return HttpResult(200, b"<title>ShotSeek</title>")

    online = ShotSeekDoctor(
        DoctorConfig(project_root=root), environ={}, local_get=local_get
    )
    assert online._check_runtime_health().status == CheckStatus.PASS
    assert online._check_frontend_served().status == CheckStatus.PASS
    assert online._runtime_job_counts() == ({"QUEUED": 1}, 1)


def test_runtime_http_error_is_warning_not_service_absent(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    error = urllib.error.HTTPError(
        "http://127.0.0.1:8000/health", 404, "missing", {}, None
    )
    doctor = ShotSeekDoctor(
        DoctorConfig(project_root=root),
        environ={},
        local_get=lambda url, timeout: (_ for _ in ()).throw(error),
    )
    check = doctor._check_runtime_health()
    assert check.status == CheckStatus.WARN
    assert "HTTP 404" in check.message


def test_local_http_rejects_public_network_and_live_url_rejects_credentials() -> None:
    with pytest.raises(ValueError, match="localhost"):
        local_http_get("https://example.com/health", 1)
    with pytest.raises(ValueError, match="must not contain credentials"):
        stepfun_live_probe("https://user:pass@example.com/v1", "secret", "model", 1)


def test_terminal_output_uses_required_status_vocabulary(tmp_path: Path) -> None:
    checks = (
        DoctorCheck("a.pass", CheckStatus.PASS, "pass"),
        DoctorCheck("a.warn", CheckStatus.WARN, "warn"),
        DoctorCheck("a.skip", CheckStatus.SKIP, "skip"),
    )
    report = DoctorReport(
        status="pass_with_warnings",
        checks=checks,
        summary={"pass": 1, "warn": 1, "fail": 0, "skip": 1},
        project_root=str(tmp_path),
        deep=False,
        live=False,
    )
    output = format_terminal(report, verbose=True)
    assert "[PASS] pass" in output
    assert "[WARN] warn" in output
    assert "[SKIP] skip" in output
    assert "Result: PASS WITH WARNINGS" in output


def test_doctor_source_contains_no_destructive_service_commands() -> None:
    source = Path("shotseek/doctor.py").read_text(encoding="utf-8")
    for forbidden in ("fuser", "pkill", "systemctl", "sudo"):
        assert forbidden not in source
    assert "env_file.read_text" not in source
    assert "env_file.open" not in source
    assert "load_dotenv" not in source


@pytest.mark.parametrize(
    ("status", "expected_code"),
    (("pass", 0), ("pass_with_warnings", 0), ("fail", 1)),
)
def test_cli_json_output_and_exit_code(
    tmp_path: Path,
    monkeypatch,
    capsys,
    status: str,
    expected_code: int,
) -> None:
    check_status = CheckStatus.FAIL if status == "fail" else CheckStatus.PASS
    report = DoctorReport(
        status=status,
        checks=(DoctorCheck("cli.test", check_status, "cli result"),),
        summary={
            "pass": int(check_status == CheckStatus.PASS),
            "warn": 0,
            "fail": int(check_status == CheckStatus.FAIL),
            "skip": 0,
        },
        project_root=str(tmp_path),
        deep=False,
        live=False,
    )

    class FakeDoctor:
        def __init__(self, config: DoctorConfig) -> None:
            assert config.project_root == tmp_path.resolve()

        def run(self) -> DoctorReport:
            return report

    monkeypatch.setattr("shotseek.cli.ShotSeekDoctor", FakeDoctor)
    code = run_cli(["doctor", "--project-root", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == expected_code
    assert payload["schema_version"] == DOCTOR_SCHEMA_VERSION
    assert payload["status"] == status


def test_cli_rejects_invalid_thresholds(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        run_cli(
            [
                "doctor",
                "--project-root",
                str(tmp_path),
                "--disk-warn-gb",
                "5",
                "--disk-fail-gb",
                "5",
            ]
        )
    assert error.value.code == 2
