"""Local web interface for people who do not use a terminal.

Design constraints that matter in a corporate network:

* **Loopback only.** The server binds ``127.0.0.1`` and refuses other hosts, so the
  project is never exposed to the LAN.
* **Token-gated.** Every ``/api`` call needs a per-session token that is printed
  in the URL the launcher opens; a random web page in the same browser cannot
  drive the pipeline (CSRF defence).
* **Stdlib only.** No extra dependency to clear with IT; the UI ships with the
  package and works offline.
* **One job at a time.** Pipeline phases are heavy and stateful, so the server
  serialises them and reports plain-language progress.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
import webbrowser
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import parse_qs, urlsplit

from quarkon.core.config import PipelineConfig
from quarkon.core.exceptions import ConfigError, QuarkonError
from quarkon.core.logging import get_logger
from quarkon.core.version import __version__
from quarkon.corporate import (
    COMPLIANCE_NOTES,
    TEMPLATES,
    audit_report,
    cost_estimate,
    export_audit,
    find_template,
    problem_guidance,
    scaffold_template,
)
from quarkon.i18n import (
    LOCALE_ROADMAP,
    available_locales,
    detect_locale,
    get_translator,
    save_preferred_locale,
)

__all__ = ["UIServer", "serve"]

_log = get_logger("ui.server")

_ASSETS: Final[Path] = Path(__file__).parent / "assets"
_MAX_BODY: Final[int] = 1 * 1024 * 1024
_JOB_LOG_LIMIT: Final[int] = 500


@dataclass
class Job:
    """One pipeline phase running in the background."""

    job_id: str
    phase: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_JOB_LOG_LIMIT))
    cancel_requested: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "phase": self.phase,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round((self.finished_at or time.time()) - self.started_at, 1),
            "error": self.error,
            "result": self.result,
            "events": list(self.events)[-80:],
        }


class UIServer:
    """Owns the HTTP server, the session token and the active job."""

    def __init__(
        self,
        *,
        project_dir: Path,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
        locale: str | None = None,
    ) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigError(
                "the interface may only listen on this computer",
                hint="use --host 127.0.0.1; exposing the UI to a network is not supported",
            )
        self.project_dir = project_dir.expanduser().resolve()
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.locale = locale or detect_locale()
        self.token = secrets.token_urlsafe(24)
        self._job: Job | None = None
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None

    # ------------------------------------------------------------------ jobs
    @property
    def job(self) -> Job | None:
        return self._job

    _PHASES = frozenset({"synthesize", "train", "export"})

    def start_job(self, phase: str, config_path: Path | None = None) -> Job:
        if phase not in self._PHASES:
            raise ConfigError(
                f"unknown step: {phase!r}",
                hint="allowed steps: synthesize, train, export",
            )
        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise ConfigError("a step is already running; wait for it to finish")
            job = Job(job_id=secrets.token_hex(6), phase=phase)
            self._job = job
            thread = threading.Thread(
                target=self._run_job, args=(job, config_path), name=f"quarkon-ui-{phase}", daemon=True
            )
            thread.start()
            return job

    def _run_job(self, job: Job, config_path: Path | None) -> None:
        def on_event(payload: dict[str, Any]) -> None:
            event = dict(payload)
            event.setdefault("ts", round(time.time(), 3))
            job.events.append(event)

        try:
            path = config_path or (self.project_dir / "distill.yaml")
            if not path.is_file():
                raise ConfigError(
                    "no project configuration found",
                    hint="create a project in the Setup tab first",
                )
            config = PipelineConfig.from_yaml(path)
            from quarkon.engine import DistillationEngine, Phase

            engine = DistillationEngine(
                config,
                options=self._runtime_options(config),
                on_event=on_event,
            )
            phases = {"synthesize": "synthesize", "train": "train", "export": "export"}
            phase = phases.get(job.phase)
            if phase is None:
                raise ConfigError(f"unknown step: {job.phase}")
            selected: tuple[Phase, ...] = (cast(Phase, phase),)
            result = engine.run(phases=selected)
            job.result = result.to_dict() if hasattr(result, "to_dict") else {"ok": True}
            job.status = "done" if result.ok else "failed"
            if not result.ok and not job.error:
                job.error = {
                    "code": "PipelineFailed",
                    "title": "The step did not finish",
                    "meaning": "See the log below for the exact reason.",
                    "fix": "Fix the issue and press the button again; finished work is saved.",
                }
        except QuarkonError as exc:
            job.status = "failed"
            code = type(exc).__name__
            guidance = problem_guidance(code) or problem_guidance("ConfigError") or {}
            job.error = {
                "code": code,
                "title": guidance.get("title", code),
                "meaning": guidance.get("meaning", exc.message),
                "fix": guidance.get("fix", exc.hint or "Check the log below."),
                "details": exc.to_dict(),
            }
            job.events.append({"phase": "error", "error": exc.message, "ts": round(time.time(), 3)})
        except Exception as exc:
            job.status = "failed"
            job.error = {
                "code": type(exc).__name__,
                "title": "An unexpected problem stopped the step",
                "meaning": str(exc),
                "fix": "Run 'quarkon doctor' in a terminal and send the log to support.",
            }
            _log.exception("ui.job_failed", phase=job.phase)
        finally:
            job.finished_at = time.time()

    def _runtime_options(self, config: PipelineConfig) -> Any:
        from quarkon.core.config import RuntimeOptions

        _ = config
        return RuntimeOptions(workspace=self.project_dir)

    # ------------------------------------------------------------------ http
    def serve_forever(self) -> None:
        handler = _make_handler(self)
        httpd = ThreadingHTTPServer((self.host, self.port), handler)
        httpd.daemon_threads = True
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        url = f"http://{self.host}:{self.port}/?token={self.token}"
        if self.open_browser:
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        _log.info("ui.listening", url=f"http://{self.host}:{self.port}/")
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()

    def shutdown(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()

    # ------------------------------------------------------------- API logic
    def bootstrap(self) -> dict[str, Any]:
        config_path = self.project_dir / "distill.yaml"
        manifest_path = self.project_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
        dataset = self.project_dir / "samples.jsonl"
        return {
            "version": __version__,
            "locale": self.locale,
            "locales": [info.to_dict() for info in available_locales()],
            "locale_roadmap": [{"code": code, "english_name": name} for code, name in LOCALE_ROADMAP],
            "templates": [template.to_dict() for template in TEMPLATES],
            "project_dir": str(self.project_dir),
            "has_project": config_path.is_file(),
            "config_path": str(config_path),
            "dataset_lines": _count_lines(dataset),
            "manifest": {
                "run_id": manifest.get("run_id"),
                "status": manifest.get("status"),
                "synthesis": manifest.get("synthesis"),
                "training": manifest.get("training"),
                "deployment": manifest.get("deployment"),
            },
            "compliance": [dict(note) for note in COMPLIANCE_NOTES],
            "job": self._job.snapshot() if self._job is not None else None,
        }

    def state(self) -> dict[str, Any]:
        config_path = self.project_dir / "distill.yaml"
        dataset = self.project_dir / "samples.jsonl"
        adapter = self.project_dir / "adapter" / "adapter_model.safetensors"
        models_dir = self.project_dir / "models"
        gguf = sorted(models_dir.rglob("*.gguf")) if models_dir.is_dir() else []
        return {
            "has_project": config_path.is_file(),
            "dataset_lines": _count_lines(dataset),
            "has_adapter": adapter.is_file(),
            "gguf": [str(path) for path in gguf],
            "job": self._job.snapshot() if self._job is not None else None,
        }

    def setup_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        template_id = str(payload.get("template") or "").strip()
        objective = str(payload.get("objective") or "").strip()
        samples = int(payload.get("samples") or 300)
        language = str(payload.get("language") or "en")
        directory = str(payload.get("directory") or "").strip()
        if template_id:
            template = find_template(template_id)
            if template is None:
                raise ConfigError(f"unknown template: {template_id}")
            target = Path(directory) if directory else self.project_dir / template.template_id
            config_path, seeds_path = scaffold_template(template, target)
        else:
            if len(objective) < 20:
                raise ConfigError(
                    "describe the task in at least one full sentence",
                    hint="for example: extract the invoice number, vendor and total from a PDF",
                )
            slug = _slug(objective)
            target = Path(directory) if directory else self.project_dir / slug
            template = None
            from quarkon.core.config import render_config_template

            target.mkdir(parents=True, exist_ok=True)
            config_path = target / "distill.yaml"
            config_path.write_text(render_config_template(task_name=slug, objective=objective), encoding="utf-8")
            seeds_path = target / "seeds.jsonl"
            seeds_path.write_text(
                json.dumps({"prompt": objective}, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        text = config_path.read_text(encoding="utf-8")
        text = text.replace("  target_samples: 500", f"  target_samples: {max(50, samples)}")
        text = text.replace("  language: en", f"  language: {language}")
        config_path.write_text(text, encoding="utf-8")
        PipelineConfig.from_yaml(config_path)  # validates before reporting success
        self.project_dir = target
        return {
            "ok": True,
            "project_dir": str(target),
            "config_path": str(config_path),
            "seeds_path": str(seeds_path),
            "template": template.template_id if template else None,
        }

    def doctor(self) -> dict[str, Any]:
        from quarkon.core.linter import run_preflight_checks

        config_path = self.project_dir / "distill.yaml"
        if not config_path.is_file():
            return {
                "ok": False,
                "findings": [
                    {
                        "code": "no_project",
                        "severity": "warning",
                        "title": "No project yet",
                        "meaning": "Create one in the Setup tab; the check runs afterwards.",
                        "fix": "Open Setup and choose a task.",
                    }
                ],
            }
        config = PipelineConfig.from_yaml(config_path)
        result = run_preflight_checks(config)
        findings: list[dict[str, Any]] = []
        for finding in [*result.blockers, *result.warnings]:
            guidance = problem_guidance(finding.code) or {}
            findings.append(
                {
                    "code": finding.code,
                    "severity": finding.severity,
                    "title": guidance.get("title", finding.message),
                    "meaning": guidance.get("meaning", finding.message),
                    "fix": guidance.get("fix", finding.hint or ""),
                }
            )
        for passed in result.checks_run:
            findings.append(
                {
                    "code": passed,
                    "severity": "ok",
                    "title": passed.replace("_", " ").capitalize(),
                    "meaning": "Check passed.",
                    "fix": "",
                }
            )
        return {
            "ok": result.ok,
            "duration_ms": result.duration_ms,
            "hardware": result.hardware.to_dict() if result.hardware else None,
            "findings": findings,
        }

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        samples = int(payload.get("samples") or 300)
        completion = int(payload.get("completion_tokens") or 300)
        return cost_estimate(samples, completion_tokens=completion)

    def audit(self) -> dict[str, Any]:
        return audit_report(self.project_dir)

    def export_audit(self, payload: dict[str, Any]) -> dict[str, Any]:
        fmt = str(payload.get("format") or "json")
        destination = self.project_dir / f"audit-report.{fmt}"
        written = export_audit(self.project_dir, destination, fmt=fmt)
        return {"ok": True, "path": str(written)}

    def set_locale(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.locale = save_preferred_locale(str(payload.get("locale") or "en"))
        return {"ok": True, "locale": self.locale}

    def translator(self) -> Any:
        return get_translator(self.locale)

    def labels(self) -> dict[str, str]:
        """Every interface string for the current locale (the JS applies them)."""
        keys = (
            "app.tagline",
            "nav.setup",
            "nav.run",
            "nav.help",
            "nav.languages",
            "common.next",
            "common.back",
            "common.close",
            "common.loading",
            "common.yes",
            "common.no",
            "setup.title",
            "setup.step1",
            "setup.step2",
            "setup.step3",
            "setup.step4",
            "field.objective",
            "field.template",
            "field.samples",
            "field.language",
            "field.seeds",
            "action.create",
            "action.preview",
            "action.synthesize",
            "action.train",
            "action.export",
            "action.check_system",
            "status.running",
            "status.done",
            "status.failed",
            "doctor.title",
            "doctor.ok",
            "doctor.fix",
            "cost.title",
            "cost.note",
            "results.title",
            "results.next",
            "help.title",
            "help.plain",
            "help.security",
            "error.title",
            "error.fix",
            "privacy.no_telemetry",
            "privacy.local_first",
            "lang.current",
            "lang.partial",
        )
        translator = self.translator()
        return {key: translator.t(key) for key in keys}


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _line in handle)
    except OSError:  # pragma: no cover - permission problems
        return 0


def _slug(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts)[:60] or "my-task"


def _make_handler(server: UIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"Quarkon/{__version__}"
        sys_version = ""

        # -------------------------------------------------------- plumbing
        def log_message(self, format: str, *args: object) -> None:
            _log.debug("ui.request", detail=format % args)

        def _authorised(self) -> bool:
            token = self.headers.get("X-Quarkon-Token", "")
            if not token:
                query = parse_qs(urlsplit(self.path).query)
                token = (query.get("token") or [""])[0]
            return bool(token) and secrets.compare_digest(token, server.token)

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_asset(self, name: str, content_type: str) -> None:
            path = (_ASSETS / name).resolve()
            if not str(path).startswith(str(_ASSETS.resolve())) or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > _MAX_BODY:
                raise ConfigError("request body too large")
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ConfigError("invalid JSON body", details={"error": str(exc)}) from exc
            return payload if isinstance(payload, dict) else {}

        # ----------------------------------------------------------- routes
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            route = parsed.path
            if route in {"/", "/index.html"}:
                self._send_asset("index.html", "text/html; charset=utf-8")
                return
            if route == "/assets/style.css":
                self._send_asset("style.css", "text/css; charset=utf-8")
                return
            if route == "/assets/app.js":
                self._send_asset("app.js", "application/javascript; charset=utf-8")
                return
            if not self._authorised():
                self._send_json({"error": "unauthorised"}, HTTPStatus.FORBIDDEN)
                return
            try:
                if route == "/api/bootstrap":
                    payload = server.bootstrap()
                    payload["labels"] = server.labels()
                    self._send_json(payload)
                    return
                if route == "/api/state":
                    self._send_json(server.state())
                    return
                if route == "/api/doctor":
                    self._send_json(server.doctor())
                    return
                if route == "/api/audit":
                    self._send_json(server.audit())
                    return
                if route == "/api/locales":
                    self._send_json(
                        {
                            "locale": server.locale,
                            "locales": [info.to_dict() for info in available_locales()],
                        }
                    )
                    return
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except QuarkonError as exc:
                self._send_json({"error": exc.to_dict()}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                _log.exception("ui.get_failed", route=route)
                self._send_json({"error": {"message": str(exc)}}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:
            route = urlsplit(self.path).path
            if not self._authorised():
                self._send_json({"error": "unauthorised"}, HTTPStatus.FORBIDDEN)
                return
            try:
                body = self._read_body()
                if route == "/api/setup":
                    self._send_json(server.setup_project(body))
                    return
                if route == "/api/plan":
                    self._send_json(server.plan(body))
                    return
                if route == "/api/locale":
                    self._send_json(server.set_locale(body))
                    return
                if route == "/api/audit/export":
                    self._send_json(server.export_audit(body))
                    return
                if route == "/api/run":
                    phase = str(body.get("phase") or "")
                    job = server.start_job(phase)
                    self._send_json({"ok": True, "job": job.snapshot()})
                    return
                if route == "/api/cancel":
                    running_job = server.job
                    if running_job is not None and running_job.status == "running":
                        running_job.cancel_requested = True
                        running_job.events.append(
                            {"phase": "cancel_requested", "ts": round(time.time(), 3)}
                        )
                    self._send_json({"ok": True})
                    return
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except ConfigError as exc:
                self._send_json({"error": exc.to_dict()}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                _log.exception("ui.post_failed", route=route)
                self._send_json({"error": {"message": str(exc)}}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def serve(
    *,
    project_dir: Path,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    locale: str | None = None,
    on_ready: Callable[[UIServer], None] | None = None,
) -> None:
    """Start the interface and block until interrupted."""
    server = UIServer(
        project_dir=project_dir,
        host=host,
        port=port,
        open_browser=open_browser,
        locale=locale,
    )
    if on_ready is not None:
        on_ready(server)
    server.serve_forever()
