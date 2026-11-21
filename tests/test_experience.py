"""Tests for the corporate-experience layer: i18n, templates, doctor, GUI server."""

from __future__ import annotations

import csv
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from quarkon.cli.main import main
from quarkon.corporate import (
    COMPLIANCE_NOTES,
    PROBLEM_GUIDE,
    TEMPLATES,
    audit_report,
    cost_estimate,
    export_audit,
    find_template,
    problem_guidance,
    scaffold_template,
)
from quarkon.i18n import (
    BASE_LOCALE,
    LOCALE_ROADMAP,
    SUPPORTED_LOCALES,
    Translator,
    available_locales,
    detect_locale,
    get_translator,
    load_catalog,
    save_preferred_locale,
)
from quarkon.ui.server import UIServer

# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------


class TestI18n:
    def test_every_supported_locale_has_a_catalog(self) -> None:
        for info in SUPPORTED_LOCALES:
            catalog = load_catalog(info.code)
            assert catalog, f"missing catalog for {info.code}"

    def test_catalogs_have_the_same_keys_as_english(self) -> None:
        base_keys = set(load_catalog(BASE_LOCALE))
        assert len(base_keys) >= 40
        for info in SUPPORTED_LOCALES:
            keys = set(load_catalog(info.code))
            assert keys == base_keys, f"{info.code} missing: {sorted(base_keys - keys)}"

    def test_no_russian_catalog_is_shipped(self) -> None:
        codes = {info.code for info in SUPPORTED_LOCALES}
        assert not any(code == "ru" or code.startswith("ru-") for code in codes)
        locale_dir = Path("quarkon/i18n/locales")
        assert not any(path.stem.startswith("ru") for path in locale_dir.glob("*.json"))

    def test_fallback_to_english_for_unknown_locale(self) -> None:
        translator = Translator("xx-unknown")
        assert translator.locale == BASE_LOCALE
        assert translator.t("nav.setup")

    def test_missing_key_falls_back_to_base_catalog(self) -> None:
        translator = Translator("de")
        assert translator.t("nav.setup") == load_catalog("de")["nav.setup"]

    def test_placeholder_formatting(self) -> None:
        translator = Translator("en")
        assert "{value}" not in translator.t("nav.setup")
        assert translator.t("nav.setup") == translator("nav.setup")

    def test_detect_locale_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QUARKON_LANG", "de")
        assert detect_locale() == "de"
        monkeypatch.setenv("QUARKON_LANG", "zh-cn")
        assert detect_locale() == "zh-Hans"

    def test_saved_preference_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        chosen = save_preferred_locale("ja")
        assert chosen == "ja"
        assert get_translator("ja").locale == "ja"

    def test_registry_is_honest_about_roadmap(self) -> None:
        shipped = {info.code for info in available_locales()}
        planned = {code for code, _name in LOCALE_ROADMAP}
        assert not shipped & planned
        assert len(shipped) >= 20
        assert len(planned) >= 30

    def test_rtl_flags(self) -> None:
        by_code = {info.code: info for info in SUPPORTED_LOCALES}
        assert by_code["ar"].rtl and by_code["he"].rtl
        assert not by_code["en"].rtl


# ---------------------------------------------------------------------------
# Corporate helpers
# ---------------------------------------------------------------------------


class TestCorporate:
    def test_templates_are_complete(self) -> None:
        assert len(TEMPLATES) >= 5
        for template in TEMPLATES:
            assert len(template.objective) >= 20
            assert len(template.seeds) >= 3
            assert template.recommended_samples >= 50
            assert find_template(template.template_id) is template

    def test_every_template_scaffolds_a_valid_project(self, tmp_path: Path) -> None:
        from quarkon.core.config import PipelineConfig

        for template in TEMPLATES:
            target = tmp_path / template.template_id
            config_path, seeds_path = scaffold_template(template, target)
            assert seeds_path.is_file()
            config = PipelineConfig.from_yaml(config_path)
            assert config.task.name == template.template_id
            assert config.synthesis.target_samples == template.recommended_samples
            assert config.task.objective == template.objective
            if template.schema is not None:
                assert config.resolved_output_schema() is not None

    def test_cost_estimate_is_monotonic_and_explained(self) -> None:
        small = cost_estimate(100)
        large = cost_estimate(1000)
        assert large["total_cost_usd"] > small["total_cost_usd"]
        assert small["attempts_estimate"] > small["samples"]
        assert "estimate" in small["note"]
        assert cost_estimate(0)["total_cost_usd"] == 0

    def test_cost_prices_override(self) -> None:
        cheap = cost_estimate(100, prices={"prompt_per_mtok": 0.0, "completion_per_mtok": 0.0})
        assert cheap["total_cost_usd"] == 0

    def test_problem_guide_covers_common_failures(self) -> None:
        codes = {entry["code"] for entry in PROBLEM_GUIDE}
        for expected in ("api_key_missing", "vram_below_requirement", "secret_in_input"):
            assert expected in codes
        guidance = problem_guidance("api_key_missing")
        assert guidance is not None and guidance["fix"]
        assert problem_guidance("nope") is None

    def test_compliance_notes_are_present(self) -> None:
        topics = {note["topic"] for note in COMPLIANCE_NOTES}
        assert {"Telemetry", "Data residency", "Personal data", "Credentials"} <= topics

    def test_audit_report_and_exports(self, tmp_path: Path) -> None:
        dataset = tmp_path / "samples.jsonl"
        dataset.write_text('{"id": "a"}\n', encoding="utf-8")
        (tmp_path / "manifest.json").write_text(
            json.dumps({"run_id": "task-1", "config_fingerprint": "abc"}), encoding="utf-8"
        )
        report = audit_report(tmp_path)
        assert report["run_id"] == "task-1"
        assert report["dataset_present"] is True
        assert any(item["path"] == "samples.jsonl" for item in report["artifacts"])
        as_json = export_audit(tmp_path, tmp_path / "audit.json", fmt="json")
        assert json.loads(as_json.read_text(encoding="utf-8"))["tool"] == "Quarkon"
        as_csv = export_audit(tmp_path, tmp_path / "audit.csv", fmt="csv")
        rows = list(csv.reader(as_csv.read_text(encoding="utf-8").splitlines()))
        assert rows[0] == ["section", "key", "value"]
        assert any(row[0] == "artifact" for row in rows)
        with pytest.raises(ValueError):
            export_audit(tmp_path, tmp_path / "audit.xml", fmt="xml")


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _run_main(args: list[str]) -> tuple[int, str]:
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(args)
    return code, buffer.getvalue()


class TestCliAssist:
    def test_templates_json(self) -> None:
        code, out = _run_main(["--format", "json", "templates"])
        assert code == 0
        payload = json.loads(out)
        assert len(payload["templates"]) == len(TEMPLATES)

    def test_template_detail_json(self) -> None:
        code, out = _run_main(["--format", "json", "templates", "invoice-extract"])
        assert code == 0
        payload = json.loads(out)
        assert payload["template"]["id"] == "invoice-extract"

    def test_languages_json(self) -> None:
        code, out = _run_main(["--format", "json", "languages"])
        assert code == 0
        payload = json.loads(out)
        assert len(payload["languages"]) >= 20
        assert all(item["code"] != "ru" for item in payload["languages"])

    def test_cost_json(self) -> None:
        code, out = _run_main(["--format", "json", "cost", "--samples", "500"])
        assert code == 0
        payload = json.loads(out)
        assert payload["samples"] == 500
        assert payload["total_cost_usd"] > 0

    def test_setup_non_interactive_creates_project(self, tmp_path: Path) -> None:
        target = tmp_path / "project"
        code, out = _run_main(
            [
                "--format",
                "json",
                "setup",
                "--template",
                "support-triage",
                "--dir",
                str(target),
                "--samples",
                "120",
                "--non-interactive",
            ]
        )
        assert code == 0
        payload = json.loads(out)
        assert (target / "distill.yaml").is_file()
        assert payload["samples"] == 120
        from quarkon.core.config import PipelineConfig

        config = PipelineConfig.from_yaml(target / "distill.yaml")
        assert config.synthesis.target_samples == 120

    def test_doctor_without_project_explains_next_step(self, tmp_path: Path) -> None:
        code, out = _run_main(
            ["--format", "json", "--config", str(tmp_path / "missing.yaml"), "doctor"]
        )
        payload = json.loads(out)
        assert payload["command"] == "doctor"
        assert any(finding["code"] == "no_project" for finding in payload["findings"])
        assert code == 0

    def test_ui_help_lists_options(self) -> None:
        code, out = _run_main(["ui", "--help"])
        assert code == 0
        assert "--port" in out and "--no-browser" in out


# ---------------------------------------------------------------------------
# GUI server (no browser needed)
# ---------------------------------------------------------------------------


class _ServerThread:
    def __init__(self, project: Path) -> None:
        self.server = UIServer(project_dir=project, port=0, open_browser=False)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> tuple[UIServer, str]:
        self.thread.start()
        deadline = time.time() + 5
        while self.server.port == 0 and time.time() < deadline:
            time.sleep(0.01)
        return self.server, f"http://127.0.0.1:{self.server.port}"

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)


def _get(url: str, token: str | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(url)
    if token:
        request.add_header("X-Quarkon-Token", token)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _post(url: str, token: str, payload: dict[str, Any]) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Quarkon-Token": token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class TestUiServer:
    def test_static_assets_and_token_gate(self, tmp_path: Path) -> None:
        with _ServerThread(tmp_path) as (server, base):
            status, body = _get(base + "/")
            assert status == 200 and b"Quarkon" in body
            status, _ = _get(base + "/assets/app.js")
            assert status == 200
            status, _ = _get(base + "/api/bootstrap")
            assert status == 403  # no token
            status, body = _get(base + "/api/bootstrap", server.token)
            assert status == 200
            payload = json.loads(body)
            assert payload["version"]
            assert len(payload["locales"]) >= 20
            assert payload["labels"]["nav.setup"]

    def test_setup_plan_state_and_audit_endpoints(self, tmp_path: Path) -> None:
        with _ServerThread(tmp_path) as (server, base):
            status, body = _post(
                base + "/api/setup",
                server.token,
                {"template": "invoice-extract", "samples": 80, "language": "de"},
            )
            assert status == 200, body
            created = json.loads(body)
            assert Path(created["config_path"]).is_file()
            status, body = _post(base + "/api/plan", server.token, {"samples": 100})
            assert status == 200 and json.loads(body)["samples"] == 100
            status, body = _get(base + "/api/state", server.token)
            state = json.loads(body)
            assert state["has_project"] is True
            status, body = _post(base + "/api/audit/export", server.token, {"format": "json"})
            assert status == 200
            assert Path(json.loads(body)["path"]).is_file()
            status, _ = _post(base + "/api/run", server.token, {"phase": "not-a-phase"})
            assert status == 400

    def test_rejects_non_loopback_host(self, tmp_path: Path) -> None:
        from quarkon.core.exceptions import ConfigError

        with pytest.raises(ConfigError):
            UIServer(project_dir=tmp_path, host="0.0.0.0", open_browser=False)
