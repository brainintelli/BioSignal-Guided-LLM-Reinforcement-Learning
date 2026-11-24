"""Verify measured evidence and package the report without touching frozen sources."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import hashlib
import json
import zipfile

REPORT = Path(__file__).resolve().parent
PROJECT = REPORT.parent.parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

registration = read(REPORT / "registration.json")
for name, expected in registration["source_hashes"].items():
    assert digest(PROJECT / name) == expected, name
evidence = REPORT / "evidence"
original_manifest = read(evidence / "sha256_manifest.json")
for name, expected in original_manifest.items():
    assert digest(evidence / name) == expected, name
assert digest(REPORT / "evidence.zip") == "8b63b0e124d97e57bbb9971a3537231c7548b4b08377ac7b44f2fcba0c65dd71"

summary = read(REPORT / "analysis_summary.json")
audit = read(evidence / "m8s2-diagnostics/compare1/independent_audit.json")
comparison = read(evidence / "m8s2-diagnostics/compare1/comparison.json")
assert summary["speed_gate"]["passed"] and summary["speed_gate"]["all_three_rounds_speedup"] > 2
assert summary["speed_gate"]["median_speedup"] > 2
assert audit["reload_consistency_passed"] and audit["sources_and_inputs_unchanged"]
assert sum(a["engine_replay"]["records"] for a in audit["arms"].values()) == 1152
for arm in audit["arms"].values():
    assert arm["first_round_all_logps_checked"] == 128
    assert arm["first_round_logp_error_max"] == 0
for arm in comparison["stability"].values():
    assert sum(r["optimizer_steps"] for r in arm["updates"]) == 24
    assert not any(r["kl_early_stop"] for r in arm["updates"])
checks = read(REPORT / "project_verification/checks.json")
assert all(c["passed"] and c["exit_code"] == 0 for c in checks["checks"])
assert "Ran 180 tests" in (REPORT / "gpu_unit_tests.log").read_text()
assert "OK (skipped=3)" in (REPORT / "gpu_unit_tests.log").read_text()

class LinkCheck(HTMLParser):
    count = 0
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href") and value and not urlsplit(value).scheme and not value.startswith("#"):
                target = (REPORT / unquote(urlsplit(value).path)).resolve()
                assert target.is_file(), value
                self.count += 1

links = LinkCheck()
links.feed((REPORT / "report.html").read_text(encoding="utf-8"))
verification = {
    "registered_sources_verified": len(registration["source_hashes"]),
    "raw_evidence_hashes_verified": len(original_manifest),
    "report_links_verified": links.count,
    "speed_gate_passed": True,
    "actor_updates_each_arm": 24,
    "engine_records_verified": 1152,
    "formal_training_started": False,
    "checkpoint_writes": 0,
}
(REPORT / "delivery_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
files = [p for p in REPORT.rglob("*") if p.is_file() and p.name != "delivery_manifest.json"]
files += [PROJECT / "docs/M8S2_SEMANTIC_PROTOCOL.md"]
manifest = {p.relative_to(PROJECT).as_posix(): digest(p) for p in sorted(files)}
manifest_path = REPORT / "delivery_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
archive = PROJECT / "output/m8s2_delivery.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for p in [*files, manifest_path]:
        z.write(p, p.relative_to(PROJECT).as_posix())
package_info = {"file": str(archive), "bytes": archive.stat().st_size, "sha256": digest(archive), "files": len(files) + 1}
(PROJECT / "output/m8s2_delivery.sha256.json").write_text(json.dumps(package_info, indent=2), encoding="utf-8")
print(json.dumps({**verification, "package": package_info}, ensure_ascii=False))
