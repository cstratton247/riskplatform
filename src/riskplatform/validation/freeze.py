"""Design freeze: a tamper-evident record of EXACTLY what will be evaluated on the final holdout.

`riskplatform freeze` writes freeze/freeze_manifest.json (to be committed). `riskplatform holdout`
refuses to run unless the code, model definitions, frozen config sections and pre-holdout data
still match it. This is an audit trail and a discipline mechanism: it makes accidental or
convenient changes impossible to hide, not impossible to make.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FROZEN_DIRS = ["features", "models", "validation", "strategy", "risk", "ingest"]
FROZEN_CONFIG_SECTIONS = ["data", "target", "validation", "overlay", "risk", "stats", "dq"]
MANIFEST_NAME = "freeze_manifest.json"
RESULTS_NAME = "holdout_results.md"

HYPOTHESES = {
    "H1": "Best ML models beat the HAR-logit benchmark on fold-averaged PR-AUC (Holm-adjusted, 95% block bootstrap).",
    "H2": "Best ML models beat the VIX-only benchmark on fold-averaged PR-AUC.",
    "H3": "A model-driven exposure overlay (random forest primary) improves net Sharpe versus the "
          "trailing-volatility (EWMA) overlay, 5 bps costs, 2-day execution lag.",
    "H4": "ML models beat the GJR-GARCH benchmark (added after the walk-forward results were seen; reported as its own family).",
}


def _normalize(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")                    # Windows checkouts must hash like Linux ones


def file_hashes(pkg_root: Path) -> dict[str, str]:
    out = {}
    for d in FROZEN_DIRS:
        for f in sorted((pkg_root / d).rglob("*.py")):
            rel = f.relative_to(pkg_root).as_posix()
            out[rel] = hashlib.sha256(_normalize(f.read_bytes())).hexdigest()
    return out


def code_hash(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def config_hash(cfg: dict) -> str:
    frozen = {k: cfg[k] for k in FROZEN_CONFIG_SECTIONS if k in cfg}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()


def spec_summary(specs: dict) -> dict:
    return {n: {"family": s.family, "features": list(s.features), "hyperparameters": s.hyperparameters}
            for n, s in sorted(specs.items())}


def manifest_id(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "freeze_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _git(repo: Path, *args: str):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=repo)


def git_state(repo: Path) -> dict | None:
    try:
        sha = _git(repo, "rev-parse", "HEAD")
        status = _git(repo, "status", "--porcelain")
    except FileNotFoundError:
        return None
    if sha.returncode != 0:
        return None
    return {"sha": sha.stdout.strip(), "dirty": bool(status.stdout.strip())}


def manifest_committed(repo: Path, manifest_path: Path, freeze_sha: str) -> list[str]:
    rel = manifest_path.relative_to(repo).as_posix()
    problems = []
    if _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        problems.append(f"{rel} is not tracked by git; commit it before running the holdout")
    elif _git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        problems.append(f"{rel} has uncommitted changes")
    if _git(repo, "merge-base", "--is-ancestor", freeze_sha, "HEAD").returncode != 0:
        problems.append("the freeze commit is not an ancestor of HEAD")
    return problems


def build_manifest(
    cfg: dict, specs: dict, pkg_root: Path, pre_holdout_frame_hash: str, n_pre_holdout_rows: int,
    n_holdout_rows_present: int, git: dict | None, spec_doc: Path | None = None,
) -> dict:
    hashes = file_hashes(pkg_root)
    manifest = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git["sha"] if git else None,
        "git_dirty_at_freeze": git["dirty"] if git else None,
        "frozen_config_sections": {k: cfg[k] for k in FROZEN_CONFIG_SECTIONS if k in cfg},
        "config_hash": config_hash(cfg),
        "code_files": hashes,
        "code_hash": code_hash(hashes),
        "models": spec_summary(specs),
        "hypotheses": HYPOTHESES,
        "pre_holdout_frame_hash": pre_holdout_frame_hash,
        "n_pre_holdout_rows": n_pre_holdout_rows,
        "n_holdout_rows_present_at_freeze": n_holdout_rows_present,
        "holdout_start": cfg["validation"]["final_holdout_start"],
        "spec_doc_sha256": (hashlib.sha256(_normalize(spec_doc.read_bytes())).hexdigest()
                            if spec_doc and spec_doc.exists() else None),
    }
    manifest["freeze_id"] = manifest_id(manifest)
    return manifest


def verify(
    manifest: dict | None, cfg: dict, specs: dict, pkg_root: Path, pre_holdout_frame_hash: str | None,
) -> list[str]:
    """Returns a list of problems; empty means the frozen design is intact."""
    if manifest is None:
        return ["no freeze manifest found; run `riskplatform freeze` and commit freeze/"]
    problems = []
    if manifest_id(manifest) != manifest.get("freeze_id"):
        problems.append("manifest contents do not match its freeze_id (file was edited)")
    if config_hash(cfg) != manifest["config_hash"]:
        problems.append("frozen config sections changed since the freeze")
    now = file_hashes(pkg_root)
    for rel in sorted(set(now) | set(manifest["code_files"])):
        if now.get(rel) != manifest["code_files"].get(rel):
            kind = "added" if rel not in manifest["code_files"] else "removed" if rel not in now else "modified"
            problems.append(f"code {kind} since freeze: {rel}")
    if json.loads(json.dumps(spec_summary(specs), default=str)) != json.loads(json.dumps(manifest["models"], default=str)):
        problems.append("model definitions differ from the frozen ones")
    if pre_holdout_frame_hash is not None and pre_holdout_frame_hash != manifest["pre_holdout_frame_hash"]:
        problems.append("pre-holdout data changed since the freeze (vendor restatement or a reload); "
                        "reload data BEFORE freezing")
    return problems


def holdout_already_run(results_path: Path, succeeded_runs_in_db: int) -> bool:
    return results_path.exists() or succeeded_runs_in_db > 0
