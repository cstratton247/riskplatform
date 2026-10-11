import copy
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskplatform.models.benchmarks import default_ladder
from riskplatform.models.garch import garch_logit
from riskplatform.risk.stress import joint_shock_loss
from riskplatform.validation import freeze as fz
from riskplatform.validation import holdout as ho
from test_walkforward import HOLDOUT, MODELING_START, _frame

CFG = {
    "data": {"start_date": "2004-01-01"}, "target": {"asset": "^GSPC"}, "dq": {},
    "validation": {"modeling_start": MODELING_START, "final_holdout_start": HOLDOUT},
    "stats": {"bootstrap_draws": 100, "mean_block_length": 10, "dm_horizon": 5, "seed": 1},
    "overlay": {"primary_model": "har_logit", "secondary_model": "vix_logit", "target_vol": 0.10,
                "max_weight": 1.0, "ewma_lambda": 0.94, "execution_lag_days": 2, "cost_bps_base": 5,
                "lag_sensitivity": [1, 2], "cost_bps_sensitivity": [0, 5], "target_vol_sensitivity": [0.10]},
    "risk": {"confidence_levels": [0.95, 0.99], "window": 250, "ewma_lambda": 0.94,
             "student_t_df": 5, "garch_min_fit_obs": 300, "portfolio": {"SPY": 1.0}},
}


def _pkg(tmp: Path) -> Path:
    root = tmp / "pkg"
    for d in fz.FROZEN_DIRS + ["api"]:
        (root / d).mkdir(parents=True)
        (root / d / "mod.py").write_text(f"X = '{d}'\n")
    return root


def _specs():
    s = default_ladder()
    s["garch_logit"] = garch_logit(pd.Series(dtype=float))
    return s


def _manifest(root, cfg=CFG, specs=None, frame_hash="abc"):
    return fz.build_manifest(cfg, specs or _specs(), root, frame_hash, 100, 10, {"sha": "deadbeef", "dirty": False})


# --------------------------------------------------------------------------- freeze integrity
def test_intact_freeze_has_no_problems_and_each_tamper_is_caught():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        root = _pkg(Path(t))
        m = _manifest(root)
        assert fz.verify(m, CFG, _specs(), root, "abc") == []

        (root / "models" / "mod.py").write_text("X = 'changed'\n")
        assert any("modified since freeze: models/mod.py" in p for p in fz.verify(m, CFG, _specs(), root, "abc"))
        (root / "models" / "mod.py").write_text("X = 'models'\n")
        (root / "risk" / "new.py").write_text("pass\n")
        assert any("added since freeze: risk/new.py" in p for p in fz.verify(m, CFG, _specs(), root, "abc"))
        (root / "risk" / "new.py").unlink()

        cfg2 = copy.deepcopy(CFG)
        cfg2["overlay"]["cost_bps_base"] = 1
        assert any("config" in p for p in fz.verify(m, cfg2, _specs(), root, "abc"))
        assert any("pre-holdout data changed" in p for p in fz.verify(m, CFG, _specs(), root, "different"))
        assert any("model definitions" in p for p in fz.verify(m, CFG, default_ladder(), root, "abc"))
        tampered = copy.deepcopy(m)
        tampered["n_pre_holdout_rows"] = 1
        assert any("freeze_id" in p for p in fz.verify(tampered, CFG, _specs(), root, "abc"))
        assert fz.verify(None, CFG, _specs(), root, "abc")[0].startswith("no freeze manifest")


def test_code_hash_ignores_line_endings_and_non_frozen_dirs_but_config_hash_ignores_other_sections():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        root = _pkg(Path(t))
        before = fz.code_hash(fz.file_hashes(root))
        (root / "features" / "mod.py").write_bytes(b"X = 'features'\r\n")          # same text, CRLF
        assert fz.code_hash(fz.file_hashes(root)) == before
        (root / "api" / "mod.py").write_text("X = 'api changed'\n")                  # not a frozen dir
        assert fz.code_hash(fz.file_hashes(root)) == before
    other = copy.deepcopy(CFG)
    other["api"] = {"port": 8000}                                                      # unfrozen section
    assert fz.config_hash(other) == fz.config_hash(CFG)


def test_single_use_guard():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        path = Path(t) / "r.md"
        assert not fz.holdout_already_run(path, 0)
        assert fz.holdout_already_run(path, 1)
        path.write_text("x")
        assert fz.holdout_already_run(path, 0)


def _has_git():
    return shutil.which("git") is not None


def test_git_state_and_commit_checks():
    if not _has_git():
        pytest.importorskip("a_module_that_does_not_exist_to_skip")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        repo = Path(t)
        run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)
        run("init", "-q")
        run("config", "user.email", "t@example.com")
        run("config", "user.name", "t")
        (repo / "freeze").mkdir()
        mp = repo / "freeze" / fz.MANIFEST_NAME
        mp.write_text("{}")
        run("add", "-A")
        run("-c", "commit.gpgsign=false", "commit", "-qm", "one")
        st = fz.git_state(repo)
        assert st["dirty"] is False and len(st["sha"]) == 40
        assert fz.manifest_committed(repo, mp, st["sha"]) == []
        mp.write_text('{"edited": 1}')
        assert fz.git_state(repo)["dirty"] is True
        assert any("uncommitted" in p for p in fz.manifest_committed(repo, mp, st["sha"]))


# --------------------------------------------------------------------------- holdout mechanics
def _eval():
    frame = _frame()
    returns = frame["ret_1d"]
    specs = {k: v for k, v in _specs().items() if k in ("prevalence", "har_logit", "vix_logit")}
    specs["garch_logit"] = garch_logit(returns)
    result, _ = ho.evaluate_models(frame, returns, CFG, specs)
    return frame, returns, specs, result


def test_holdout_folds_cover_only_the_holdout_and_ignore_own_labels():
    frame, returns, specs, result = _eval()
    p = result.preds
    assert set(p["fold"]) == {"ho_2024", "ho_2025"}
    assert p["as_of_date"].min() >= pd.Timestamp(HOLDOUT)
    mutated = frame.copy()
    in24 = (mutated.index >= pd.Timestamp("2024-01-01")) & (mutated.index <= pd.Timestamp("2024-12-31"))
    mutated.loc[in24, "y"] = (np.random.default_rng(3).random(in24.sum()) < 0.5).astype(int)
    again, _ = ho.evaluate_models(mutated, returns, CFG, {k: v for k, v in specs.items() if k != "garch_logit"})
    a = p[(p["fold"] == "ho_2024") & p["model"].isin(["har_logit", "vix_logit", "prevalence"])]
    b = again.preds[again.preds["fold"] == "ho_2024"]
    np.testing.assert_allclose(a["prob"].to_numpy(), b["prob"].to_numpy())      # 2024 predictions never see 2024 labels


def test_tables_and_report_render_end_to_end():
    frame, returns, specs, result = _eval()
    cls = ho.classification_tables(result.preds, CFG)
    assert {"metrics", "fold_mean", "tests", "positives_per_fold"} <= set(cls)
    first, last = frame.index[frame.index >= HOLDOUT].min(), result.preds["as_of_date"].max()
    cash = pd.Series(0.0001, index=frame.index)
    ov_t = ho.overlay_tables(frame["ret_1d"], frame["vix_level"], cash, result.preds, CFG, first, last)
    assert {"performance", "tests", "sensitivity"} <= set(ov_t) and len(ov_t["tests"]) == 6
    port = frame["ret_1d"]
    rk = ho.risk_tables(port, CFG, first, last)
    assert set(rk["method"]) == {"historical", "normal_ewma", "student_t_ewma", "fhs_gjr_garch"}
    manifest = {"freeze_id": "x" * 8, "git_sha": "abc", "hypotheses": fz.HYPOTHESES}
    text = ho.render_report(manifest, {"classification": cls, "overlay": ov_t, "risk": rk},
                            {"run_utc": "now", "first_date": str(first.date()), "last_date": str(last.date())})
    assert text.startswith("# Final holdout results") and "Low power" in text and "H3" in text


def test_joint_shock_loss():
    w = {"SPY": 0.6, "IEF": 0.4}
    assert np.isclose(joint_shock_loss(w, {"SPY": -0.20, "IEF": -0.10}), -0.16)
    assert np.isclose(joint_shock_loss(w, {"SPY": -0.20, "IEF": 0.05}), -0.10)
    assert joint_shock_loss(w, {}) == 0.0
