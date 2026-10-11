"""One-shot evaluation on the final holdout (2024 onward), with the design frozen.

Same procedure as the walk-forward experiment, extended: each calendar year of the holdout is a
fold, retrained on every row whose label was known before that year began (purged). Nothing is
tuned or chosen here; every table is the pre-declared one.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from riskplatform.features.compute import BASE_FEATURES
from riskplatform.models.learners import full_ladder
from riskplatform.risk import backtest as rb
from riskplatform.risk import stress as rs
from riskplatform.risk.engine import all_estimates
from riskplatform.strategy import overlay as ov

from .freeze import MANIFEST_NAME, RESULTS_NAME, holdout_already_run, verify
from .report import fold_mean_table, pairwise_tests
from .walkforward import evaluate, make_folds, run_walk_forward

NO_HOLDOUT_CUT = "2100-01-01"          # the harness's own holdout cut is disabled here, deliberately


def evaluate_models(frame: pd.DataFrame, returns: pd.Series, cfg: dict, specs: dict | None = None):
    v = cfg["validation"]
    first = int(v["final_holdout_start"][:4])
    last = int(frame.index.max().year)
    specs = specs or full_ladder(returns)
    result = run_walk_forward(frame, specs, make_folds(first, last, prefix="ho"), list(BASE_FEATURES),
                              v["modeling_start"], NO_HOLDOUT_CUT)
    return result, specs


def classification_tables(preds: pd.DataFrame, cfg: dict) -> dict:
    metrics = evaluate(preds)
    have = set(preds["model"].unique())
    ml = ("logit_tuned", "hist_gb", "random_forest")
    out = {"metrics": metrics, "fold_mean": fold_mean_table(metrics),
           "tests": pairwise_tests(preds, cfg["stats"])}
    if "garch_logit" in have:
        out["garch_tests"] = pairwise_tests(preds, cfg["stats"],
                                            pairs=[(m, "garch_logit") for m in ml if m in have])
    pos = preds[preds["model"] == "prevalence"].groupby("fold")["y"].agg(["sum", "count"])
    out["positives_per_fold"] = pos.rename(columns={"sum": "elevated_days", "count": "days"})
    return out


def overlay_tables(spy_ret, vix, cash, preds, cfg: dict, start, end) -> dict:
    o = cfg["overlay"]
    names = [o["primary_model"], o["secondary_model"]]
    probs = {m: preds[preds["model"] == m].set_index("as_of_date")["prob"] for m in names}
    w = ov.build_weights(spy_ret, vix, probs, o["target_vol"], o["max_weight"], o["ewma_lambda"])
    bt = ov.run_backtest(w, spy_ret, cash, o["execution_lag_days"], o["cost_bps_base"], start, end,
                         f"model_{o['primary_model']}")
    vs = ["buy_hold", "ewma_vol_target", "vix_target"]
    tests = pd.concat([ov.sharpe_tests(bt, f"model_{m}", vs, cfg["stats"]) for m in names])
    return {"performance": ov.performance(bt), "tests": tests, "returns": bt["returns"],
            "sensitivity": ov.sensitivity(spy_ret, vix, probs, cash, o, (start, end), o["primary_model"])}


def risk_tables(port: pd.Series, cfg: dict, start, end) -> pd.DataFrame:
    est = all_estimates(port, cfg["risk"], start)
    return rb.backtest_table(port, est, start, end)


def _md(df: pd.DataFrame, floatfmt: int = 4) -> str:
    return "```\n" + df.round(floatfmt).to_string() + "\n```\n"


def render_report(manifest: dict, tables: dict, meta: dict) -> str:
    c = tables["classification"]
    lines = [
        "# Final holdout results",
        "",
        f"*Run (UTC): {meta['run_utc']}  |  freeze_id: `{manifest['freeze_id']}`  |  freeze commit: `{manifest['git_sha']}`*",
        f"*Holdout: {meta['first_date']} to {meta['last_date']}; evaluated exactly once.*",
        "",
        "Every table below was specified before the holdout was evaluated. Nothing was tuned, "
        "selected, or changed after seeing these numbers. See the freeze manifest for the exact "
        "code, configuration, and model definitions.",
        "",
        "## Pre-declared hypotheses", "",
    ]
    lines += [f"- **{k}** {v}" for k, v in manifest["hypotheses"].items()]
    lines += ["", "## Power: elevated-volatility days per holdout fold", "", _md(c["positives_per_fold"], 0),
              "Few elevated days per fold means wide intervals; read the tests with that in mind.", "",
              "## Fold-averaged classification metrics (primary)", "", _md(c["fold_mean"]),
              "## Pre-declared comparisons: ML vs HAR / VIX (Holm-adjusted)", "", _md(c["tests"], 4)]
    if "garch_tests" in c:
        lines += ["## ML vs GARCH (own family)", "", _md(c["garch_tests"], 4)]
    o = tables["overlay"]
    lines += ["## Decision-level overlay (net of costs)", "", _md(o["performance"]),
              "### Sharpe differences vs baselines", "", _md(o["tests"].set_index(["model_rule", "vs"])),
              "### Net Sharpe across lag / cost / target-volatility settings", "", _md(o["sensitivity"], 3)]
    lines += ["## VaR / ES backtest on the holdout", "", _md(tables["risk"].drop(
        columns=["basel_green", "basel_yellow", "basel_red"])),
              f"Low power: at 99% only about {tables['risk'].query('conf == 0.99')['expected'].iloc[0]} "
              "exceptions are expected, so these tests can reject only gross miscalibration.", ""]
    return "\n".join(lines)


def run(engine, cfg: dict, repo: Path, pkg_root: Path, manifest_path: Path, check_only: bool = False) -> list[str]:
    """Verify the freeze; if intact and unused, evaluate the holdout once. Returns problems (if any)."""
    from sqlalchemy import text

    from riskplatform.features.pit import pit_daily
    from riskplatform.features.store import finish_run, load_macro, load_prices, start_run
    from riskplatform.risk.stress import portfolio_returns

    from .freeze import manifest_committed
    from .store import frame_hash, load_modeling_frame, write_walk_forward

    results_path = manifest_path.parent / RESULTS_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    frame = load_modeling_frame(engine, cfg)
    hs = pd.Timestamp(cfg["validation"]["final_holdout_start"])
    pre = frame[frame.index < hs]
    returns = frame["ret_1d"]
    specs = full_ladder(pd.Series(dtype=float))                   # definitions only, for comparison
    problems = verify(manifest, cfg, specs, pkg_root, frame_hash(pre))
    if manifest is not None and manifest.get("git_sha"):
        problems += manifest_committed(repo, manifest_path, manifest["git_sha"])
    with engine.connect() as conn:
        done = conn.execute(text("select count(*) from ops.pipeline_run where pipeline_name = 'holdout' "
                                 "and status = 'succeeded'")).scalar_one()
    if holdout_already_run(results_path, done):
        problems.append("the holdout has already been evaluated; it cannot be run a second time")
    if problems or check_only:
        return problems

    run_id = start_run(engine, "holdout")
    try:
        result, specs = evaluate_models(frame, returns, cfg)
        metrics = evaluate(result.preds)
        cls = classification_tables(result.preds, cfg)
        first, last = frame.index[frame.index >= hs].min(), result.preds["as_of_date"].max()

        o = cfg["overlay"]
        prices = load_prices(engine, [o["instrument"], "^VIX"] + sorted(cfg["risk"]["portfolio"]))
        spy = prices[o["instrument"]]["adj_close"]
        cash = pit_daily(load_macro(engine, ["DFF"])["DFF"], spy.index).shift(1) / 100 / 252
        ov_t = overlay_tables(spy.pct_change(), prices["^VIX"]["close"], cash, result.preds, cfg, first, last)
        adj = pd.DataFrame({t: prices[t]["adj_close"] for t in cfg["risk"]["portfolio"]})
        port = portfolio_returns(adj.pct_change(), cfg["risk"]["portfolio"]).dropna()
        rk = risk_tables(port, cfg, first, last)

        report = render_report(manifest, {"classification": cls, "overlay": ov_t, "risk": rk},
                               {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "first_date": str(first.date()), "last_date": str(last.date())})
        n = write_walk_forward(engine, cfg, specs, result.preds, metrics, result.fold_info, result.trials,
                               frame_hash(frame), run_id, purpose="final_holdout")
        out = repo / "artifacts" / "holdout"
        out.mkdir(parents=True, exist_ok=True)
        result.preds.to_csv(out / "predictions.csv", index=False)
        ov_t["returns"].to_csv(out / "overlay_returns.csv")
        rk.to_csv(out / "var_backtest.csv", index=False)
        results_path.write_text(report, encoding="utf-8")
        finish_run(engine, run_id, n)
    except Exception as exc:
        finish_run(engine, run_id, 0, error=f"{type(exc).__name__}: {exc}")
        raise
    return []
