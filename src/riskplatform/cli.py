"""Command-line entry point: `riskplatform <command>`."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from riskplatform.config import load_config
from riskplatform.db import get_engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="riskplatform")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("init-calendar", help="load NYSE trading days into ref.trading_day")
    cal.add_argument("--start", type=date.fromisoformat)
    cal.add_argument("--end", type=date.fromisoformat)

    lp = sub.add_parser("load-prices", help="full-history price refresh with DQ checks")
    lp.add_argument("--tickers", nargs="+")
    lp.add_argument("--start", type=date.fromisoformat)
    lp.add_argument("--end", type=date.fromisoformat, help="exclusive; defaults to today")

    lm = sub.add_parser("load-macro", help="load FRED/ALFRED series into macro.observation_vintage")
    lm.add_argument("--series", nargs="+")

    sub.add_parser("build-features", help="compute features and write them to feat.feature_value")
    bl = sub.add_parser("build-labels", help="compute labels and write them to ml.label")
    bl.add_argument("--quantile", type=float, help="override target.threshold_quantile (e.g. 0.90)")

    wf = sub.add_parser("walk-forward", help="run the benchmark ladder over purged walk-forward folds")
    wf.add_argument("--no-db", action="store_true", help="print results without writing to the database")
    wf.add_argument("--quantile", type=float, help="override target.threshold_quantile (e.g. 0.90)")
    wf.add_argument("--ladder-only", action="store_true", help="skip tuned models (fast)")

    cp = sub.add_parser("compare", help="significance tests and fold-averaged summary from stored predictions")
    cp.add_argument("--quantile", type=float, help="override target.threshold_quantile (e.g. 0.90)")

    fz = sub.add_parser("freeze", help="record the frozen design (commit freeze/ afterwards)")
    fz.add_argument("--allow-dirty", action="store_true", help="freeze even with uncommitted changes (recorded)")
    fz.add_argument("--verify", action="store_true", help="only check an existing freeze; write nothing")
    ho = sub.add_parser("holdout", help="evaluate the final holdout ONCE, if the freeze is intact")
    ho.add_argument("--check", action="store_true", help="verify the freeze without touching the holdout")
    sub.add_parser("risk", help="VaR/ES methods, backtests (Kupiec, Christoffersen, Basel) and stress tests")
    sub.add_parser("overlay", help="decision-level backtest: model-driven exposure vs baselines, net of costs")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    if getattr(args, "quantile", None):
        cfg["target"]["threshold_quantile"] = args.quantile
    engine = get_engine()

    if args.command == "init-calendar":
        from riskplatform.ingest.calendar import load_calendar

        start = args.start or date.fromisoformat(cfg["data"]["start_date"])
        end = args.end or date.today()
        n = load_calendar(engine, start, end)
        print(f"Loaded {n} trading days ({start} to {end})")
        return 0

    if args.command == "load-macro":
        import os

        from riskplatform.ingest.macro import load_macro

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            print("FRED_API_KEY is not set in .env", file=sys.stderr)
            return 2
        ok = load_macro(engine, cfg, args.series, api_key)
        print("All series loaded." if ok else "Finished with failures; see ops.pipeline_run.")
        return 0 if ok else 1

    if args.command == "build-features":
        from riskplatform.features.compute import compute_features
        from riskplatform.features.store import (
            finish_run, load_feature_inputs, start_run, write_features,
        )

        run_id = start_run(engine, "build_features")
        try:
            feats = compute_features(load_feature_inputs(engine))
            n = write_features(engine, feats, cfg["target"]["asset"], run_id)
        except Exception as exc:
            finish_run(engine, run_id, 0, error=f"{type(exc).__name__}: {exc}")
            raise
        finish_run(engine, run_id, n)
        print(f"Wrote {n} feature values ({feats.shape[1]} features, {feats.shape[0]} days)")
        return 0

    if args.command == "build-labels":
        from riskplatform.features.store import build_and_write_labels, finish_run, start_run

        run_id = start_run(engine, "build_labels")
        try:
            n = build_and_write_labels(engine, cfg)
        except Exception as exc:
            finish_run(engine, run_id, 0, error=f"{type(exc).__name__}: {exc}")
            raise
        finish_run(engine, run_id, n)
        print(f"Wrote {n} label rows")
        return 0

    if args.command == "walk-forward":
        import pandas as pd

        from riskplatform.features.compute import BASE_FEATURES
        from riskplatform.features.store import finish_run, start_run
        from riskplatform.models.benchmarks import default_ladder
        from riskplatform.models.learners import full_ladder
        from riskplatform.validation.report import fold_mean_table
        from riskplatform.validation.store import frame_hash, load_modeling_frame, write_walk_forward
        from riskplatform.validation.walkforward import evaluate, make_folds, run_walk_forward

        v = cfg["validation"]
        frame = load_modeling_frame(engine, cfg)
        returns = frame["ret_1d"][frame.index < pd.Timestamp(v["final_holdout_start"])]
        if args.ladder_only:
            from riskplatform.models.garch import garch_logit

            specs = default_ladder()
            specs["garch_logit"] = garch_logit(returns)
        else:
            specs = full_ladder(returns)
        folds = make_folds(v["test_years"][0], v["test_years"][1])
        result = run_walk_forward(frame, specs, folds, list(BASE_FEATURES),
                                  v["modeling_start"], v["final_holdout_start"])
        metrics = evaluate(result.preds)

        pd.set_option("display.width", 160)
        print("\nPRIMARY: metrics averaged over folds (%d-%d), each year weighted equally:" % tuple(v["test_years"]))
        print(fold_mean_table(metrics).round(4).to_string())
        print(f"\nConfigurations tried (inner CV): {len(result.trials)} fits logged "
              f"across {result.trials['model'].nunique()} tuned models")

        if not args.no_db:
            run_id = start_run(engine, "walk_forward")
            try:
                n = write_walk_forward(engine, cfg, specs, result.preds, metrics, result.fold_info,
                                       result.trials, frame_hash(frame), run_id)
            except Exception as exc:
                finish_run(engine, run_id, 0, error=f"{type(exc).__name__}: {exc}")
                raise
            finish_run(engine, run_id, n)
            print(f"Saved {n} predictions, metrics and trials. Next: riskplatform compare")
        return 0

    if args.command == "compare":
        import pandas as pd

        from riskplatform.validation.report import fold_mean_table, pairwise_tests
        from riskplatform.validation.store import load_predictions
        from riskplatform.validation.walkforward import evaluate

        preds = load_predictions(engine, cfg)
        pd.set_option("display.width", 200)
        print("\nPRIMARY: fold-averaged metrics:")
        print(fold_mean_table(evaluate(preds)).round(4).to_string())
        print("\nPre-declared comparisons (ML vs HAR / VIX benchmarks); Holm-adjusted p-values.")
        print("pr_auc_diff = fold-mean PR-AUC difference (positive favors `model`); 95% block-bootstrap CI.")
        print("logloss_diff < 0 favors `model`; DM test with HAC variance.\n")
        tests = pairwise_tests(preds, cfg["stats"])
        print(tests.round(4).to_string(index=False) if not tests.empty else "(no ML models found)")
        have = set(preds["model"].unique())
        if "garch_logit" in have:
            gp = [(m, "garch_logit") for m in ("logit_tuned", "hist_gb", "random_forest") if m in have]
            print("\nAdditional benchmark added after the first results were seen: GARCH. "
                  "Reported as its own family (Holm within these three).\n")
            print(pairwise_tests(preds, cfg["stats"], pairs=gp).round(4).to_string(index=False))
        return 0

    if args.command in ("freeze", "holdout"):
        import json
        from pathlib import Path

        import pandas as pd

        import riskplatform
        from riskplatform.models.learners import full_ladder
        from riskplatform.validation import freeze as fz_mod
        from riskplatform.validation import holdout as ho_mod
        from riskplatform.validation.store import frame_hash, load_modeling_frame

        pkg_root = Path(riskplatform.__file__).resolve().parent
        repo = pkg_root.parents[1]
        manifest_path = repo / "freeze" / fz_mod.MANIFEST_NAME

        if args.command == "holdout":
            problems = ho_mod.run(engine, cfg, repo, pkg_root, manifest_path, check_only=args.check)
            if problems:
                print("Holdout NOT run. Problems:")
                print("\n".join(f"  - {p}" for p in problems))
                return 1
            if args.check:
                print("Freeze intact; the holdout has not been run. Safe to run `riskplatform holdout`.")
            else:
                print(f"Holdout evaluated once. Report: {manifest_path.parent / fz_mod.RESULTS_NAME}")
                print("Commit freeze/ and artifacts/holdout/ as the record.")
            return 0

        frame = load_modeling_frame(engine, cfg)
        hs = pd.Timestamp(cfg["validation"]["final_holdout_start"])
        pre = frame[frame.index < hs]
        specs = full_ladder(pd.Series(dtype=float))
        if args.verify:
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
            problems = fz_mod.verify(manifest, cfg, specs, pkg_root, frame_hash(pre))
            print("Freeze intact." if not problems else "Freeze BROKEN:\n" + "\n".join(f"  - {p}" for p in problems))
            return 1 if problems else 0
        git = fz_mod.git_state(repo)
        if git and git["dirty"] and not args.allow_dirty:
            print("Uncommitted changes present. Commit them first (or pass --allow-dirty; it is recorded).")
            return 1
        manifest = fz_mod.build_manifest(cfg, specs, pkg_root, frame_hash(pre), len(pre),
                                         int((frame.index >= hs).sum()), git, repo / "docs" / "milestone1_spec.md")
        manifest_path.parent.mkdir(exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(f"Freeze written: {manifest_path}\nfreeze_id {manifest['freeze_id']}")
        print("Next: git add freeze/ && git commit -m 'Freeze design' && git tag freeze-v1")
        return 0

    if args.command == "risk":
        import time
        from pathlib import Path

        import numpy as np
        import pandas as pd

        from riskplatform.features.pit import pit_daily
        from riskplatform.features.store import load_macro, load_prices
        from riskplatform.models.garch import fit_gjr, next_variance, variance_path
        from riskplatform.risk import backtest as rb
        from riskplatform.risk import stress as rs
        from riskplatform.risk import var_es as ve

        rc = cfg["risk"]
        w = rc["portfolio"]
        end = pd.Timestamp(rc["backtest_end"])
        tickers = sorted(set(w) | {"SPY"})
        prices = load_prices(engine, tickers)
        adj = pd.DataFrame({t: prices[t]["adj_close"] for t in tickers})
        rets = adj.pct_change().loc[:end]
        port = rs.portfolio_returns(rets, w).dropna()
        confs = rc["confidence_levels"]

        from riskplatform.risk.engine import all_estimates

        est = all_estimates(port, rc, rc["backtest_start"])
        table = rb.backtest_table(port, est, rc["backtest_start"], rc["backtest_end"])

        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 30)
        print(f"\nPortfolio {w}, 1-day VaR/ES backtest {rc['backtest_start']} to {rc['backtest_end']}")
        show = table.drop(columns=["basel_green", "basel_yellow", "basel_red"]).round(4)
        print(show.to_string(index=False))
        print("\nBasel traffic light, 99% VaR, share of rolling 250-day windows:")
        print(table[table["conf"] == 0.99][["method", "basel_green", "basel_yellow", "basel_red"]].round(3).to_string(index=False))

        macro = load_macro(engine, ["DFF"])["DFF"]
        cash = pit_daily(macro, port.index).shift(1) / 100 / 252
        print("\nPortfolio metrics (through backtest_end):")
        print(rs.portfolio_metrics(rets[list(w)], w, rets["SPY"], cash).round(4).to_string())
        print("\nHistorical stress replays:")
        print(rs.replay(port, rets["SPY"], rc["stress"]["replays"]).round(4).to_string())

        b = rs.betas(rets[list(w)], rets["SPY"])
        st = rc["stress"]
        cov = rets[list(w)].dropna().cov()
        base = rs.stressed_normal_var_es(cov, w, 0.99)
        hyp = rs.stressed_normal_var_es(cov, w, 0.99, st["vol_multiplier"], st["correlation_blend"])
        print(f"\nHypothetical: market {st['equity_shock']:.0%} with beta-scaled asset moves -> portfolio "
              f"{rs.equity_shock_loss(w, b, st['equity_shock']):.2%}  (betas: {b.round(2).to_dict()})")
        for name, shocks in st.get("joint_shocks", {}).items():
            print(f"Joint shock '{name}' {shocks} -> portfolio {rs.joint_shock_loss(w, shocks):.2%}")
        print(f"Gaussian 1-day 99%  baseline: VaR {base['var']:.2%} ES {base['es']:.2%}  |  "
              f"vol x{st['vol_multiplier']}, correlation blend {st['correlation_blend']}: "
              f"VaR {hyp['var']:.2%} ES {hyp['es']:.2%}")

        x = port.to_numpy() * 100
        params = fit_gjr(x)
        h0 = float(np.var(x))
        h = variance_path(x, params, h0)
        resid = (x / np.sqrt(h))[-rc["window"]:]
        h_next = float(next_variance(x, params, h)[-1])
        t0 = time.perf_counter()
        mv, me = ve.gjr_monte_carlo(params, h_next, resid, 0.99, rc["mc_horizon_days"], rc["mc_paths"])
        print(f"\nGJR-GARCH Monte Carlo ({rc['mc_paths']:,} paths, {rc['mc_horizon_days']}-day) from {port.index[-1].date()}: "
              f"VaR99 {mv:.2%}  ES99 {me:.2%}  [{time.perf_counter() - t0:.2f}s]")

        out = Path("artifacts"); out.mkdir(exist_ok=True)
        table.to_csv(out / "var_backtest.csv", index=False)
        print("Backtest table saved to artifacts/var_backtest.csv.")
        return 0

    if args.command == "overlay":
        from pathlib import Path

        import pandas as pd

        from riskplatform.features.pit import pit_daily
        from riskplatform.features.store import load_macro, load_prices
        from riskplatform.strategy import overlay as ov
        from riskplatform.validation.store import load_predictions

        o, v = cfg["overlay"], cfg["validation"]
        prices = load_prices(engine, [o["instrument"], "^VIX"])
        spy = prices[o["instrument"]]["adj_close"]
        spy_ret = spy.pct_change()
        vix = prices["^VIX"]["close"]
        cash = pit_daily(load_macro(engine, ["DFF"])["DFF"], spy.index).shift(1) / 100 / 252
        preds = load_predictions(engine, cfg)
        names = [o["primary_model"], o["secondary_model"]]
        probs = {m: preds[preds["model"] == m].set_index("as_of_date")["prob"] for m in names}
        window = (f"{v['test_years'][0]}-01-01", f"{v['test_years'][1]}-12-31")

        w = ov.build_weights(spy_ret, vix, probs, o["target_vol"], o["max_weight"], o["ewma_lambda"])
        bt = ov.run_backtest(w, spy_ret, cash, o["execution_lag_days"], o["cost_bps_base"],
                             *window, f"model_{o['primary_model']}")
        pd.set_option("display.width", 200)
        n = len(bt["returns"])
        print(f"\nBase case: {bt['returns'].index[0].date()} to {bt['returns'].index[-1].date()} ({n} days), "
              f"{o['cost_bps_base']} bps costs, {o['execution_lag_days']}-day lag, target vol {o['target_vol']:.0%}, no leverage")
        print(ov.performance(bt).round(4).to_string())
        print("\nEpisodes (total return / max drawdown):")
        print(ov.episode_table(bt, o["episodes"]).round(4).to_string())
        vs = ["buy_hold", "ewma_vol_target", "vix_target"]
        for m in names:
            print(f"\nSharpe difference, model_{m} minus baseline (95% block-bootstrap CI, Holm within family):")
            print(ov.sharpe_tests(bt, f"model_{m}", vs, cfg["stats"]).round(4).to_string(index=False))
        print("\nNet Sharpe across lag / cost / target-volatility settings:")
        sens = ov.sensitivity(spy_ret, vix, probs, cash, o, window, o["primary_model"])
        print(sens.round(3).to_string(index=False))

        out = Path("artifacts"); out.mkdir(exist_ok=True)
        bt["returns"].to_csv(out / "overlay_returns.csv")
        bt["held"].to_csv(out / "overlay_exposure.csv")
        print("\nDaily returns and exposures saved to artifacts/.")
        return 0

    from riskplatform.ingest.prices import load_prices
    from riskplatform.ingest.yahoo import YahooSource

    start = args.start or date.fromisoformat(cfg["data"]["start_date"])
    end = args.end or date.today()
    ok = load_prices(engine, YahooSource(), cfg, args.tickers, start, end)
    print("All tickers loaded." if ok else "Finished with failures; see ops.pipeline_run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
