"""Database I/O for features and labels."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine, bindparam, text

from riskplatform.ingest.prices import _config_hash, _git_sha

from .compute import (
    ALL_FEATURES, BASE_FEATURES, FEATURE_DEFS, MACRO_SERIES, PRICE_TICKERS, FeatureInputs,
    to_long,
)
from .labels import log_returns, make_labels

PRICE_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def load_prices(engine: Engine, tickers: list[str]) -> dict[str, pd.DataFrame]:
    q = text(
        "select a.ticker, p.trade_date, p.open, p.high, p.low, p.close, p.adj_close, p.volume "
        "from mkt.price_daily p join ref.asset a using (asset_id) "
        "where a.ticker in :t order by a.ticker, p.trade_date"
    ).bindparams(bindparam("t", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(q, {"t": tickers}).fetchall()
    df = pd.DataFrame(rows, columns=["ticker", "trade_date"] + PRICE_COLS)
    out = {}
    for ticker, g in df.groupby("ticker"):
        g = g.set_index(pd.to_datetime(g["trade_date"]))[PRICE_COLS].astype(float)
        out[ticker] = g
    missing = [t for t in tickers if t not in out]
    if missing:
        raise RuntimeError(f"No price data in the database for: {missing}")
    return out


def load_macro(engine: Engine, series_ids: list[str]) -> dict[str, pd.DataFrame]:
    q = text(
        "select series_id, observation_date, valid_from, valid_to, value "
        "from macro.observation_vintage where series_id in :s and value is not null"
    ).bindparams(bindparam("s", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(q, {"s": series_ids}).fetchall()
    df = pd.DataFrame(rows, columns=["series_id", "observation_date", "valid_from", "valid_to", "value"])
    out = {}
    for sid, g in df.groupby("series_id"):
        g = g.drop(columns="series_id").reset_index(drop=True)
        g["value"] = g["value"].astype(float)
        out[sid] = g                          # dates stay python date objects (9999-12-31 safe)
    missing = [s for s in series_ids if s not in out]
    if missing:
        raise RuntimeError(f"No macro data in the database for: {missing}")
    return out


def load_feature_inputs(engine: Engine) -> FeatureInputs:
    return FeatureInputs(load_prices(engine, PRICE_TICKERS), load_macro(engine, MACRO_SERIES))


def start_run(engine: Engine, name: str) -> int:
    with engine.begin() as conn:
        return conn.execute(
            text("insert into ops.pipeline_run (pipeline_name, git_sha, config_hash) "
                 "values (:n, :sha, :cfg) returning run_id"),
            {"n": name, "sha": _git_sha(), "cfg": _config_hash()},
        ).scalar_one()


def finish_run(engine: Engine, run_id: int, rows_written: int, error: str | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("update ops.pipeline_run set finished_at = now(), status = :st, "
                 "rows_written = :rw, error_message = :err where run_id = :id"),
            {"st": "failed" if error else "succeeded", "rw": rows_written, "err": error, "id": run_id},
        )


def _asset_id(conn, ticker: str) -> int:
    return conn.execute(text("select asset_id from ref.asset where ticker = :t"), {"t": ticker}).scalar_one()


def write_features(engine: Engine, features: pd.DataFrame, target_ticker: str, run_id: int) -> int:
    """Upsert definitions, feature sets ('base' v1, 'extended' v1) and values (non-null only)."""
    with engine.begin() as conn:
        asset_id = _asset_id(conn, target_ticker)
        for d in FEATURE_DEFS:
            conn.execute(
                text("insert into feat.feature_definition "
                     "(name, version, feature_group, description, lookback_days, formula) "
                     "values (:n, 1, :g, :d, :l, :f) "
                     "on conflict (name, version) do update set feature_group = excluded.feature_group, "
                     "description = excluded.description, lookback_days = excluded.lookback_days, "
                     "formula = excluded.formula"),
                {"n": d.name, "g": d.group, "d": d.description, "l": d.lookback_days, "f": d.formula},
            )
        ids = {r.name: r.feature_id for r in conn.execute(
            text("select name, feature_id from feat.feature_definition where version = 1"))}

        for set_name, members in (("base", BASE_FEATURES), ("extended", ALL_FEATURES)):
            set_id = conn.execute(
                text("insert into feat.feature_set (name, version, description) values (:n, 1, :d) "
                     "on conflict (name, version) do update set description = excluded.description "
                     "returning feature_set_id"),
                {"n": set_name, "d": f"{set_name} feature set v1"},
            ).scalar_one()
            conn.execute(
                text("insert into feat.feature_set_member (feature_set_id, feature_id) "
                     "values (:s, :f) on conflict do nothing"),
                [{"s": set_id, "f": ids[m]} for m in members],
            )

        long = to_long(features)
        records = [
            {"a": asset_id, "d": r.as_of_date.date(), "f": ids[r.name], "v": float(r.value), "run": run_id}
            for r in long.itertuples(index=False)
        ]
        conn.execute(
            text("insert into feat.feature_value (asset_id, as_of_date, feature_id, value, run_id) "
                 "values (:a, :d, :f, :v, :run) "
                 "on conflict (asset_id, as_of_date, feature_id) do update set "
                 "value = excluded.value, run_id = excluded.run_id, computed_at = now()"),
            records,
        )
    return len(records)


def build_and_write_labels(engine: Engine, cfg: dict) -> int:
    t = cfg["target"]
    close = load_prices(engine, [t["asset"]])[t["asset"]]["close"]
    labels = make_labels(
        log_returns(close), t["horizon_days"], t["threshold_quantile"],
        t["threshold_lookback_obs"], t["trading_days_per_year"],
    )
    name = f"rv{t['horizon_days']}_q{int(round(t['threshold_quantile'] * 100))}"

    def _clean(x):
        return None if pd.isna(x) else x

    with engine.begin() as conn:
        asset_id = _asset_id(conn, t["asset"])
        label_def_id = conn.execute(
            text("insert into ml.label_definition "
                 "(name, version, horizon_days, threshold_quantile, threshold_lookback_obs, spec) "
                 "values (:n, 1, :h, :q, :lb, cast(:spec as jsonb)) "
                 "on conflict (name, version) do update set spec = excluded.spec returning label_def_id"),
            {"n": name, "h": t["horizon_days"], "q": t["threshold_quantile"],
             "lb": t["threshold_lookback_obs"], "spec": json.dumps(t)},
        ).scalar_one()
        records = [
            {"ld": label_def_id, "a": asset_id, "d": idx.date(),
             "rv": _clean(r.rv_fwd), "th": _clean(r.threshold),
             "y": None if pd.isna(r.y) else int(r.y),
             "kd": None if pd.isna(r.label_known_date) else r.label_known_date.date()}
            for idx, r in labels.iterrows()
        ]
        conn.execute(
            text("insert into ml.label (label_def_id, asset_id, as_of_date, rv_fwd, threshold, y, label_known_date) "
                 "values (:ld, :a, :d, :rv, :th, :y, :kd) "
                 "on conflict (label_def_id, asset_id, as_of_date) do update set "
                 "rv_fwd = excluded.rv_fwd, threshold = excluded.threshold, y = excluded.y, "
                 "label_known_date = excluded.label_known_date"),
            records,
        )
    return len(records)
