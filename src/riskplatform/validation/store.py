"""Database I/O for the walk-forward experiment."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
from sqlalchemy import Engine, text

from riskplatform.ingest.prices import _git_sha
from riskplatform.models.benchmarks import ModelSpec

METRICS = ["roc_auc", "pr_auc", "brier", "log_loss", "ece", "prevalence"]


def label_name(cfg: dict) -> str:
    t = cfg["target"]
    return f"rv{t['horizon_days']}_q{int(round(t['threshold_quantile'] * 100))}"


def load_modeling_frame(engine: Engine, cfg: dict) -> pd.DataFrame:
    """Feature matrix joined with labels for the target asset, indexed by as_of_date."""
    t = cfg["target"]
    with engine.connect() as conn:
        asset_id = conn.execute(text("select asset_id from ref.asset where ticker = :t"),
                                {"t": t["asset"]}).scalar_one()
        feats = conn.execute(
            text("select v.as_of_date, d.name, v.value from feat.feature_value v "
                 "join feat.feature_definition d using (feature_id) where v.asset_id = :a"),
            {"a": asset_id}).fetchall()
        labels = conn.execute(
            text("select l.as_of_date, l.y, l.label_known_date, l.threshold, l.rv_fwd "
                 "from ml.label l join ml.label_definition ld using (label_def_id) "
                 "where l.asset_id = :a and ld.name = :n and ld.version = 1"),
            {"a": asset_id, "n": label_name(cfg)}).fetchall()
    if not feats or not labels:
        raise RuntimeError("No features/labels found; run build-labels and build-features first")
    wide = (pd.DataFrame(feats, columns=["as_of_date", "name", "value"])
            .assign(as_of_date=lambda d: pd.to_datetime(d["as_of_date"]), value=lambda d: d["value"].astype(float))
            .pivot(index="as_of_date", columns="name", values="value"))
    lab = pd.DataFrame(labels, columns=["as_of_date", "y", "label_known_date", "threshold", "rv_fwd"])
    lab["as_of_date"] = pd.to_datetime(lab["as_of_date"])
    lab["label_known_date"] = pd.to_datetime(lab["label_known_date"])
    for c in ("threshold", "rv_fwd"):
        lab[c] = lab[c].astype(float)
    lab = lab.set_index("as_of_date")
    return wide.join(lab, how="inner")


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def write_walk_forward(
    engine: Engine, cfg: dict, specs: dict[str, ModelSpec], preds: pd.DataFrame,
    metrics: pd.DataFrame, fold_info: dict, data_hash: str,
) -> int:
    sha = _git_sha() or "unknown"
    written = 0
    with engine.begin() as conn:
        label_def_id = conn.execute(
            text("select label_def_id from ml.label_definition where name = :n and version = 1"),
            {"n": label_name(cfg)}).scalar_one()
        asset_id = conn.execute(text("select asset_id from ref.asset where ticker = :t"),
                                {"t": cfg["target"]["asset"]}).scalar_one()
        base_set = conn.execute(
            text("select feature_set_id from feat.feature_set where name = 'base' and version = 1")
        ).scalar_one_or_none()

        for name, spec in specs.items():
            old = ("select model_version_id from ml.model_version where name = :n "
                   "and purpose = 'walk_forward' and label_def_id = :ld")
            p = {"n": name, "ld": label_def_id}
            conn.execute(text(f"delete from ml.evaluation_result where model_version_id in ({old})"), p)
            conn.execute(text(f"delete from ml.prediction where model_version_id in ({old})"), p)
            conn.execute(text("delete from ml.model_version where name = :n and purpose = 'walk_forward' "
                              "and label_def_id = :ld"), p)

            def _insert_version(fold_label, train_start, train_end):
                return conn.execute(
                    text("insert into ml.model_version (name, family, track, purpose, fold_label, label_def_id, "
                         "feature_set_id, train_start, train_end, hyperparameters, data_hash, git_sha) "
                         "values (:n, :fam, 'classification', 'walk_forward', :fl, :ld, :fs, :ts, :te, "
                         "cast(:hp as jsonb), :dh, :sha) returning model_version_id"),
                    {"n": name, "fam": spec.family, "fl": fold_label, "ld": label_def_id,
                     "fs": base_set if name == "logit_full" else None,
                     "ts": train_start.date(), "te": train_end.date(),
                     "hp": json.dumps({**spec.hyperparameters, "features": spec.features}),
                     "dh": data_hash, "sha": sha},
                ).scalar_one()

            def _insert_metrics(mv_id, scope, row):
                recs = [{"mv": mv_id, "sc": scope, "m": m, "v": float(row[m]), "n": int(row["n"])}
                        for m in METRICS if pd.notna(row[m])]
                conn.execute(
                    text("insert into ml.evaluation_result (model_version_id, scope, metric_name, value, n_obs) "
                         "values (:mv, :sc, :m, :v, :n)"), recs)

            m = metrics[metrics["model"] == name]
            for fold_label, fi in fold_info.items():
                g = preds[(preds["model"] == name) & (preds["fold"] == fold_label)]
                if g.empty:
                    continue
                mv = _insert_version(fold_label, fi["train_start"], fi["train_end"])
                conn.execute(
                    text("insert into ml.prediction (model_version_id, asset_id, as_of_date, probability) "
                         "values (:mv, :a, :d, :p)"),
                    [{"mv": mv, "a": asset_id, "d": r.as_of_date.date(), "p": float(r.prob)}
                     for r in g.itertuples(index=False)])
                _insert_metrics(mv, f"fold:{fold_label}", m[m["scope"] == fold_label].iloc[0])
                written += len(g)

            first = min(fi["train_start"] for fi in fold_info.values())
            last = max(fi["train_end"] for fi in fold_info.values())
            _insert_metrics(_insert_version("pooled", first, last), "pooled", m[m["scope"] == "pooled"].iloc[0])
    return written
