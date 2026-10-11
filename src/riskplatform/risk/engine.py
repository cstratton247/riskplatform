"""Runs all VaR/ES methods with the pre-declared settings. Database-free."""
from __future__ import annotations

import pandas as pd

from . import var_es as ve


def all_estimates(port: pd.Series, rc: dict, start) -> dict:
    """Maps (method, confidence) -> DataFrame[var, es] for the four declared methods."""
    confs = rc["confidence_levels"]
    est = {}
    for c in confs:
        est[("historical", c)] = ve.historical(port, c, rc["window"])
        est[("normal_ewma", c)] = ve.normal(port, c, rc["ewma_lambda"])
        est[("student_t_ewma", c)] = ve.student_t(port, c, rc["ewma_lambda"], rc["student_t_df"])
    fhs = ve.filtered_historical(port, confs, rc["window"], rc["garch_min_fit_obs"], start)
    for c in confs:
        est[("fhs_gjr_garch", c)] = fhs[c]
    return est
