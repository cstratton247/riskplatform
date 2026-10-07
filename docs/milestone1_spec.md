Milestone 1 Specification: Financial Risk & Predictive Analytics Platform
Status: Draft v0.2 (open decisions resolved) Date: 2026-10-07 Scope: Tier 1 (S&P 500 market-level study). Tier 2 (asset panel) is specified only where the design must anticipate it.
This document fixes the target definition, data sources, time periods, leakage controls, evaluation plan, and non-functional requirements before any modeling code is written. Changes after results are seen must be logged in the Change Log (Section 14), not made silently.
1. Purpose and research questions
Core question. Can publicly available market and economic data provide forecasting signal for periods of elevated equity-market volatility that is useful beyond what simple, well-established benchmarks already provide?
ID	Question
RQ1	Can we forecast whether S&P 500 volatility over the next 5 trading days will be unusually high?
RQ2	Does an ML model improve on established benchmarks (persistence, HAR-RV, GARCH, VIX)?
RQ3	Which factors are associated with elevated risk, and is the association stable across regimes?
RQ4	Does the forecast improve a portfolio decision after transaction costs, relative to a trailing-volatility rule?
RQ5	How do standard VaR/ES methods perform when backtested, and how does a portfolio behave under stress?


Pre-declared hypotheses (null results will be reported with equal prominence):
- H1: The best ML model has higher out-of-sample PR-AUC than the HAR-based logistic benchmark.
- H2: The best ML model has higher out-of-sample PR-AUC than a VIX-only logistic benchmark.
- H3: A volatility-targeting overlay driven by the model improves risk-adjusted performance net of costs versus the trailing-volatility overlay.
2. Target definition
2.1 Forward realized volatility
Let r_t be the daily log return of ^GSPC (close-to-close, price index) on day t.
RV_fwd(t) = sqrt( (252 / 5) * sum_{i=1..5} r_{t+i}^2 )
- Uses squared returns without demeaning (standard for realized variance; avoids noisy mean estimation over 5 observations).
- Annualized. Horizon h = 5 trading days.
2.2 Binary label
y_t = 1  if  RV_fwd(t) > Q80( { RV_fwd(s) : s <= t - 5, s in trailing 252 obs } )
y_t = 0  otherwise
Critical detail: RV_fwd(s) is only fully observed once day s + 5 has closed. At the close of day t, the latest fully known value is RV_fwd(t - 5). The threshold must therefore be computed from s <= t - 5. Using s <= t would leak information from t+1 ... t+5.
- Primary quantile: 80th percentile (about 20% positive prevalence).
- Sensitivity: 90th percentile, and a fixed-threshold variant for comparison.
- Warm-up: the first 252 + 5 observations produce no label and are excluded.
2.3 Prediction timestamp
Prediction is made at the close of day t using only information available at that time (Section 5 defines availability for each series).
2.4 Companion regression target
Forecast RV_fwd(t) directly (log scale). This supports a like-for-like comparison with HAR-RV and GARCH, evaluated with QLIKE and MSE on variance.
3. Scope
Tier	Scope	Priority
1	S&P 500 volatility study: pipeline, benchmarks, ML, statistical tests, overlay backtest, VaR/ES backtests, stress tests	Must complete
2	Panel of individual equities, portfolio-level risk engine	Stretch


The database schema and package structure are designed for Tier 2 from day one (asset-keyed tables), but no Tier 2 modeling is part of Milestone 1.
4. Time periods
Period	Dates	Use
Data pull	2004-01-01 to present	Warm-up for rolling windows
Modeling sample	2006-01-01 onward	First labels available once warm-up completes
Initial training window	2006-01-01 to 2012-12-31	First fold
Walk-forward test folds	Yearly, 2013 through 2023	Model selection and comparison
Final holdout	2024-01-01 to latest data	Touched once, after all model choices are frozen


Stress episodes inside the walk-forward folds (2015, 2018, 2020, 2022) are genuinely out of sample for the models trained before them. The 2008 crisis is in the initial training window; it is used for stress-scenario replay in the risk engine, not for model testing.
5. Data dictionary
5.1 Market data (daily)
Source (development): Yahoo Finance via yfinance behind a source adapter (Section 11.3). All raw pulls stored with source, fetched_at, and a payload hash.
Ticker	Description	Notes
^GSPC	S&P 500 index	Target series. Price index, close-to-close
^VIX	CBOE Volatility Index	Benchmark and feature. No volume
^IXIC, ^DJI	NASDAQ Composite, Dow Jones	Features
SPY	S&P 500 ETF	Tradable instrument for the overlay (adjusted close)
QQQ, DIA	Broad ETFs	Features
TLT, IEF, SHY	Treasury ETFs (long/mid/short)	Rate-sensitivity features
LQD	Investment-grade credit ETF	Credit proxy
HYG	High-yield credit ETF	Starts April 2007. See Section 5.4
XLF XLK XLE XLV XLY XLP XLI XLU XLB	Sector SPDRs	Sector dispersion and correlation features


Excluded: XLRE (2015 start) and XLC (2018 start), because their short histories create structural missingness.
Timing convention: all market series are end-of-day (US close). Feature timestamp t means information through the close of t.
5.2 Economic data (FRED)
Series	ID	Frequency	Assumed availability (v1)
Effective fed funds rate	DFF	Daily (incl. weekends)	Lag 1 business day
10-year Treasury yield	DGS10	Daily	Lag 1 business day
2-year Treasury yield	DGS2	Daily	Lag 1 business day
10y minus 2y spread	T10Y2Y	Daily	Lag 1 business day
Unemployment rate	UNRATE	Monthly	ALFRED vintages: each value stored with the dates it was first published and later revised
CPI (all urban, SA)	CPIAUCSL	Monthly	ALFRED vintages (as above)


Point-in-time policy (decided). The monthly series (UNRATE, CPIAUCSL) are revised after release, so they use ALFRED vintage data: at any as-of date the model sees the value that was actually published and current on that date. The daily series are not revised and use a rule-based availability of one business day after the observation date. Both are stored in one vintage table, with an availability_method field recording which applies. A fixed conservative lag (UNRATE from the 10th and CPIAUCSL from the 20th of the following month) is kept as a sensitivity check and as a fallback if vintage data is unavailable for a series.
Excluded from v1: GDP (quarterly, heavily revised, little value at a 5-day horizon). Credit-spread series from ICE BofA on FRED: verify available history before relying on them; HYG/LQD is the fallback proxy.
5.3 Data-quality requirements
Automated checks that fail the pipeline loudly (not warnings that get ignored):
- No duplicate (asset, date) keys
- Prices strictly positive; high >= low; low <= close <= high
- Calendar completeness against the NYSE trading calendar (missing days flagged, not silently filled)
- Return outlier flag at |r| > 15% for indices (manual review list, not auto-deleted)
- Stale-price detection (identical closes over multiple consecutive days)
- Adjusted-close ratio changes consistent with recorded splits/dividends
- Macro series: value ranges, release-lag assertions
- Row counts per run logged to a pipeline_runs table
5.4 Known availability constraints
- HYG-derived features exist only from mid-2007. They are placed in an extended feature set, evaluated as an ablation over the sample where they exist, rather than silently imputed for 2006-07.
- Missing values are never forward-filled across a gap longer than the series' native frequency allows; forward-fill is permitted only from past values.
6. Candidate feature set (v1)
All features are computed with information available at the close of day t. Definitions are frozen before modeling; additions go through the Change Log.
Group	Features
Returns	r_t, 5d, 21d, 63d cumulative returns
Backward realized vol	RV over 5d, 21d, 63d (HAR components: daily, weekly, monthly)
Downside risk	Downside semi-deviation (21d), drawdown from 252d high
Range-based vol	Parkinson and Garman-Klass estimators (21d), from OHLC
Implied vol	VIX level, 1d and 5d change, VIX minus trailing RV (variance-premium proxy)
Volume	SPY volume change vs 21d average
Rates	Fed funds change, 10y and 2y yield changes, 10y-2y spread
Macro (lagged)	UNRATE and CPI changes, using availability rules in 5.2
Credit (extended set)	Log change in HYG/LQD ratio
Cross-section	Sector ETF return dispersion, average pairwise correlation (21d)


Interpretation of feature importance uses permutation importance on out-of-sample folds, reported with the caveat that correlated features share importance.
7. Benchmark ladder
Every model is compared against everything below it. Benchmarks are implemented before the ML models.
Rung	Model	Notes
0	Prevalence	Constant probability equal to the trailing label rate. Floor
1	Persistence	Score = current backward RV (or last label); "regime persists"
2	HAR-RV	Corsi's heterogeneous autoregressive model on daily/weekly/monthly RV; regression track, and logistic on HAR components for the classification track
3	GARCH(1,1) / GJR-GARCH	Forecast 5-day variance, converted to a classification score
4	VIX-only	VIX level as the sole score, or logistic on VIX. Note: VIX is a 30-day implied measure, so horizon mismatch with the 5-day target is documented
5	Logistic regression (L2)	Full feature set
6	Gradient boosting	e.g. scikit-learn HistGradientBoosting or LightGBM
7	Random forest	Included for the nonlinearity comparison


Model-search budget is declared up front: a fixed, small hyperparameter grid per ML model, tuned only inside training windows via inner purged time-series CV. Every configuration tried is written to a model_trials table. The total number of trials is reported in the validation report to address multiple testing.
8. Validation methodology
8.1 Walk-forward with purging
- Expanding window, retrained annually.
- Purge: because labels at t use returns through t+5, any training sample with t > T_test_start - 5 is dropped. An additional embargo of 5 trading days follows each test block.
- Preprocessing (scaling, imputation) is fit on the training window only.
- Hyperparameter tuning uses an inner purged time-series CV within the training window. The outer test folds are never used for tuning.
8.2 Final holdout
The 2024-onward holdout is evaluated once, after the model, features, thresholds, and decision rule are frozen. The result is reported regardless of outcome.
8.3 Metrics
Classification track
Metric	Role
PR-AUC (against prevalence baseline)	Primary, since the positive class is a minority
ROC-AUC	Secondary
Brier score, log loss	Probability quality
Calibration curve and ECE	Is "70%" actually about 70%?
Precision/recall at declared operating thresholds	Decision relevance


Regression track: QLIKE (primary, robust to noisy volatility proxies) and MSE on variance.
8.4 Statistical testing
- Forecast comparison: Diebold-Mariano test with Newey-West HAC variance (lag h - 1 = 4) and the Harvey-Leybourne-Newbold small-sample correction.
- Confidence intervals: stationary (block) bootstrap, expected block length about 20 days, because overlapping 5-day labels are autocorrelated and an i.i.d. bootstrap understates uncertainty.
- AUC comparisons use the block bootstrap (DeLong assumes independent observations, which does not hold here).
- Regime breakdown: performance reported separately for calm, stressed, and crisis periods, since an average can hide a model that only works in one regime.
- Stability: performance by fold, not only pooled.
9. Decision-level evaluation
Question: does the forecast improve a portfolio decision after realistic costs?
- Instrument: SPY (adjusted close, total return).
- Strategies compared: (a) buy and hold; (b) volatility target using trailing 20d/60d volatility (EWMA); (c) overlay driven by the model's probability.
- Exposure rule (example, frozen before testing): w_t = clip(sigma_target / sigma_hat_t, 0, w_max), with w_max = 1.0 (no leverage) in the base case; a probability-based de-risking variant is also tested.
- Execution: signal at close of t; the base case applies the position to the return of day t+2 (one-day execution lag). A t+1 sensitivity is reported separately.
- Costs: base case 5 bps per unit of one-way turnover, with sensitivity at 0, 2, 5, and 10 bps.
- Metrics: annualized return, volatility, Sharpe, maximum drawdown, Calmar, turnover, CVaR(95%), and tail behavior in 2020 and 2022.
- Honest comparison: the model must beat rule (b), not only buy-and-hold. If it does not, that is the finding.
10. Risk engine (scoped in Milestone 1, built after the model track)
Component	Specification
VaR/ES methods	Historical, parametric (normal and Student-t), filtered historical simulation (GARCH-filtered), Monte Carlo
Confidence levels	95% and 99%, 1-day horizon (10-day as an extension)
VaR backtests	Kupiec proportion-of-failures, Christoffersen independence and conditional coverage, Basel traffic-light (250 obs, 99% VaR: green 0-4, yellow 5-9, red 10+ exceptions)
ES backtest	Exceedance-based comparison; formal ES test as an extension
Stress scenarios	Historical replays (2008, March 2020, 2022), plus hypothetical shocks: volatility spike, equity decline, correlation increase
Portfolio metrics	Volatility, max drawdown, VaR, ES, Sharpe, beta, correlation


11. Non-functional requirements
11.1 Architecture principle
Compute offline, serve precomputed. Training, walk-forward evaluation, and daily scoring run as batch jobs that write versioned results to PostgreSQL. The API reads precomputed rows. No model fitting or heavy query happens in a request path.
11.2 Runtime budgets
These are targets to be measured and reported, not claims.
Item	Target
Daily pipeline (ingest, validate, incremental features, score, risk metrics)	< 5 min on a laptop
Full historical feature rebuild (Tier 1)	< 60 s
Full walk-forward run, all models	< 15 min
API p95 latency on precomputed endpoints	< 100 ms
Historical VaR, 10k days x 50 assets	< 100 ms (vectorized)
Monte Carlo VaR, 100k paths	< 2 s


Measured with pytest-benchmark (library code), a load-test tool such as locust or k6 (API), and EXPLAIN ANALYZE before and after indexing (SQL).
11.3 Engineering requirements
- Source adapter interface: DataSource.fetch_prices(tickers, start, end) returns a normalized frame. yfinance is the development adapter; the public-website version swaps to a licensed or permissively-licensed provider without touching downstream code. Yahoo's terms do not permit redistribution, so this is a hard requirement for the website goal.
- Library first: all logic lives in an installable package (riskplatform); the CLI, API, and UI are thin layers over it.
- Idempotent ETL: upserts via INSERT ... ON CONFLICT; reruns are safe.
- Versioning: every feature row and prediction carries feature_version, model_version, and an as_of timestamp. A lightweight model registry records training window, feature list, data hash, metrics, and artifact path.
- Config-driven: tickers, windows, horizons, thresholds, costs, and lags live in versioned config files.
- Vectorized computation: no per-row Python loops in features or the risk engine.
- Observability: structured logs, a pipeline_runs table, and drift monitoring on features and prediction distributions.
12. Leakage and bias register
Each item has a control and, where possible, an automated test.
ID	Risk	Control
L1	Label threshold uses future data	Threshold from RV_fwd(s), s <= t-5 only (2.2)
L2	Overlapping labels leak across the train/test boundary	Purge and embargo (8.1)
L3	Macro release timing	Vintage valid_from dates (ALFRED) and rule-based lags (5.2); lookups only through the point-in-time function
L4	Macro revisions	ALFRED vintages return the value known on each as-of date, never the later-revised value
L5	Look-ahead in rolling features (off-by-one, centered windows)	Truncation-invariance test: features at t computed on data truncated at t must equal features at t computed on the full dataset
L6	Scaling or imputation fit on all data	Fit inside training window only
L7	Hyperparameters tuned on test folds	Inner CV only; final holdout touched once
L8	Multiple testing / researcher degrees of freedom	Pre-declared hypotheses, trial log, trial count reported
L9	Adjusted-price artifacts	Use returns, not absolute price levels, as features
L10	Vendor retroactive revisions	Store fetched_at and raw payload hash; flag diffs between pulls
L11	Calendar misalignment across series	Align on the NYSE calendar; forward-fill from past only
L12	Survivorship bias (Tier 2)	Use point-in-time constituents or state the limitation explicitly
L13	Overly optimistic backtest execution	One-day execution lag, transaction costs, no leverage in base case


13. Deliverables and definition of done
Documentation (in docs/)
1. This specification
2. Model Development Document (bank-style): purpose, data, methodology, assumptions, limitations
3. Independent Validation Report, written as if by a separate validator: conceptual soundness, outcomes analysis, sensitivity, limitations, ongoing-monitoring plan (framed on the Fed's SR 11-7 model-risk guidance)
Code
- Installable riskplatform package with tests (unit, leakage, benchmark)
- PostgreSQL schema with migrations (Alembic)
- FastAPI service and a Streamlit front end
- docker-compose.yml reproducing the full stack
Done means: a fresh clone runs end to end with one command; the leakage tests pass; every number in the documents is generated from code and reproducible; the final holdout has been evaluated exactly once; and limitations are written down.
Suggested execution order and cut lines
Priority	Work
P0	Data layer and DQ checks; target and leakage tests; benchmark ladder; walk-forward harness; ML models; statistical tests; final holdout; overlay backtest; validation report
P1	Risk engine and VaR backtests; stress scenarios; FastAPI + Streamlit + Docker Compose; SQL optimization story; runtime benchmarks
P2	Tier 2 asset panel; extended drift monitoring; public website


Suggested three-week rhythm: week 1 data layer, target, benchmarks, harness; week 2 ML, statistics, overlay, risk engine; week 3 stress tests, API/UI/Docker, benchmarks, documentation, and interview rehearsal. If time runs short, cut from the bottom of the list, so everything that remains is finished and defensible.
14. Open decisions and change log
Resolved decisions (v0.2)
#	Decision	Choice	Rationale
1	Label quantile	80th primary; 90th as a sensitivity analysis	About 20% prevalence leaves enough positive examples per walk-forward fold; 90th tests robustness to a rarer, more extreme definition
2	Macro data timing	ALFRED vintages for revised monthly series; rule-based 1-business-day lag for daily series; fixed lags as a sensitivity check	Closest to what was actually known on each date; avoids look-ahead from revisions
3	Gradient boosting library	scikit-learn HistGradientBoosting (LightGBM optional later)	Fewer dependencies, native integration with scikit-learn pipelines and custom purged splitters, simpler reproducibility
4	HYG features	Ablation only	HYG starts in 2007; keeps the primary sample uniform from 2006


Change log
Date	Change	Reason	Results seen before change?
2026-10-07	v0.1 draft	Initial	n/a
2026-10-07	v0.2: resolved the four open decisions; macro timing moved to ALFRED vintages	Closer to point-in-time accuracy	No (no modeling done yet)
