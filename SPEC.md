# factorlab — cross-sectional alpha research lab

Read `..\CONVENTIONS.md` first. Package name: `factorlab`.

## Goal

A research harness for cross-sectional equity factors: build alphas from panel
data, clean them properly, measure predictive power with purged cross-validation,
form quantile portfolios, and guard against multiple-testing false discoveries.

Panel convention: `pandas.DataFrame` with MultiIndex `(date, asset)` and columns
`open, high, low, close, volume, returns, market_cap`. Long format, sorted.

## Modules

`src/factorlab/`

1. `panel.py` — `synthetic_panel(n_assets=40, n_days=750, seed=42)` generating a
   realistic panel: common market factor + per-asset betas + AR(1) idiosyncratic
   noise, lognormal market caps, volume correlated with |returns|. Validation
   helpers (`validate_panel`) and `to_wide(field)` / `from_wide`.
2. `preprocess.py` — `winsorize(s, limits=(0.01, 0.99))` (cross-sectional),
   `zscore(s)`, `rank_normal(s)` (Gaussian via `scipy.stats.norm.ppf`),
   `demean(s)`, `neutralize(s, exposures, add_const=True)` (cross-sectional OLS
   residuals, pure numpy lstsq), `fillna_cross_section(s, method)` refusing to
   forward-fill beyond `max_gap`. All operate per date where relevant.
3. `factors.py` — factor library returning a `pd.Series` indexed like the panel:
   - `momentum(close_wide, lookback=126, skip=21)`
   - `short_term_reversal(returns_wide, window=5)`
   - `volatility(returns_wide, window=60)` (low-vol factor = negative)
   - `liquidity(volume_wide, market_cap_wide)` = turnover z-score
   - `amihud(returns_wide, volume_wide, window=21)` illiquidity
   - `trend_quality(close_wide, window=126)` = R² of log-price vs time
   Each: documented formula, `min_periods`, NaN before warm-up.
4. `evaluate.py` — `forward_returns(returns_wide, horizon=1)` (sum of returns
   over the next `h` days, no lookahead); `ic_series(factor, fwd)` (Spearman
   per date); `ic_summary` (mean IC, IC IR = mean/std, t-stat with Newey-West
   lags, hit rate, IC decay table over horizons 1..10); `quantile_returns(factor,
   fwd, n_quantiles=5, weighting='equal'|'cap')`; `long_short(quantile_returns)`;
   `turnover(factor, n_quantiles)`.
5. `cv.py` — `PurgedKFold(n_splits, purge, embargo)` (López de Prado): labels
   spanning `[t, t+horizon]` must not leak into the test fold — implement
   explicit purge + embargo on a date index; yields `(train_idx, test_idx)`.
6. `deflated.py` — `deflated_sharpe_ratio(sr, n_trials, n_obs, skew, kurt)`
   (Bailey & López de Prado) and `pbo(cscv_paths)` (probability of backtest
   overfitting via combinatorial symmetric cross-validation, small, vectorized).
   Document the math; return also the null-expected max Sharpe.
7. `report.py` — `factor_report(panel, factor, fwd_horizon)` returning a dict +
   `to_markdown()` with IC table, quantile returns, turnover, DSR. Sampled
   quantile monotonicity flag.
8. `__main__.py` — `demo`: build synthetic panel, all factors, print IC summary +
   quantile table for momentum, per-factor comparison table.

## Tests

- `test_panel.py`: shape/dtype/MultiIndex contract, determinism, cross-sectional
  variance > 0 each day.
- `test_preprocess.py`: winsorize clips at exact quantiles, zscore mean 0 / std 1,
  rank_normal approx normality, neutralize residuals orthogonal to exposures
  (|corr| < 1e-8), fillna respects max_gap.
- `test_factors.py`: momentum on a hand-built 2-asset price path gives expected
  ordering; reversal sign; volatility factor higher for the noisier asset;
  amihud monotone in price impact; warm-up NaNs present.
- `test_evaluate.py`: IC of a factor with perfectly known relationship ≈ 1;
  IC of noise ≈ 0 within tolerance; forward returns never use future beyond
  horizon; quantile portfolio counts are balanced.
- `test_cv.py`: no train date within purge+embargo of any test date; fold sizes
  sum to the usable range; n_splits validation errors.
- `test_deflated.py`: DSR decreases as `n_trials` grows; PBO in [0,1] and lower
  for a genuinely predictive synthetic factor than for noise.

## README sections

Research workflow diagram, factor catalogue table with formulas, IC/IR
definitions, purged CV explanation, DSR/PBO notes, honest limitations.
