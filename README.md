# factorlab

A small, offline research harness for **cross-sectional equity factors**:
generate or load a panel, build alphas, clean them, measure predictive power
with purged cross-validation, form quantile portfolios, and deflate the
resulting Sharpe ratio for the number of trials you ran.

Everything runs locally with `numpy`, `pandas`, `scipy` and (optionally)
`matplotlib`. No network, no API keys, no TA-Lib, no scikit-learn.

```text
raw panel  ->  factors  ->  preprocessing  ->  evaluation  ->  guardrails
(open/high/   momentum     winsorize          IC / IC IR      purged K-fold
 low/close/   reversal      z-score            quantile rets   DSR
 volume/      low-vol       rank-normal        turnover        PBO
 returns/     liquidity     neutralize
 market_cap)  amihud        fillna
              trend-quality
```

## Install and quickstart

```bash
python -m pip install -e .
python -m pytest -q
python -m factorlab demo
python -m factorlab demo --assets 60 --days 1000 --seed 7 --horizon 5
python -m factorlab demo --plot factorlab_demo.png   # headless PNG, no window
```

Python 3.12, `src/` layout, console script `factorlab` (equivalent to
`python -m factorlab`).

```python
from factorlab import PanelData, synthetic_panel
from factorlab.factors import momentum
from factorlab.evaluate import forward_returns, ic_summary, quantile_returns
from factorlab.report import factor_report

panel = PanelData(synthetic_panel(n_assets=40, n_days=750, seed=42))
mom = momentum(panel.close, lookback=126, skip=21)

print(ic_summary(mom, panel.returns, max_horizon=10, nw_lags=5).to_dict())
q = quantile_returns(mom, forward_returns(panel.returns, 5), n_quantiles=5)
print(q.mean())

report = factor_report(panel, mom, fwd_horizon=5, n_trials=100)
print(report.to_markdown())
```

## Panel convention

A `pandas.DataFrame` with MultiIndex `(date, asset)`, sorted and unique, and
exactly these columns:

| column | meaning |
|---|---|
| `open`, `high`, `low`, `close` | positive prices |
| `volume` | non-negative share volume |
| `returns` | simple daily return, `close.pct_change()` |
| `market_cap` | positive market capitalization |

`validate_panel` enforces the contract, including strictly positive
cross-sectional return variance on every date (a flat cross-section cannot be
ranked). `to_wide(field)` pivots to dates × assets; `to_long` / `from_wide`
convert back **without dropping NaN warm-up rows**.

## Factor catalogue

All factors return a long `Series` on the full panel product, with NaN during
the warm-up. Values at date `t` use only information available at the close of
`t`.

| factor | formula | orientation |
|---|---|---|
| `momentum(close, lookback=126, skip=21)` | `prod_{j=skip+1}^{skip+lookback}(1+r_{t-j}) - 1` | high past return → high factor |
| `short_term_reversal(returns, window=5)` | `-sum_{j=0}^{window-1} r_{t-j}` | recent losers → high factor |
| `volatility(returns, window=60)` | `-std(r_{t-window+1..t}, ddof=1)` | **low-vol factor**: quieter asset → higher factor |
| `liquidity(volume, market_cap)` | cross-sectional z-score of `volume / market_cap` | high turnover → high factor |
| `amihud(returns, volume, window=21)` | `mean(|r| / volume)` over the window | illiquidity: high price impact → high factor |
| `trend_quality(close, window=126)` | `R^2` of `log(close)` regressed on a linear time trend | clean trend → high factor |

Notes:

* Momentum compounds log returns (`log1p`) and uses `min_periods=lookback`, so
  the first `lookback + skip` rows are NaN.
* `volatility` is deliberately negated. The low-volatility anomaly predicts
  *higher* risk-adjusted returns for *lower*-volatility names, so the raw
  standard deviation would have the wrong sign. A noisier asset therefore gets
  a **lower** factor value.
* `liquidity` and `amihud` are raw characteristics: whether you go long or
  short them is an empirical question answered by the IC sign.
* `trend_quality` is bounded in `[0, 1]`; NaN if a window has zero variance.

## Preprocessing

Every function operates per date (cross-section) and preserves the input index
and NaN pattern.

| function | definition |
|---|---|
| `winsorize(s, limits=(0.01, 0.99))` | clip to the per-date lower/upper quantiles |
| `zscore(s)` | `(x - mean) / std(ddof=0)`; zero variance → NaN |
| `rank_normal(s)` | `Phi^{-1}(rank / (n+1))`, clipped to `[1e-12, 1-1e-12]` |
| `demean(s)` | `x - mean(x)` |
| `neutralize(s, exposures, add_const=True)` | OLS residuals from a per-date regression on exposures (`numpy.linalg.lstsq`) |
| `fillna_cross_section(s, method, max_gap)` | bounded `ffill`/`bfill`, or `ffill` then cross-sectional median |

`neutralize` returns residuals orthogonal to every exposure by the normal
equations, so `|corr(residual, exposure)| < 1e-8` (machine precision) in the
tests. `fillna_cross_section` refuses to forward-fill more than `max_gap`
consecutive periods: stale prices manufacture fake signals.

## Evaluation

**Forward returns** (no look-ahead):

```text
fwd_t = sum_{j=1..h} r_{t+j}
```

The last `h` rows are NaN. A partial window is never returned: if any of the
`h` future returns is missing, `fwd_t` is NaN. Tests perturb data at `t+h+1`
and verify that `fwd_t` does not move.

**Information coefficient** (per date):

```text
IC_t = corr_cs( factor_t , fwd_t )          # Spearman by default
```

**IC summary**:

* `mean_ic = mean(IC_t)`
* `IC IR = mean(IC) / std(IC, ddof=1)` — the per-period information ratio
* `t-stat` — Newey-West HAC t-statistic for `mean(IC) = 0`, with
  `Var = (gamma_0 + 2 sum_{j=1..L}(1 - j/(L+1)) gamma_j) / n`
* `hit_rate = P(IC_t > 0)`
* `ic_decay` — mean IC at horizons `1..10`

**Quantile portfolios**: assets are ranked each date with
`method="first"`, so bucket counts are exactly balanced (differ by at most
one). Q1 is the lowest factor. Returns are equal-weighted or cap-weighted:

```text
r_q,t = sum_{i in q} w_i r_i,t / sum_{i in q} w_i
```

`long_short` is `Q_last - Q1`. `turnover` is the fraction of assets that change
bucket versus the previous date (a holdings-churn proxy).

## Purged cross-validation

Labels span `[t, t + horizon - 1]`. A plain K-Fold leaks: a training label that
overlaps the test window encodes the test period's future. `PurgedKFold`:

1. splits the date index into `n_splits` contiguous test blocks;
2. **purges** every training date whose label interval overlaps the test block;
3. applies an additional **embargo** of `embargo` days after the test block to
   break serial correlation.

For a test block `[s, e)`, training dates are restricted to
`i <= s - purge - embargo - horizon` or `i >= e + purge + embargo + horizon`.
Tests assert that no training label overlaps any test window, that the
training/test sets are disjoint, that the minimum positional distance exceeds
`purge + embargo`, and that the test folds partition the full sample exactly
once.

## Multiple-testing guardrails

**Deflated Sharpe Ratio** (Bailey & López de Prado, 2014). With `N` independent
zero-skill trials and `n` observations:

```text
SR_0 = sqrt(1/(n-1)) * [ (1-g) Phi^{-1}(1 - 1/N)
                          + g Phi^{-1}(1 - 1/(N e)) ]
PSR(SR*) = Phi( (SR - SR*) sqrt(n-1)
                / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )
DSR = PSR(SR_0)
```

`g` is the Euler-Mascheroni constant and `kurt` is non-excess kurtosis.
Sharpe ratios are **per observation** (divide an annualized Sharpe by
`sqrt(252)` for daily data). `DSR` falls as `n_trials` grows; values below
~0.95 mean the result is not convincing after selection.

**Probability of Backtest Overfitting** (Bailey et al., 2017) via
Combinatorially Symmetric Cross-Validation. Split the sample into `S` blocks
(`S` even); every choice of `S/2` blocks is in-sample, its complement is
out-of-sample. For each path, select the in-sample best strategy; it is
"overfit" if its out-of-sample rank is in the bottom half. `PBO` is the
fraction of paths that overfit. With a single strategy the rule reduces to
"out-of-sample Sharpe <= 0". `cscv_combinations(10)` enumerates all 252 paths;
`cscv_paths_from_returns` and `cscv_paths_from_splits` build the inputs.

## Synthetic panel

`synthetic_panel(n_assets=40, n_days=750, seed=42)` simulates:

* a common market factor `r_m ~ N(0.0002, 0.01^2)`;
* per-asset betas `~ U(0.5, 1.5)`;
* AR(1) idiosyncratic noise (`rho = 0.1`, `sigma_eta = 0.015`);
* lognormal market caps that evolve with returns;
* volume correlated with `|returns|`.

Prices are built as `100 * prod(1 + r)`, so `close.pct_change()` reproduces the
`returns` column to machine precision. The generator is deterministic for a
given seed.

## Tests

```bash
python -m pytest -q
```

Coverage includes: panel contract and determinism, winsorization at exact
quantiles, z-score moments, rank-normal normality, neutralization orthogonality
(`|corr| < 1e-8`), `max_gap` enforcement, factor signs and ordering, forward
returns beyond-horizon invariance, perfect/noise IC, balanced quantiles,
purged-fold leakage, DSR monotonicity in `n_trials`, and PBO discrimination
between a genuinely predictive series and noise.

## Honest limitations

* Synthetic data is not a market. Real panels have delistings, survivorship
  bias, corporate actions, asynchronous closes, shorting constraints and
  transaction costs; none are modelled here.
* The panel has no sector, country or currency fields, so neutralization is
  limited to whatever exposures you supply.
* `amihud` uses share volume, not dollar volume, because the factor signature
  has no price argument; pass a dollar-volume frame for the textbook version.
* IC t-statistics assume a stationary relationship. Regime shifts, publication
  bias and data snooping beyond the declared `n_trials` are not handled.
* DSR assumes independent trials and normal-ish higher moments; correlated
  trials make the true hurdle lower than reported.
* PBO is a diagnostic, not a guarantee: a low PBO on a short sample can still
  be luck. CSCV paths overlap, so their fraction is not an independent sample.
* Quantile portfolios ignore costs, borrow and capacity; turnover is reported
  so you can judge them yourself.
* No live trading, order management or portfolio construction is included.

## License

MIT — see [LICENSE](LICENSE).
