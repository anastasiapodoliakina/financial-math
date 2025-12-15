# Flash Crashes and Delta/Gamma Hedging:How Hedging Creates Systemic Risk

Agent-based simulation of a single risky asset where heterogeneous traders generate order flow and an option dealer may hedge using **delta-change hedging** (gamma feedback). The goal is to compare tail risk and crash behavior across **short-gamma**, **long-gamma**, and **no-hedge** regimes.

---

## What the model does

The simulation evolves a single risky asset price over `T_STEPS` discrete steps. At each step:

1. Agents submit signed order flow into an aggregate demand variable `Q`.
2. A **dealer** may hedge an option position using **delta-change hedging**:
   - Compute Black–Scholes delta and gamma for a call option.
   - Compute `net_delta_t = option_position * delta_t`.
   - Trade proportional to delta change:
     - `d_delta = net_delta_t - net_delta_{t-1}`
     - `hedge_order_t = - intensity * d_delta`
3. The market converts total order flow into a **log-price change** with price impact + exogenous noise:
   - `dlogS = clip( impact * Q + exog_noise , -MAX_DLOGS, +MAX_DLOGS )`
4. A rare **liquidity shock** can temporarily multiply price impact.

**Key mechanism:** in **short-gamma**, dealer hedging can become *positive feedback* (trend-following), increasing tail risk and crash frequency. In **long-gamma**, hedging is more stabilizing.

---

## Scenarios

Experiments compare three regimes:

- **short_gamma**: dealer is net short options (negative gamma exposure) and hedges with higher intensity.
- **long_gamma**: dealer is net long options (positive gamma exposure) and hedges with lower intensity.
- **no_hedge**: dealer holds an option position but hedging intensity is set to zero.

---

## Hypotheses tested

**H1 (Distribution shape):**  
Return distribution under short-gamma differs in shape from long-gamma (tails / non-Gaussianity).

**H2 (Flash crashes, PRIMARY):**  
Short-gamma produces more **volatility-adjusted crash events** than long-gamma.  

**H3 (Tail risk via drawdowns):**  
Short-gamma has deeper maximum drawdowns than long-gamma.

---

## Crash definitions used

Two crash counters are computed per simulated path:

1. **Primary (z-crashes)**: volatility-adjusted events  
   - `r_t < CRASH_Z * rolling_std_CRASH_Z_WIN(r)`
2. **Secondary (absolute crashes)**: fixed threshold  
   - `r_t < FLASH_CRASH_ABS`

---

## Outputs (per path)

Each simulated path produces:

- `realized_vol`: annualized volatility of log-returns
- `max_drawdown`: maximum peak-to-trough drawdown of the price series
- `crashes_z`: count of z-crashes (primary)
- `crashes_abs`: count of absolute crashes (secondary)
- `kurtosis`: return kurtosis (Pearson, i.e., `fisher=False`)

---

## Statistical tests

The notebook / `analysis.py` applies robust two-sample and multi-sample tests:

### H1 (distribution shape)
- Kolmogorov–Smirnov (KS) two-sample test
- Cramér–von Mises (CvM) two-sample test
- Anderson–Darling k-sample test
- Additional proxy: Welch t-test on **per-path kurtosis**

### H2 (crash counts)
- Mann–Whitney U (one-sided: short > long) on `crashes_z`
- Welch t-test on `crashes_z`
- Kruskal–Wallis test across (short, long, no_hedge)
- Binary test for “any crash occurred”: Chi-square or Fisher exact (fallback)

### H3 (drawdowns)
- Mann–Whitney U (one-sided: short < long) on `max_drawdown`
- Welch t-test on `max_drawdown`
- KS test on drawdown distributions
- Kruskal–Wallis across (short, long, no_hedge)
- (Optional) Levene test for variance differences

---

## Repository structure

```text
project/
├── agents.py        # agent classes (noise, fundamental, momentum, dealer)
├── market.py        # parameters, pricing/impact update helpers, crash metrics
├── model.py         # Mesa Model + run_one_path / run_experiments
├── analysis.py      # statistical tests for H1–H3
├── visualization.py # plots (paths, histograms, crash marks, etc.)
├── main.ipynb       # experiments / demo notebook
└── README.md
