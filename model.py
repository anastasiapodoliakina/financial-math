import numpy as np
import pandas as pd
from mesa import Model

from agents import NoiseTrader, FundamentalTrader, MomentumTrader, OptionDealer
from market import (
    S0,
    SIGMA0,
    N_NOISE,
    N_FUND,
    N_MOM,
    TOTAL_OPTION_NOTIONAL,
    PRICE_IMPACT,
    EXOG_LOGNOISE_SIGMA,
    MAX_DLOGS,
    LIQ_SHOCK_PROB,
    LIQ_SHOCK_MULT,
    T_STEPS,
    estimate_hist_vol,
    realized_volatility,
    max_drawdown,
    crash_count_abs,
    crash_count_zscore,
)
from scipy.stats import kurtosis


class GammaMarket(Model):
    def __init__(self, scenario, seed=None):
        super().__init__(seed=seed)
        self.scenario = scenario

        self.logS = np.log(S0)
        self.S = S0

        self.Q = 0.0
        self.sigma = SIGMA0
        self.last_gamma = 0.0

        self.log_price_history = [self.logS]
        self.price_history = [self.S]
        self.ret_history = []

        self.agent_list = []

        for _ in range(N_NOISE):
            self.agent_list.append(NoiseTrader(self, np.random.uniform(0.2, 0.8)))

        for _ in range(N_FUND):
            self.agent_list.append(FundamentalTrader(self, np.random.uniform(0.01, 0.05)))

        for _ in range(N_MOM):
            self.agent_list.append(MomentumTrader(self))

        # Symmetric option book
        net_mag = TOTAL_OPTION_NOTIONAL / S0
        if scenario == "short_gamma":
            net_pos = -net_mag
        elif scenario == "long_gamma":
            net_pos = +net_mag
        elif scenario == "no_hedge":
            net_pos = -net_mag
        else:
            raise ValueError("Unknown scenario")

        self.dealer = OptionDealer(self, net_pos)
        self.agent_list.append(self.dealer)

    def step(self):
        # update sigma estimate
        self.sigma = estimate_hist_vol(self.ret_history, win=50, fallback=SIGMA0)

        # reset order flow
        self.Q = 0.0

        # random activation
        self.random.shuffle(self.agent_list)
        for a in self.agent_list:
            a.step()

        # liquidity shock
        shock = (np.random.rand() < LIQ_SHOCK_PROB)
        impact = PRICE_IMPACT * (LIQ_SHOCK_MULT if shock else 1.0)

        # price update
        dlogS_det = impact * self.Q
        dlogS_noise = EXOG_LOGNOISE_SIGMA * np.random.normal()
        dlogS = float(np.clip(dlogS_det + dlogS_noise, -MAX_DLOGS, MAX_DLOGS))

        new_logS = self.logS + dlogS
        new_S = float(np.exp(new_logS))
        r = float(new_logS - self.logS)

        self.logS = new_logS
        self.S = new_S

        self.log_price_history.append(new_logS)
        self.price_history.append(new_S)
        self.ret_history.append(r)


def run_one_path(scenario, seed):
    m = GammaMarket(scenario=scenario, seed=seed)
    gamma_series = []
    for _ in range(T_STEPS - 1):
        m.step()
        gamma_series.append(m.last_gamma)

    S = np.array(m.price_history, dtype=float)
    r = np.array(m.ret_history, dtype=float)
    g = np.array(gamma_series, dtype=float)
    return S, r, g


def run_experiments(n_paths, base_seed=1000):
    scenarios = ["short_gamma", "long_gamma", "no_hedge"]
    rows = []
    price_paths = {s: [] for s in scenarios}
    return_paths = {s: [] for s in scenarios}
    gamma_paths = {s: [] for s in scenarios}

    for s in scenarios:
        for i in range(n_paths):
            seed = base_seed + 10_000 * (scenarios.index(s) + 1) + i
            S, r, g = run_one_path(s, seed)
            price_paths[s].append(S)
            return_paths[s].append(r)
            gamma_paths[s].append(g)

            rows.append({
                "scenario": s,
                "path_id": i,
                "realized_vol": realized_volatility(r),
                "max_drawdown": max_drawdown(S),
                "crashes_abs": crash_count_abs(r),
                "crashes_z": crash_count_zscore(r),
                "kurtosis": float(kurtosis(r, fisher=False)),
            })

    return pd.DataFrame(rows), price_paths, return_paths, gamma_paths
