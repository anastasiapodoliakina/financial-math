import numpy as np
from mesa import Agent

from market import (
    FUNDAMENTAL_VALUE,
    MOM_LOOKBACK,
    MOM_ALPHA,
    MOM_MAX_ORDER,
    K_STRIKE,
    RISK_FREE,
    T_MATURITY,
    bs_delta_gamma_call,
)

class NoiseTrader(Agent):
    def __init__(self, model, sigma):
        super().__init__(model)
        self.sigma = float(sigma)

    def step(self):
        self.model.Q += np.random.normal(0.0, self.sigma)


class FundamentalTrader(Agent):
    def __init__(self, model, kappa):
        super().__init__(model)
        self.kappa = float(kappa)

    def step(self):
        S = self.model.S
        self.model.Q += self.kappa * (FUNDAMENTAL_VALUE - S)


class MomentumTrader(Agent):
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        lp = self.model.log_price_history
        if len(lp) < MOM_LOOKBACK + 1:
            return
        avg_ret = np.diff(lp[-(MOM_LOOKBACK + 1) :]).mean()
        q = MOM_ALPHA * np.sign(avg_ret)
        self.model.Q += float(np.clip(q, -MOM_MAX_ORDER, MOM_MAX_ORDER))


class OptionDealer(Agent):
    """
    Dealer hedges using delta change (gamma-feedback):
      hedge_order_t ∝ -(net_delta_t - net_delta_{t-1})
    """

    def __init__(self, model, net_option_pos):
        super().__init__(model)
        self.net_option_pos = float(net_option_pos)
        self.prev_net_delta = 0.0

        if model.scenario == "short_gamma":
            self.intensity = 2.2
        elif model.scenario == "long_gamma":
            self.intensity = 0.7
        else:
            self.intensity = 0.0

    def step(self):
        if self.intensity <= 0:
            self.model.last_gamma = 0.0
            return

        S = self.model.S
        sigma = self.model.sigma
        delta, gamma = bs_delta_gamma_call(S, K_STRIKE, RISK_FREE, sigma, T_MATURITY)

        net_delta = self.net_option_pos * delta
        d_delta = net_delta - self.prev_net_delta

        hedge_order = -self.intensity * d_delta

        self.prev_net_delta = net_delta
        self.model.Q += hedge_order
        self.model.last_gamma = self.net_option_pos * gamma
