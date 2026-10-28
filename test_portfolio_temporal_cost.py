import unittest

import numpy as np
import pandas as pd

from env.env_portfolio import StockPortfolioEnv


def make_portfolio_env(temporal_cost_weight=0.0):
    lookback = 3
    stock_dim = 4
    rows = []
    price_list = np.array(
        [
            [1.00, 1.00, 1.00],
            [1.01, 1.02, 1.00],
            [1.02, 1.03, 1.01],
        ],
        dtype=np.float32,
    )
    return_list = np.array(
        [
            [0.00, 0.010, 0.009, 0.008],
            [0.00, 0.011, 0.010, 0.009],
            [0.00, 0.012, 0.011, 0.010],
        ],
        dtype=np.float32,
    )
    for week_id in range(3):
        for tic in range(stock_dim):
            rows.append(
                {
                    "Week_ID": week_id,
                    "tic": tic,
                    "Close": 1.0 + 0.01 * week_id + 0.001 * tic,
                    "price_list": price_list,
                    "return_list": return_list,
                }
            )
    df = pd.DataFrame(rows)
    df.index = df.Week_ID
    indexing = pd.Series([0.010, 0.011, 0.012])
    return StockPortfolioEnv(
        df=df,
        indexing=indexing,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=1000000,
        transaction_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=stock_dim,
        action_space=stock_dim - 1,
        lookback=lookback,
        tech_indicator_list=[],
        temporal_cost_weight=temporal_cost_weight,
    )


class PortfolioTemporalCostTest(unittest.TestCase):
    def test_temporal_cost_weight_penalizes_action_changes(self):
        no_cost_env = make_portfolio_env(temporal_cost_weight=0.0)
        cost_env = make_portfolio_env(temporal_cost_weight=1.0)

        no_cost_env.reset()
        cost_env.reset()
        _, no_cost_reward, _, _ = no_cost_env.step(np.array([0, 1]))
        _, cost_reward, _, _ = cost_env.step(np.array([0, 1]))

        self.assertLess(cost_reward, no_cost_reward)


if __name__ == "__main__":
    unittest.main()
