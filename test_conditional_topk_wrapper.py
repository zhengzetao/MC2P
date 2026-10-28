import unittest

import numpy as np
import pandas as pd
from gym import spaces

from env.conditional_topk_wrapper import ConditionalTopKWrapper
from env.env_redset import RedsetAdmissionEnv, load_redset
from train_redset import _conditional_topk_enabled


def make_records():
    frame = pd.DataFrame(
        {
            "query_id": [1, 2, 3, 4, 5, 6],
            "arrival_timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:10",
                    "2024-01-01 00:00:20",
                    "2024-01-01 00:00:30",
                    "2024-01-01 00:01:00",
                    "2024-01-01 00:01:30",
                ]
            ),
            "execution_duration_ms": [100, 200, 300, 400, 500, 600],
            "mbytes_scanned": [10, 20, 30, 40, 50, 60],
            "num_joins": [0, 1, 0, 1, 0, 1],
            "num_aggregations": [0, 0, 1, 1, 2, 2],
            "query_type": ["select"] * 6,
            "was_aborted": [0] * 6,
        }
    )
    return load_redset(frame)


def make_env():
    return RedsetAdmissionEnv(
        make_records(),
        max_pending_queries=5,
        min_pending_queries=3,
        episode_length=5,
        resource_budget=1.0,
        w_min=0.05,
    )


class DuplicateIndexEnv:
    def __init__(self):
        self.df = pd.DataFrame({"value": [1, 2, 3, 4]}, index=[0, 0, 1, 1])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.zeros((2, 3), dtype=np.float32),
            high=np.ones((2, 3), dtype=np.float32),
            dtype=np.float32,
        )


class ConditionalTopKWrapperTest(unittest.TestCase):
    def test_reset_adds_selected_mask_feature(self):
        wrapped = ConditionalTopKWrapper(make_env(), K=3)
        state = wrapped.reset()

        self.assertEqual(state.shape, (8, 5))
        self.assertTrue(np.all(state[-1] == 0.0))
        self.assertEqual(wrapped.action_space.n, 5)

    def test_partial_picks_do_not_advance_base_environment(self):
        base = make_env()
        wrapped = ConditionalTopKWrapper(base, K=3)
        wrapped.reset()
        period_before = base.current_period

        state, reward, done, info = wrapped.step(0)

        self.assertEqual(base.current_period, period_before)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertTrue(info["partial_selection"])
        self.assertEqual(info["selected_indices"], [0])
        self.assertEqual(state[-1, 0], 1.0)

    def test_kth_pick_commits_selected_set_to_base_environment(self):
        base = make_env()
        wrapped = ConditionalTopKWrapper(base, K=3)
        wrapped.reset()
        period_before = base.current_period

        wrapped.step(0)
        wrapped.step(1)
        state, reward, done, info = wrapped.step(2)

        self.assertEqual(base.current_period, period_before + 1)
        self.assertNotEqual(reward, 0.0)
        self.assertFalse(info["partial_selection"])
        self.assertEqual(info["selected_indices"], [0, 1, 2])
        self.assertTrue(np.all(state[-1] == 0.0))

    def test_duplicate_action_is_masked_without_advancing_selection(self):
        wrapped = ConditionalTopKWrapper(make_env(), K=3)
        wrapped.reset()

        wrapped.step(0)
        state, reward, done, info = wrapped.step(0)

        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["selected_indices"], [0])
        self.assertEqual(state[-1, 0], 1.0)

    def test_conditional_topk_config_parser_accepts_on_off_values(self):
        self.assertTrue(_conditional_topk_enabled("on"))
        self.assertTrue(_conditional_topk_enabled("true"))
        self.assertTrue(_conditional_topk_enabled("1"))
        self.assertFalse(_conditional_topk_enabled("off"))
        self.assertFalse(_conditional_topk_enabled("false"))
        self.assertFalse(_conditional_topk_enabled("0"))

    def test_expanded_df_supports_duplicate_period_index(self):
        wrapped = ConditionalTopKWrapper(DuplicateIndexEnv(), K=3)

        self.assertEqual(len(wrapped.df.index.unique()), 6)


if __name__ == "__main__":
    unittest.main()
