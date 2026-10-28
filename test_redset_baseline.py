import unittest
import numpy as np

from baseline_redset import BASELINE_POLICIES, max_wait_first_policy, run_baseline
from test_redset_env import make_redset_like_frame
from env.env_redset import load_redset


class RedsetBaselineTest(unittest.TestCase):
    def test_all_redset_baselines_run_and_return_metrics(self):
        records = load_redset(make_redset_like_frame(), max_records=6)
        expected = {
            "random",
            "fcfs",
            "sjf",
            "srpt",
            "max_wait_first",
            "greedy_progress",
            "greedy_cost_aware",
            "auto_wlm_inspired",
            "mlp_topk",
        }

        self.assertEqual(set(BASELINE_POLICIES), expected)
        for name in sorted(BASELINE_POLICIES):
            metrics = run_baseline(
                name,
                records,
                K=2,
                max_pending_queries=4,
                min_pending_queries=2,
                episode_length=4,
                seed=7,
            )
            self.assertFalse(metrics.empty)
            self.assertIn("method", metrics.columns)
            self.assertIn("gap", metrics.columns)
            self.assertIn("completion_ratio", metrics.columns)
            self.assertEqual(metrics["method"].iloc[-1], name)

    def test_max_wait_first_selects_longest_waiting_queries(self):
        records = load_redset(make_redset_like_frame(), max_records=6)
        from env.env_redset import RedsetAdmissionEnv

        env = RedsetAdmissionEnv(
            records,
            max_pending_queries=4,
            min_pending_queries=4,
            episode_length=4,
        )
        env.reset()
        env.waiting_time = np.array([2.0, 9.0, 5.0, 1.0], dtype=np.float32)

        action = max_wait_first_policy(env, K=2)

        self.assertEqual(action.tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
