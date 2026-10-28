import unittest

import numpy as np
import pandas as pd

from env.env_redset import RedsetAdmissionEnv, load_redset


def make_redset_like_frame():
    return pd.DataFrame(
        {
            "query_id": [1, 2, 3, 4, 5, 6],
            "arrival_timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:10",
                    "2024-01-01 00:00:20",
                    "2024-01-01 00:01:10",
                    "2024-01-01 00:01:20",
                    "2024-01-01 00:02:10",
                ]
            ),
            "execution_duration_ms": [100, 200, 400, 800, 1600, 3200],
            "mbytes_scanned": [10, 20, 40, 80, 160, 320],
            "mbytes_spilled": [0, 1, 0, 2, 0, 3],
            "num_joins": [0, 1, 2, 0, 1, 2],
            "num_scans": [1, 1, 2, 2, 3, 3],
            "num_aggregations": [0, 0, 1, 1, 2, 2],
            "query_type": ["select"] * 6,
            "was_aborted": [0] * 6,
            "was_cached": [0] * 6,
        }
    )


class RedsetLoaderTest(unittest.TestCase):
    def test_load_redset_maps_real_fields_to_internal_records(self):
        records = load_redset(make_redset_like_frame(), max_records=6)

        self.assertEqual(len(records), 6)
        self.assertIn("arrival_period", records.columns)
        self.assertIn("remaining_demand_init", records.columns)
        self.assertIn("cpu_intensity", records.columns)
        self.assertIn("io_intensity", records.columns)
        self.assertTrue((records["remaining_demand_init"] > 0).all())
        self.assertTrue((records["sort_count"] == 0).all())


class RedsetEnvTest(unittest.TestCase):
    def make_env(self):
        records = load_redset(make_redset_like_frame(), max_records=6)
        return RedsetAdmissionEnv(
            records,
            max_pending_queries=4,
            episode_length=4,
            resource_budget=1.0,
            w_min=0.05,
            lambda_D=0.8,
            lambda_W=0.2,
        )

    def test_reset_returns_feature_by_candidate_matrix_with_zero_padding(self):
        env = self.make_env()
        state = env.reset()

        self.assertEqual(state.shape, (7, 4))
        self.assertGreaterEqual((state != 0).any(axis=0).sum(), 3)

    def test_allocation_respects_budget_and_minimum_weight(self):
        env = self.make_env()
        env.reset()
        selected = np.array([0, 1, 2])
        weights = env.allocate_resources(selected)

        self.assertLessEqual(weights.sum(), 1.0 + 1e-8)
        self.assertTrue((weights[selected] >= 0.05 - 1e-8).all())
        self.assertEqual(np.count_nonzero(weights), 3)

    def test_progress_increases_with_more_own_resource(self):
        env = self.make_env()
        env.reset()
        low = env._compute_progress_for_query(0, 0.1, np.array([0]), np.array([0.1]))
        high = env._compute_progress_for_query(0, 0.5, np.array([0]), np.array([0.5]))

        self.assertGreater(high, low)

    def test_progress_decreases_with_stronger_contention(self):
        env = self.make_env()
        env.reset()
        light = env._compute_progress_for_query(
            0, 0.3, np.array([0, 1]), np.array([0.3, 0.3])
        )
        heavy = env._compute_progress_for_query(
            0, 0.3, np.array([0, 2]), np.array([0.3, 0.3])
        )

        self.assertLess(heavy, light)

    def test_step_updates_selected_queries_only_and_records_info(self):
        env = self.make_env()
        env.reset()
        before = env.remaining_demand.copy()
        next_state, reward, done, info = env.step(np.array([0, 1]))

        self.assertEqual(next_state.shape, (7, 4))
        self.assertFalse(done)
        self.assertLessEqual(info["resource_utilization"], 1.0 + 1e-8)
        self.assertIn("objective", info)
        self.assertGreaterEqual(reward, 0.0)
        self.assertLess(env.remaining_demand[0], before[0])
        self.assertLess(env.remaining_demand[1], before[1])

    def test_redset_metrics_are_reported_with_paper_aligned_names(self):
        env = self.make_env()
        env.reset()
        _, _, _, info = env.step(np.array([0, 1]))
        metrics = env.save_metric_memory()

        self.assertIn("gap", info)
        self.assertIn("wait", info)
        self.assertIn("cost", info)
        self.assertIn("completion_ratio", info)
        self.assertIn("throughput", info)
        self.assertIn("p95_waiting_time", info)
        self.assertIn("resource_utilization", info)
        self.assertIn("gap", metrics.columns)
        self.assertIn("completion_ratio", metrics.columns)
        self.assertGreaterEqual(info["completion_ratio"], 0.0)
        self.assertLessEqual(info["completion_ratio"], 1.0)

    def test_p95_waiting_penalty_increases_objective(self):
        records = load_redset(make_redset_like_frame(), max_records=6)
        base_env = RedsetAdmissionEnv(
            records,
            max_pending_queries=4,
            episode_length=10,
            lambda_D=0.8,
            lambda_W=0.2,
            lambda_P95=0.0,
        )
        p95_env = RedsetAdmissionEnv(
            records,
            max_pending_queries=4,
            episode_length=10,
            lambda_D=0.8,
            lambda_W=0.2,
            lambda_P95=0.5,
        )
        base_env.reset()
        p95_env.reset()
        base_env.waiting_time = np.array([0.0, 1.0, 8.0, 9.0], dtype=np.float32)
        p95_env.waiting_time = base_env.waiting_time.copy()

        base_objective, _, _ = base_env._objective()
        p95_objective, _, _ = p95_env._objective()

        self.assertGreater(p95_objective, base_objective)


if __name__ == "__main__":
    unittest.main()
