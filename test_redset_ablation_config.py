import argparse
import unittest

from redset_ablation import _baseline_env_kwargs


class RedsetAblationConfigTest(unittest.TestCase):
    def test_baseline_env_kwargs_include_tuned_reward_and_contention(self):
        args = argparse.Namespace(
            max_pending=64,
            K=5,
            episode_length=200,
            lambda_D=0.5,
            lambda_W=0.35,
            lambda_P95=0.15,
            eta_cpu=1.0,
            eta_io=1.0,
        )

        kwargs = _baseline_env_kwargs(args)

        self.assertEqual(kwargs["lambda_D"], 0.5)
        self.assertEqual(kwargs["lambda_W"], 0.35)
        self.assertEqual(kwargs["lambda_P95"], 0.15)
        self.assertEqual(kwargs["eta_cpu"], 1.0)
        self.assertEqual(kwargs["eta_io"], 1.0)


if __name__ == "__main__":
    unittest.main()
