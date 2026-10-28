import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import train_SP500


class PreprocessingReached(Exception):
    """Stop the integration check before feature engineering and training."""


class SP500DataLoadingTest(unittest.TestCase):
    def test_training_loads_local_prices_before_feature_engineering(self):
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir) / "datasets" / "SP500"
            data_dir.mkdir(parents=True)
            prices = pd.DataFrame({"Week_ID": range(3231), "Close": [100.0] * 3231})
            prices.to_csv(data_dir / "0.csv", index=False)
            prices.to_csv(data_dir / "1.csv", index=False)

            def check_loaded_prices(engineer, frame):
                self.assertEqual(len(frame), 6462)
                self.assertEqual(set(frame["tic"]), {"0", "1"})
                raise PreprocessingReached()

            try:
                os.chdir(temporary_dir)
                with patch.object(
                    train_SP500.FeatureEngineer, "preprocess_data", check_loaded_prices
                ):
                    with self.assertRaises(PreprocessingReached):
                        train_SP500.train_stock_trading("SP500", 5, "interactive")
            finally:
                os.chdir(original_dir)


if __name__ == "__main__":
    unittest.main()
