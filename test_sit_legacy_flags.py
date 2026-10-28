import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class SITLegacyFlagTest(unittest.TestCase):
    def test_env_flag_parser_accepts_on_off_values(self):
        from env.env_portfolio import _env_flag

        with patch.dict(os.environ, {"MC2P_LEGACY_SIT_REWARD": "on"}):
            self.assertTrue(_env_flag("MC2P_LEGACY_SIT_REWARD"))
        with patch.dict(os.environ, {"MC2P_LEGACY_SIT_REWARD": "0"}):
            self.assertFalse(_env_flag("MC2P_LEGACY_SIT_REWARD"))

    def test_policy_flag_parser_accepts_on_off_values(self):
        from DQN.policy import _env_flag

        with patch.dict(os.environ, {"MC2P_LEGACY_INTERACTIVE_FLATTEN": "true"}):
            self.assertTrue(_env_flag("MC2P_LEGACY_INTERACTIVE_FLATTEN"))
        with patch.dict(os.environ, {"MC2P_LEGACY_INTERACTIVE_FLATTEN": "off"}):
            self.assertFalse(_env_flag("MC2P_LEGACY_INTERACTIVE_FLATTEN"))

    def test_true_index_qp_target_uses_price_list_index_column(self):
        from env.env_portfolio import _qp_target_index_value

        price_list = np.array([[10.0, 1.0, 2.0], [20.0, 3.0, 4.0]])
        state = price_list[:, 1:]
        data = pd.DataFrame({"price_list": [price_list]})

        with patch.dict(os.environ, {"MC2P_TRUE_INDEX_QP_TARGET": "off"}):
            np.testing.assert_array_equal(_qp_target_index_value(data, state), state[:, 0])
        with patch.dict(os.environ, {"MC2P_TRUE_INDEX_QP_TARGET": "on"}):
            np.testing.assert_array_equal(_qp_target_index_value(data, state), price_list[:, 0])


if __name__ == "__main__":
    unittest.main()
