import unittest


class SITConditionalTopKConfigTest(unittest.TestCase):
    def test_sit_conditional_topk_wraps_env_and_uses_single_step_action(self):
        from train import _apply_conditional_topk, _conditional_topk_enabled
        from env.conditional_topk_wrapper import ConditionalTopKWrapper
        from test_conditional_topk_wrapper import make_env

        train_env, selected_num = _apply_conditional_topk(make_env(), K=3, enabled=True)

        self.assertTrue(_conditional_topk_enabled("on"))
        self.assertFalse(_conditional_topk_enabled("off"))
        self.assertIsInstance(train_env, ConditionalTopKWrapper)
        self.assertEqual(selected_num, 1)

    def test_sit_conditional_topk_off_keeps_base_env_and_k_actions(self):
        from train import _apply_conditional_topk
        from test_conditional_topk_wrapper import make_env

        env = make_env()
        train_env, selected_num = _apply_conditional_topk(env, K=3, enabled=False)

        self.assertIs(train_env, env)
        self.assertEqual(selected_num, 3)


if __name__ == "__main__":
    unittest.main()
