import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from DQN.dqn import DQN
from env.conditional_topk_wrapper import ConditionalTopKWrapper
from test_conditional_topk_wrapper import make_env


class DQNTerminalTargetTest(unittest.TestCase):
    def test_empty_terminal_observation_keeps_training_loss_and_weights_finite(self):
        environment = ConditionalTopKWrapper(make_env(), 3)
        observation = environment.reset()
        model = DQN(
            environment, selected_num=1, strategy="set_transformer",
            buffer_size=8, batch_size=1, seed=123, device="cpu",
        )
        model._current_progress_remaining = 1.0
        model.replay_buffer.add(
            observation[None], np.zeros_like(observation)[None],
            np.array([[0]]), np.array([1.0]), np.array([True]), [{}],
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            model._logger = configure(temporary_dir, format_strings=[])
            model.train(gradient_steps=1, batch_size=1)
            self.assertTrue(np.isfinite(model._logger.name_to_value["train/loss"]))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))


if __name__ == "__main__":
    unittest.main()
