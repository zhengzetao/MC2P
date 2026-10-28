import tempfile
import unittest
from collections.abc import Mapping

import torch

from DQN.dqn import DQN


class DQNCheckpointTest(unittest.TestCase):
    def test_saved_checkpoint_reloads_without_changing_predictions(self):
        original = DQN.__new__(DQN)
        original.policy = torch.nn.Linear(3, 2)
        restored = DQN.__new__(DQN)
        restored.policy = torch.nn.Linear(3, 2)
        observations = torch.tensor([[1.0, 2.0, 3.0]])

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = temporary_dir + "/"
            original.save(path)
            checkpoint = torch.load(path + "best_policy.pth")
            self.assertIsInstance(checkpoint, Mapping)
            restored.load(path)

        torch.testing.assert_close(
            restored.policy(observations), original.policy(observations), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
