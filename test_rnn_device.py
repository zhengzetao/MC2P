import unittest

import torch

from tools.torch_layers import RNN


class RNNDeviceTest(unittest.TestCase):
    def test_cpu_forward_preserves_initialization_and_supports_backpropagation(self):
        network = RNN(input_shape=1, hidden_dim=4).cpu()
        observations = torch.arange(12, dtype=torch.float32).reshape(2, 6)

        torch.manual_seed(123)
        expected_hidden = torch.nn.init.xavier_uniform_(torch.Tensor(2, 2, 4), gain=1)
        expected, _ = network.rnn(observations.reshape(2, 6, 1), expected_hidden)

        torch.manual_seed(123)
        actual = network(observations)

        self.assertEqual(actual.device, observations.device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        self.assertTrue(all(p.grad is not None for p in network.parameters()))


if __name__ == "__main__":
    unittest.main()
