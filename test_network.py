import unittest
import numpy as np
from network import NeuralNetwork, sigmoid, sigmoid_derivative
from two_layer_network import train_two_layer


class TestNeuralNetwork(unittest.TestCase):

    def test_sigmoid_values(self):
        self.assertAlmostEqual(float(sigmoid(np.array([0.0]))[0]), 0.5, places=5)
        self.assertAlmostEqual(float(sigmoid(np.array([1000.0]))[0]), 1.0, places=5)
        self.assertAlmostEqual(float(sigmoid(np.array([-1000.0]))[0]), 0.0, places=5)

    def test_sigmoid_derivative(self):
        s_val = np.array([0.5, 0.8, 0.2])
        expected = s_val * (1 - s_val)
        computed = sigmoid_derivative(s_val)
        np.testing.assert_allclose(computed, expected, rtol=1e-5)

    def test_forward_pass_dimensions(self):
        nn = NeuralNetwork(input_dim=3, hidden_dim=5, output_dim=2, seed=42)
        X = np.random.randn(10, 3)
        l1, l2 = nn.forward(X)
        self.assertEqual(l1.shape, (10, 5))
        self.assertEqual(l2.shape, (10, 2))
        self.assertTrue(np.all((l2 >= 0.0) & (l2 <= 1.0)))

    def test_two_layer_linearly_separable(self):
        X = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
        y = np.array([[0], [0], [1], [1]])
        weights, output = train_two_layer(X, y, iterations=5000, alpha=1.0)
        mae = float(np.mean(np.abs(output - y)))
        self.assertLess(mae, 0.05)

    def test_xor_convergence_three_layer(self):
        X = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        y = np.array([[0], [1], [1], [0]])

        nn = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1, seed=1)
        res = nn.train(X, y, alpha=10.0, iterations=20000, log_interval=5000, track_direction_changes=True)

        self.assertLess(res["final_mae"], 0.01)
        self.assertEqual(nn.accuracy(X, y), 1.0)
        self.assertGreater(float(np.sum(res["synapse_0_direction_count"])), 0)
        self.assertGreater(float(np.sum(res["synapse_1_direction_count"])), 0)

    def test_prediction_threshold(self):
        nn = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1, seed=1)
        X = np.array([[0, 0, 1], [1, 1, 1]])
        y = np.array([[0], [0]])
        nn.train(X, y, alpha=1.0, iterations=5000, verbose=False)
        preds = nn.predict(X)
        self.assertEqual(preds.shape, (2, 1))
        self.assertTrue(all(p in [0, 1] for p in preds.flatten()))

    def test_momentum_acceleration(self):
        X = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        y = np.array([[0], [1], [1], [0]])

        nn_vanilla = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1, seed=123)
        res_vanilla = nn_vanilla.train(X, y, alpha=0.3, iterations=10000, momentum=0.0)

        nn_mom = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1, seed=123)
        res_mom = nn_mom.train(X, y, alpha=0.3, iterations=10000, momentum=0.8)

        # Momentum dampens oscillations and speeds progress down ravines
        self.assertLess(res_mom["final_mae"], res_vanilla["final_mae"])


if __name__ == "__main__":
    unittest.main()
