from typing import List, Tuple, Optional, Dict
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(sigmoid_out: np.ndarray) -> np.ndarray:
    return sigmoid_out * (1.0 - sigmoid_out)


class NeuralNetwork:

    def __init__(self, input_dim: int = 3, hidden_dim: int = 4, output_dim: int = 1, seed: Optional[int] = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.synapse_0 = 2.0 * np.random.random((self.input_dim, self.hidden_dim)) - 1.0
        self.synapse_1 = 2.0 * np.random.random((self.hidden_dim, self.output_dim)) - 1.0

        self.synapse_0_direction_count = np.zeros_like(self.synapse_0)
        self.synapse_1_direction_count = np.zeros_like(self.synapse_1)

        self.prev_synapse_0_update = np.zeros_like(self.synapse_0)
        self.prev_synapse_1_update = np.zeros_like(self.synapse_1)

        self.loss_history: List[Tuple[int, float]] = []

    def reset_weights(self, seed: Optional[int] = None):
        effective_seed = seed if seed is not None else self.seed
        if effective_seed is not None:
            np.random.seed(effective_seed)

        self.synapse_0 = 2.0 * np.random.random((self.input_dim, self.hidden_dim)) - 1.0
        self.synapse_1 = 2.0 * np.random.random((self.hidden_dim, self.output_dim)) - 1.0
        self.synapse_0_direction_count = np.zeros_like(self.synapse_0)
        self.synapse_1_direction_count = np.zeros_like(self.synapse_1)
        self.prev_synapse_0_update = np.zeros_like(self.synapse_0)
        self.prev_synapse_1_update = np.zeros_like(self.synapse_1)
        self.loss_history.clear()

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        layer_1 = sigmoid(np.dot(X, self.synapse_0))
        layer_2 = sigmoid(np.dot(layer_1, self.synapse_1))
        return layer_1, layer_2

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.5,
        iterations: int = 60000,
        momentum: float = 0.0,
        log_interval: int = 10000,
        track_direction_changes: bool = True,
        verbose: bool = False
    ) -> Dict[str, any]:
        self.loss_history.clear()

        prev_update_0 = np.zeros_like(self.synapse_0)
        prev_update_1 = np.zeros_like(self.synapse_1)

        for j in range(iterations):
            layer_0 = X
            layer_1, layer_2 = self.forward(layer_0)

            layer_2_error = y - layer_2
            mae = float(np.mean(np.abs(layer_2_error)))

            if (j % log_interval) == 0 or j == iterations - 1:
                self.loss_history.append((j, mae))
                if verbose:
                    print(f"Iteration {j:5d} | MAE: {mae:.8f}")

            layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
            layer_1_error = layer_2_delta.dot(self.synapse_1.T)
            layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

            synapse_1_weight_update = layer_1.T.dot(layer_2_delta)
            synapse_0_weight_update = layer_0.T.dot(layer_1_delta)

            # A sign change between successive updates indicates stepping over an extremum.
            if track_direction_changes and j > 0:
                dir_change_0 = np.abs((synapse_0_weight_update > 0).astype(int) - (prev_update_0 > 0).astype(int))
                dir_change_1 = np.abs((synapse_1_weight_update > 0).astype(int) - (prev_update_1 > 0).astype(int))
                self.synapse_0_direction_count += dir_change_0
                self.synapse_1_direction_count += dir_change_1

            delta_syn1 = alpha * synapse_1_weight_update + momentum * self.prev_synapse_1_update
            delta_syn0 = alpha * synapse_0_weight_update + momentum * self.prev_synapse_0_update

            self.synapse_1 += delta_syn1
            self.synapse_0 += delta_syn0

            self.prev_synapse_1_update = delta_syn1
            self.prev_synapse_0_update = delta_syn0
            prev_update_0 = synapse_0_weight_update
            prev_update_1 = synapse_1_weight_update

        final_l1, final_l2 = self.forward(X)
        return {
            "loss_history": self.loss_history,
            "final_mae": float(np.mean(np.abs(y - final_l2))),
            "predictions": final_l2,
            "synapse_0": self.synapse_0.copy(),
            "synapse_1": self.synapse_1.copy(),
            "synapse_0_direction_count": self.synapse_0_direction_count.copy(),
            "synapse_1_direction_count": self.synapse_1_direction_count.copy(),
        }

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        _, probabilities = self.forward(X)
        return (probabilities >= threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
        preds = self.predict(X, threshold=threshold)
        return float(np.mean(preds == y))
