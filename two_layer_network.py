import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(output: np.ndarray) -> np.ndarray:
    return output * (1 - output)


def train_two_layer(X: np.ndarray, y: np.ndarray, iterations: int = 10000, alpha: float = 1.0):
    np.random.seed(1)
    synapse_0 = 2 * np.random.random((X.shape[1], 1)) - 1

    for iteration in range(iterations):
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        layer_1_error = layer_1 - y
        layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)
        synapse_0_derivative = np.dot(layer_0.T, layer_1_delta)

        synapse_0 -= alpha * synapse_0_derivative

        if (iteration % 2000) == 0:
            mean_error = np.mean(np.abs(layer_1_error))
            print(f"Iteration {iteration:5d} | MAE: {mean_error:.6f}")

    return synapse_0, layer_1


if __name__ == "__main__":
    print("=== Training 2-Layer Network on Linearly Correlated Data ===")
    X_linear = np.array([
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0]
    ])
    y_linear = np.array([[0], [0], [1], [1]])

    weights, output = train_two_layer(X_linear, y_linear, iterations=10000, alpha=1.0)
    print("\nLearned Weights:")
    print(weights)
    print("\nOutput (Target: [[0], [0], [1], [1]]):")
    print(np.round(output, 4))

    print("\n=== Testing 2-Layer Network on Non-Linear XOR Problem ===")
    # A single-layer perceptron without hidden units cannot form non-linear decision boundaries.
    X_xor = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    y_xor = np.array([[0], [1], [1], [0]])

    _, xor_output = train_two_layer(X_xor, y_xor, iterations=10000, alpha=1.0)
    print("\nOutput on XOR Problem (Target: [[0], [1], [1], [0]]):")
    print(np.round(xor_output, 4))
