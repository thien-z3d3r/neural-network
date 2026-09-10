import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from network import NeuralNetwork, sigmoid


def get_xor_dataset():
    X = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    y = np.array([[0], [1], [1], [0]])
    return X, y


def plot_alpha_convergence(output_dir: str):
    X, y = get_xor_dataset()
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    plt.figure(figsize=(10, 6))

    colors = {
        0.001: "#1f77b4",
        0.01: "#ff7f0e",
        0.1: "#2ca02c",
        1.0: "#d62728",
        10.0: "#9467bd",
        100.0: "#8c564b",
        1000.0: "#e377c2"
    }

    for alpha in alphas:
        nn = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1, seed=1)
        res = nn.train(X, y, alpha=alpha, iterations=60000, log_interval=1000, track_direction_changes=False)
        history = res["loss_history"]
        iters = [h[0] for h in history]
        losses = [h[1] for h in history]
        plt.plot(iters, losses, label=f"alpha = {alpha}", color=colors[alpha], linewidth=2.0)

    plt.title("Convergence Across Different Learning Rates (Alpha) on XOR", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "alpha_convergence.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_direction_changes(output_dir: str):
    X, y = get_xor_dataset()
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    syn0_flips = []
    syn1_flips = []

    for alpha in alphas:
        nn = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1, seed=1)
        res = nn.train(X, y, alpha=alpha, iterations=60000, log_interval=10000, track_direction_changes=True)
        syn0_flips.append(int(np.sum(res["synapse_0_direction_count"])))
        syn1_flips.append(int(np.sum(res["synapse_1_direction_count"])))

    x = np.arange(len(alphas))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, syn0_flips, width, label="Synapse 0 Updates", color="#3498db")
    plt.bar(x + width/2, syn1_flips, width, label="Synapse 1 Updates", color="#e74c3c")

    plt.title("Derivative Update Direction Changes Across Learning Rates", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Alpha (Learning Rate)", fontsize=12)
    plt.ylabel("Total Direction Changes (Sign Flips)", fontsize=12)
    plt.xticks(x, [str(a) for a in alphas])
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "direction_changes.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_hidden_dim_comparison(output_dir: str):
    X, y = get_xor_dataset()
    alpha = 10.0
    iterations = 60000

    nn_4 = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1, seed=1)
    res_4 = nn_4.train(X, y, alpha=alpha, iterations=iterations, log_interval=500, track_direction_changes=False)

    nn_32 = NeuralNetwork(input_dim=3, hidden_dim=32, output_dim=1, seed=1)
    res_32 = nn_32.train(X, y, alpha=alpha, iterations=iterations, log_interval=500, track_direction_changes=False)

    iters = [h[0] for h in res_4["loss_history"]]
    loss_4 = [h[1] for h in res_4["loss_history"]]
    loss_32 = [h[1] for h in res_32["loss_history"]]

    plt.figure(figsize=(10, 6))
    plt.plot(iters, loss_4, label="Hidden Dim = 4", color="#e67e22", linewidth=2.0)
    plt.plot(iters, loss_32, label="Hidden Dim = 32", color="#2ecc71", linewidth=2.0)

    plt.title("Effect of Hidden Layer Capacity: 4 vs 32 Hidden Units (Alpha = 10)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean Absolute Error (log scale)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "hidden_dim_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_error_surface(output_dir: str):
    # Error surface for 2-weight network on linearly separable dataset
    X = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    y = np.array([[0], [0], [1], [1]])

    w1_vals = np.linspace(-10, 10, 80)
    w2_vals = np.linspace(-10, 10, 80)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = np.zeros_like(W1)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            weights = np.array([[W1[i, j]], [W2[i, j]]])
            preds = sigmoid(np.dot(X, weights))
            Z[i, j] = np.mean(np.abs(preds - y))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(W1, W2, Z, cmap="viridis", edgecolor="none", alpha=0.85)

    ax.set_title("3D Error Surface for 2-Weight Network", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Weight 1 (synapse_0[0])", fontsize=11, labelpad=8)
    ax.set_ylabel("Weight 2 (synapse_0[1])", fontsize=11, labelpad=8)
    ax.set_zlabel("Mean Absolute Error", fontsize=11, labelpad=8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="MAE Loss")

    out_path = os.path.join(output_dir, "error_surface.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main():
    img_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(img_dir, exist_ok=True)

    p1 = plot_alpha_convergence(img_dir)
    p2 = plot_direction_changes(img_dir)
    p3 = plot_hidden_dim_comparison(img_dir)
    p4 = plot_error_surface(img_dir)

    artifact_dir = r"C:\Users\Administrator\.gemini\antigravity-cli\brain\ee6b13e0-778a-419c-9df5-db70941f6e87"
    if os.path.exists(artifact_dir):
        for p in [p1, p2, p3, p4]:
            dest = os.path.join(artifact_dir, os.path.basename(p))
            shutil.copyfile(p, dest)


if __name__ == "__main__":
    main()
