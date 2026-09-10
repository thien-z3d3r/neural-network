import numpy as np
from network import NeuralNetwork


def get_xor_dataset():
    # Columns 0 and 1 are binary inputs; column 2 is a bias column.
    X = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    y = np.array([[0], [1], [1], [0]])
    return X, y


def run_experiment_alpha_tuning():
    print("=" * 80)
    print("EXPERIMENT 1: Alpha Parameter Tuning & Gradient Direction Analysis")
    print("=" * 80)

    X, y = get_xor_dataset()
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    results = {}

    for alpha in alphas:
        print(f"\n--- Training with Alpha: {alpha} ---")
        nn = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1, seed=1)

        res = nn.train(
            X, y,
            alpha=alpha,
            iterations=60000,
            log_interval=10000,
            track_direction_changes=True,
            verbose=True
        )

        results[alpha] = {
            "final_mae": res["final_mae"],
            "predictions": res["predictions"],
            "syn0": res["synapse_0"],
            "syn1": res["synapse_1"],
            "syn0_dirs": res["synapse_0_direction_count"],
            "syn1_dirs": res["synapse_1_direction_count"],
            "history": res["loss_history"]
        }

        print(f"\nFinal Weights Synapse 0 (Alpha = {alpha}):")
        print(np.round(res["synapse_0"], 4))
        print(f"Synapse 0 Direction Changes:")
        print(res["synapse_0_direction_count"].astype(int))

        print(f"\nFinal Weights Synapse 1 (Alpha = {alpha}):")
        print(np.round(res["synapse_1"], 4))
        print(f"Synapse 1 Direction Changes:")
        print(res["synapse_1_direction_count"].astype(int))

        print(f"\nPredictions for Alpha = {alpha}:")
        print(np.round(res["predictions"], 4).flatten(), "Target: [0, 1, 1, 0]")

    print("\n" + "-" * 80)
    print("SUMMARY OF ALPHA TUNING EXPERIMENT:")
    print("-" * 80)
    print(f"{'Alpha':<10} | {'Initial Error':<15} | {'Final Error (60k)':<18} | {'Total Syn0 Flips':<16} | {'Total Syn1 Flips':<16}")
    print("-" * 80)
    for alpha in alphas:
        r = results[alpha]
        init_err = r["history"][0][1]
        final_err = r["final_mae"]
        s0_flips = int(np.sum(r["syn0_dirs"]))
        s1_flips = int(np.sum(r["syn1_dirs"]))
        print(f"{alpha:<10} | {init_err:<15.6f} | {final_err:<18.6f} | {s0_flips:<16} | {s1_flips:<16}")
    print("-" * 80)
    return results


def run_experiment_hidden_layer_size():
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Hidden Layer Dimension (Capacity & Optimization Space)")
    print("=" * 80)

    X, y = get_xor_dataset()
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    comparison = {}

    for hidden_dim in [4, 32]:
        print(f"\n--- Testing Hidden Dimension: {hidden_dim} ---")
        comparison[hidden_dim] = {}
        for alpha in alphas:
            nn = NeuralNetwork(input_dim=3, hidden_dim=hidden_dim, output_dim=1, seed=1)
            res = nn.train(
                X, y,
                alpha=alpha,
                iterations=60000,
                log_interval=10000,
                track_direction_changes=False,
                verbose=False
            )
            comparison[hidden_dim][alpha] = res["final_mae"]
            print(f"Hidden={hidden_dim:2d} | Alpha={alpha:<6} | Final MAE: {res['final_mae']:.8f}")

    print("\n" + "-" * 80)
    print("HIDDEN DIMENSION COMPARISON TABLE (Final MAE after 60k iterations):")
    print("-" * 80)
    print(f"{'Alpha':<10} | {'Hidden Dim = 4':<20} | {'Hidden Dim = 32':<20} | {'Improvement':<15}")
    print("-" * 80)
    for alpha in alphas:
        err4 = comparison[4][alpha]
        err32 = comparison[32][alpha]
        diff = err4 - err32
        note = f"{diff:+.6f}" if abs(diff) > 1e-6 else "identical"
        print(f"{alpha:<10} | {err4:<20.8f} | {err32:<20.8f} | {note:<15}")
    print("-" * 80)
    return comparison


def run_experiment_momentum():
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Momentum vs. Vanilla Gradient Descent")
    print("=" * 80)

    X, y = get_xor_dataset()
    alpha = 0.5
    iterations = 20000

    nn_vanilla = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1, seed=42)
    res_vanilla = nn_vanilla.train(
        X, y, alpha=alpha, iterations=iterations, momentum=0.0, log_interval=5000, verbose=False
    )

    nn_momentum = NeuralNetwork(input_dim=3, hidden_dim=4, output_dim=1, seed=42)
    res_momentum = nn_momentum.train(
        X, y, alpha=alpha, iterations=iterations, momentum=0.8, log_interval=5000, verbose=False
    )

    print(f"{'Iteration':<10} | {'Vanilla GD Loss':<20} | {'Momentum (0.8) Loss':<20}")
    print("-" * 55)
    for (step, loss_v), (_, loss_m) in zip(res_vanilla["loss_history"], res_momentum["loss_history"]):
        print(f"{step:<10} | {loss_v:<20.6f} | {loss_m:<20.6f}")

    print("-" * 55)
    print(f"Final Vanilla Loss:  {res_vanilla['final_mae']:.6f}")
    print(f"Final Momentum Loss: {res_momentum['final_mae']:.6f}")


if __name__ == "__main__":
    run_experiment_alpha_tuning()
    run_experiment_hidden_layer_size()
    run_experiment_momentum()
