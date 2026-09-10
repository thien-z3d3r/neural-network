# Neural Network with Gradient Descent

A Python 3 implementation of a 3-layer neural network built from scratch using NumPy and Gradient Descent, based on Andrew Trask's tutorial:
["A Neural Network in 13 lines of Python (Part 2 - Gradient Descent)"](https://iamtrask.github.io/2015/07/27/python-network-part2/) from [practical-tutorials/project-based-learning](https://github.com/practical-tutorials/project-based-learning#python).

---

## Theoretical Background

### 1. Backpropagation vs. Gradient Descent
- **Backpropagation**: Propagates error backwards from the output layer through earlier layers, computing how each weight contributes to the total error via the chain rule ($\frac{\partial E}{\partial w}$). It does not modify weights.
- **Gradient Descent**: The optimization routine that updates the network parameters using the gradients supplied by backpropagation:
  $$w \leftarrow w - \alpha \cdot \frac{\partial E}{\partial w}$$
  where $\alpha$ is the learning rate.

---

### 2. Optimization Challenges and Solutions

| Problem | Observation | Mathematical Cause | Solution |
| :--- | :--- | :--- | :--- |
| **Slopes Too Big** | Divergence / numerical overflow | Large gradient steps overshoot the local minimum and land on steeper opposing slopes. | **Tune Alpha ($\alpha$)**: Scale gradient updates by a fraction ($0 < \alpha \le 10$). |
| **Local Minima** | Suboptimal convergence | Non-convex loss landscape traps gradient trajectories. | **Increase Hidden Layer Capacity**: More hidden units provide multiple random initial positions, expanding exploratory coverage. |
| **Slopes Too Small** | Stalled training / slow progress | Tiny gradients or excessively conservative $\alpha$ produce negligible updates. | **Increase Alpha & Use Momentum**: Larger step sizes and momentum accumulation push updates through flat regions. |

---

### 3. Derivative Update Direction Changes

Tracking the sign reversals of weight updates indicates optimization dynamics:
$$\text{Direction Change} = | \mathbb{I}(\Delta w^{(t)} > 0) - \mathbb{I}(\Delta w^{(t-1)} > 0) |$$

- **Zero or Minimal Flips ($\alpha = 0.001$)**: Step size is too small; weights move monotonically and make negligible progress.
- **Moderate Flips ($\alpha = 1.0, 10.0$)**: Optimal step size; weights oscillate across the minimum as they converge.
- **Excessive Flips or Saturation ($\alpha \ge 100$)**: Steps oscillate wildly or push sigmoid activations into flat regions where derivatives vanish.

---

## Repository Structure

- [`toy_network.py`](file:///C:/Users/Administrator/Desktop/mini%20projects/neural-network/toy_network.py): Concise implementation of the 3-layer XOR network.
- [`two_layer_network.py`](file:///C:/Users/Administrator/Desktop/mini%20projects/neural-network/two_layer_network.py): 2-layer network illustrating linear separation limits on XOR.
- [`network.py`](file:///C:/Users/Administrator/Desktop/mini%20projects/neural-network/network.py): Modular `NeuralNetwork` class supporting layer configuration, gradient tracking, and momentum.
- [`experiments.py`](file:///C:/Users/Administrator/Desktop/mini%20projects/neural-network/experiments.py): Benchmark suite evaluating learning rate sensitivity, hidden layer dimensions, and momentum.
- [`visualize.py`](file:///C:/Users/Administrator/Desktop/mini%20projects/neural-network/visualize.py): Visualization generator for loss trajectories, direction reversals, and the 3D error surface.
- [`test_network.py`](file:///C:/Users/Administrator/Desktop/mini%20projects/neural-network/test_network.py): Unit test suite.

---

## Execution Instructions

### 1. Run the Baseline 13-Line Network
```bash
python toy_network.py
```

### 2. Run the Experiment Suite
```bash
python experiments.py
```

### 3. Generate Visualization Plots
```bash
python visualize.py
```
Generated plots are saved to `./images/`:
- `alpha_convergence.png`: Training loss curves on a logarithmic scale.
- `direction_changes.png`: Total sign reversals across learning rates.
- `hidden_dim_comparison.png`: Convergence comparison between 4 and 32 hidden units.
- `error_surface.png`: 3D error surface of a 2-parameter network.

### 4. Run Unit Tests
```bash
python -m unittest test_network.py
```

---

## Summary of Empirical Results

| Alpha ($\alpha$) | Initial MAE | Final MAE (60k) | Synapse 0 Flips | Synapse 1 Flips | Optimization Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.001** | 0.496410 | 0.482132 | 3 | 3 | Underfitting; step size too small |
| **0.01** | 0.496410 | 0.075954 | 14 | 4 | Steady but slow convergence |
| **0.1** | 0.496410 | 0.011631 | 14 | 4 | Reliable convergence |
| **1.0** | 0.496410 | 0.003184 | 14 | 4 | Fast convergence |
| **10.0** | 0.496410 | **0.001192** | 136 | 56 | Optimal step size; lowest loss |
| **100.0** | 0.496410 | 0.125188 | 96 | 46 | Overshooting; stuck in suboptimal plateau |
| **1000.0** | 0.496410 | 0.500000 | 33 | 24 | Complete divergence; activations saturated |
