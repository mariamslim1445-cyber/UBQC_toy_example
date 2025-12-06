import math
import random

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt


# ==========================
# 1. QNode factory (Bob)
# ==========================

def make_blind_trap_qnode(total_qubits, logical_bits):
    dev = qml.device("default.qubit", wires=total_qubits, shots=1)

    @qml.qnode(dev)
    def circuit(deltas):
        # 1) Prepare logical states
        for i in range(total_qubits):
            qml.Hadamard(wires=i)
            if logical_bits[i] == 1:
                qml.PauliZ(wires=i)

        # 2) Apply all measurement bases
        for i in range(total_qubits):
            qml.RZ(-deltas[i], wires=i)
            qml.Hadamard(wires=i)

        # 3) Measure ALL qubits at the end
        return qml.sample(wires=range(total_qubits))

    return circuit



# ==========================
# 2. Single protocol run
# ==========================

def run_single_protocol(
    num_data_qubits=4,
    num_trap_qubits=2,
    epsilon=0.0,
    rng=None,
):
    """
    Run one full blind+trap protocol instance with:
    - num_data_qubits data qubits
    - num_trap_qubits trap qubits
    - Bob cheating by shifting all measurement angles by epsilon

    Returns:
        trap_passed (bool),
        data_output (list[int])  # logical data bits after unblinding
    """
    if rng is None:
        rng = random

    total_qubits = num_data_qubits + num_trap_qubits
    indices = list(range(total_qubits))

    # Choose random trap positions
    trap_positions = set(rng.sample(indices, num_trap_qubits))
    data_positions = [i for i in indices if i not in trap_positions]

    # Logical bits: data bits we fix (e.g. all 0), trap bits random
    logical_bits = [0] * total_qubits
    trap_bits = {}

    for idx in trap_positions:
        bit = rng.randint(0, 1)
        trap_bits[idx] = bit
        logical_bits[idx] = bit  # traps encode their own secret bit

    # (Optional) you can set data logical bits however you want; here all zeros
    # for idx in data_positions:
    #     logical_bits[idx] = 0

    # One-time pad bits r_i for blindness
    r_bits = [rng.randint(0, 1) for _ in indices]

    # Honest deltas: δ_i = π r_i  (no extra computation angle to keep it simple)
    honest_deltas = np.array([math.pi * r for r in r_bits], dtype=float)

    # Cheating Bob: apply same shift epsilon on ALL qubits (he doesn't know traps)
    cheated_deltas = honest_deltas + float(epsilon)

    # Build Bob's QNode for THIS choice of logical bits
    blind_qnode = make_blind_trap_qnode(total_qubits, logical_bits)

    raw_measurements = blind_qnode(cheated_deltas)
    raw_array = np.array(raw_measurements)  # shape (1, total_qubits)
    m_bits = [int(b) for b in raw_array[0]]

    # Alice unblinds: s_i = m_i XOR r_i
    logical_out = [m_bits[i] ^ r_bits[i] for i in indices]

    # Check traps
    trap_passed = True
    for idx in trap_positions:
        if logical_out[idx] != trap_bits[idx]:
            trap_passed = False
            break

    # Collect data output in order of data_positions
    data_output = [logical_out[i] for i in data_positions]

    return trap_passed, data_output


# ==========================
# 3. Monte Carlo estimation
# ==========================

def estimate_undetected_probability(
    num_data_qubits,
    num_trap_qubits,
    epsilon,
    runs=200,
    seed=None,
):
    """
    Estimate P(cheat undetected) for given:
      - number of traps
      - deviation epsilon
    via Monte Carlo over `runs` repetitions.
    """
    rng = random.Random(seed)
    undetected = 0

    for _ in range(runs):
        trap_passed, _ = run_single_protocol(
            num_data_qubits=num_data_qubits,
            num_trap_qubits=num_trap_qubits,
            epsilon=epsilon,
            rng=rng,
        )
        # "cheat undetected" = traps all passed even though epsilon != 0
        if trap_passed:
            undetected += 1

    return undetected / runs


def sweep_traps_and_deviation(
    num_data_qubits=4,
    trap_counts=None,
    epsilons=None,
    runs=500,
    seed=1337,
):
    """
    Grid over:
      - trap_counts = [T1, T2, ...]
      - epsilons    = [eps1, eps2, ...]
    and estimate P_undetected(T, eps).
    """
    if trap_counts is None:
        trap_counts = [1, 2, 3, 4, 5]

    if epsilons is None:
        epsilons = np.linspace(0.0, math.pi, 13)  # 0, π/12, ..., π

    probs = np.zeros((len(trap_counts), len(epsilons)), dtype=float)
    base_rng = random.Random(seed)

    for i, T in enumerate(trap_counts):
        for j, eps in enumerate(epsilons):
            # use different seed per (T, eps) so runs are reproducible-ish
            cell_seed = base_rng.randint(0, 10**9)
            p = estimate_undetected_probability(
                num_data_qubits=num_data_qubits,
                num_trap_qubits=T,
                epsilon=eps,
                runs=runs,
                seed=cell_seed,
            )
            probs[i, j] = p
            print(f"T={T:2d}, eps={eps:.3f} → P_undetected ≈ {p:.4f}")

    return trap_counts, epsilons, probs


# ==========================
# 4. Plotting helpers
# ==========================

def plot_lines_fixed_traps(trap_counts, epsilons, probs):
    """
    For each T, plot P_undetected(eps) as a line,
    and overlay the theoretical curve (cos^2(eps/2))^T.
    """
    for i, T in enumerate(trap_counts):
        plt.plot(
            epsilons,
            probs[i],
            marker="o",
            linestyle="-",
            label=f"MC T={T}",
        )

        # theoretical curve
        theory = (np.cos(epsilons / 2.0) ** 2) ** T
        plt.plot(
            epsilons,
            theory,
            linestyle="--",
            alpha=0.7,
            label=f"theory T={T}",
        )

    plt.xlabel("epsilon (radians)")
    plt.ylabel("P(cheat undetected)")
    plt.title("Cheat undetected vs deviation & traps")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_heatmap(trap_counts, epsilons, probs):
    """
    Optional: 2D heatmap of P_undetected(T, eps).
    """
    plt.imshow(
        probs,
        extent=[epsilons[0], epsilons[-1], trap_counts[0], trap_counts[-1]],
        aspect="auto",
        origin="lower",
    )
    plt.colorbar(label="P(cheat undetected)")
    plt.xlabel("epsilon (radians)")
    plt.ylabel("number of traps T")
    plt.title("P(cheat undetected) heatmap")
    plt.tight_layout()
    plt.show()


# ==========================
# 5. Example main
# ==========================

if __name__ == "__main__":
    # Example usage:
    num_data_qubits = 4
    trap_counts = [1, 2, 3, 4, 5]   # vary how many traps
    epsilons = np.linspace(0.0, math.pi, 13)  # vary deviation

    trap_counts, epsilons, probs = sweep_traps_and_deviation(
        num_data_qubits=num_data_qubits,
        trap_counts=trap_counts,
        epsilons=epsilons,
        runs=500,     # increase for smoother curves
        seed=42,
    )

    # Plot slices for each T with theory overlay
    plot_lines_fixed_traps(trap_counts, epsilons, probs)

    # Or, if you like a 2D view:
    # plot_heatmap(trap_counts, epsilons, probs)
