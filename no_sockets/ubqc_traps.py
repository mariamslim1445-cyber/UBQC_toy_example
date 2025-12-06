import math
import random

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt


# ======================================
# 1. QNode: UBQC-style cluster + traps
# ======================================

def make_ubqc_trap_qnode(total_qubits, logical_bits, data_edges):
    """
    total_qubits : int
        Total wires (data + traps).
    logical_bits : list[int]
        For each qubit i, 0 -> |+>, 1 -> |->
        (traps store their secret bit here).
    data_edges : list[tuple[int,int]]
        CZ entangling edges between *data* qubits only.
    """
    dev = qml.device("default.qubit", wires=total_qubits, shots=1)

    @qml.qnode(dev)
    def circuit(deltas):
        # --- Prepare all qubits in |+> or |-> ---
        for i in range(total_qubits):
            qml.Hadamard(wires=i)          # |0> -> |+>
            if logical_bits[i] == 1:       # encode |->
                qml.PauliZ(wires=i)

        # --- Entangle data qubits (cluster state) ---
        # Traps are NOT in data_edges, so they stay isolated.
        for u, v in data_edges:
            qml.CZ(wires=[u, v])

        # --- Measurement bases for ALL qubits ---
        for i in range(total_qubits):
            qml.RZ(-deltas[i], wires=i)
            qml.Hadamard(wires=i)

        # --- Measure all qubits at the end ---
        return qml.sample(wires=range(total_qubits))

    return circuit


# ======================================
# 2. One UBQC-style protocol run
# ======================================

def run_single_ubqc_protocol(
    num_data_qubits=4,
    num_trap_qubits=2,
    epsilon=0.0,
    rng=None,
):
    """
    Simulate one UBQC-style run:

      - Data qubits form a 1D cluster (line graph).
      - Trap qubits are isolated single qubits.
      - Honest angles δ_i = π r_i  (φ_i = θ_i = 0, identity computation).
      - Cheating Bob uses δ'_i = δ_i + epsilon on ALL qubits.

    Returns:
        trap_passed : bool
        data_output : list[int]   # logical data bits after unblinding
    """
    if rng is None:
        rng = random

    total_qubits = num_data_qubits + num_trap_qubits
    indices = list(range(total_qubits))

    # --- Randomly choose trap positions ---
    trap_positions = set(rng.sample(indices, num_trap_qubits))
    data_positions = sorted([i for i in indices if i not in trap_positions])

    # --- Data edges: 1D chain cluster on data_positions ---
    # Example: data_positions = [0,3,5,7] -> edges (0,3), (3,5), (5,7)
    data_edges = [
        (data_positions[k], data_positions[k + 1])
        for k in range(len(data_positions) - 1)
    ]

    # --- Logical bits: traps store secret bits, data = 0 (|+>) for now ---
    logical_bits = [0] * total_qubits
    trap_bits = {}

    for idx in trap_positions:
        bit = rng.randint(0, 1)      # secret trap bit
        trap_bits[idx] = bit
        logical_bits[idx] = bit      # encode |+> or |->

    # (You could also randomize logical bits on data if you want.)

    # --- One-time pad bits r_i for blindness ---
    r_bits = [rng.randint(0, 1) for _ in indices]

    # UBQC form would be δ_i = φ_i + θ_i + π r_i + corrections.
    # Here: φ_i = 0 (identity), θ_i = 0, corrections already absorbed,
    # so "honest" angles are just δ_i = π r_i.
    honest_deltas = np.array([math.pi * r for r in r_bits], dtype=float)

    # --- Cheating Bob shifts all angles by epsilon ---
    cheated_deltas = honest_deltas + float(epsilon)

    # --- Build and run the QNode ---
    ubqc_qnode = make_ubqc_trap_qnode(total_qubits, logical_bits, data_edges)

    raw_measurements = ubqc_qnode(cheated_deltas)
    raw_array = np.array(raw_measurements)      # shape (1, total_qubits)
    m_bits = [int(b) for b in raw_array[0]]

    # --- Alice unblinds logical outcomes: s_i = m_i XOR r_i ---
    logical_out = [m_bits[i] ^ r_bits[i] for i in indices]

    # --- Check traps ---
    trap_passed = True
    for idx in trap_positions:
        if logical_out[idx] != trap_bits[idx]:
            trap_passed = False
            break

    # --- Collect data outputs in order of data_positions ---
    data_output = [logical_out[i] for i in data_positions]

    return trap_passed, data_output


# ======================================
# 3. Monte Carlo for P(cheat undetected)
# ======================================

def estimate_undetected_probability(
    num_data_qubits,
    num_trap_qubits,
    epsilon,
    runs=200,
    seed=None,
):
    rng = random.Random(seed)
    undetected = 0

    for _ in range(runs):
        trap_passed, _ = run_single_ubqc_protocol(
            num_data_qubits=num_data_qubits,
            num_trap_qubits=num_trap_qubits,
            epsilon=epsilon,
            rng=rng,
        )
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
    if trap_counts is None:
        trap_counts = [1, 2, 3, 4, 5]

    if epsilons is None:
        epsilons = np.linspace(0.0, math.pi, 13)

    probs = np.zeros((len(trap_counts), len(epsilons)), dtype=float)
    base_rng = random.Random(seed)

    for i, T in enumerate(trap_counts):
        for j, eps in enumerate(epsilons):
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


# ======================================
# 4. Plotting
# ======================================

def plot_lines_fixed_traps(trap_counts, epsilons, probs):
    for i, T in enumerate(trap_counts):
        plt.plot(
            epsilons,
            probs[i],
            marker="o",
            linestyle="-",
            label=f"MC T={T}",
        )

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
    plt.title("Cheat undetected vs deviation & traps (UBQC-style cluster)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_heatmap(trap_counts, epsilons, probs):
    plt.imshow(
        probs,
        extent=[epsilons[0], epsilons[-1], trap_counts[0], trap_counts[-1]],
        aspect="auto",
        origin="lower",
    )
    plt.colorbar(label="P(cheat undetected)")
    plt.xlabel("epsilon (radians)")
    plt.ylabel("number of traps T")
    plt.title("P(cheat undetected) heatmap (UBQC-style)")
    plt.tight_layout()
    plt.show()


# ======================================
# 5. Example main
# ======================================

if __name__ == "__main__":
    num_data_qubits = 4
    trap_counts = [1, 2, 3, 4, 5]
    epsilons = np.linspace(0.0, math.pi, 13)

    trap_counts, epsilons, probs = sweep_traps_and_deviation(
        num_data_qubits=num_data_qubits,
        trap_counts=trap_counts,
        epsilons=epsilons,
        runs=500,
        seed=42,
    )

    plot_lines_fixed_traps(trap_counts, epsilons, probs)
    # or:
    # plot_heatmap(trap_counts, epsilons, probs)
