"""
UBQC Protocol 1 - Ket-only QuTiP implementation (column-major order)
Compatible with QuTiP 5 (Python 3.13)
"""

import numpy as np
import random
from math import pi, sqrt
import qutip as qt


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def idx_col_major(x, y, n, m):
    """Column-major index for qubit at column x, row y."""
    return x * m + y


def plus_theta_ket(theta):
    """Return |+_theta> = (|0> + e^{i theta}|1>) / sqrt(2)."""
    return qt.Qobj(
        np.array([1.0, np.exp(1j * theta)]) / np.sqrt(2.0),
        dims=[[2], [1]]
    )


def two_qubit_cz_qobj(num_qubits, i, j):
    """Create a CZ acting on qubits i and j inside a num_qubit Hilbert space."""
    dim = 2 ** num_qubits
    diag = np.ones(dim, dtype=complex)

    for k in range(dim):
        bits = format(k, f'0{num_qubits}b')
        if bits[i] == '1' and bits[j] == '1':
            diag[k] = -1.0

    return qt.Qobj(np.diag(diag), dims=[[2]*num_qubits, [2]*num_qubits])


def embed_projector_on_qubit(P, num_qubits, target):
    """Embed a 2×2 projector P onto qubit 'target'."""
    ops = []
    for q in range(num_qubits):
        ops.append(P if q == target else qt.qeye(2))
    return qt.tensor(ops)


def measurement_projectors(delta):
    """Return |+δ>, |-δ> projectors."""
    ket_plus = plus_theta_ket(delta)
    ket_minus = qt.Qobj(
        np.array([1.0, -np.exp(1j * delta)]) / np.sqrt(2.0),
        dims=[[2], [1]]
    )
    return ket_plus * ket_plus.dag(), ket_minus * ket_minus.dag()


# ---------------------------------------------------------
# UBQC Simulation Function
# ---------------------------------------------------------

def run_ubqc_qutip(n=2, m=2, theta_list=None, phi_prime_list=None, seed=None, verbose=True):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_qubits = n * m

    # Alice's secret random angles
    if theta_list is None:
        allowed = [k * pi / 4 for k in range(8)]
        theta_list = [random.choice(allowed) for _ in range(num_qubits)]

    # φ' list (adaptive angles) — default all zero
    if phi_prime_list is None:
        phi_prime_list = [0.0] * num_qubits

    if verbose:
        print(f"UBQC running with n={n}, m={m}, total qubits={num_qubits}")
        print("Theta:", theta_list)

    # ---------------------------------------------------------
    # 1. Alice prepares ⊗ |+_theta>
    # ---------------------------------------------------------

    kets = [plus_theta_ket(theta_list[q]) for q in range(num_qubits)]
    state = qt.tensor(kets)

    # ---------------------------------------------------------
    # 2. Bob entangles using CZ (brickwork pattern)
    # ---------------------------------------------------------

    edges = []

    # vertical CZs
    for x in range(n):
        for y in range(m - 1):
            edges.append((idx_col_major(x, y, n, m),
                          idx_col_major(x, y + 1, n, m)))

    # horizontal CZs
    for x in range(n - 1):
        for y in range(m):
            edges.append((idx_col_major(x, y, n, m),
                          idx_col_major(x + 1, y, n, m)))

    edges = sorted(list(set(tuple(sorted(e)) for e in edges)))

    if verbose:
        print("CZ edges:", edges)

    # Apply all CZ gates
    for (i, j) in edges:
        CZ = two_qubit_cz_qobj(num_qubits, i, j)
        state = CZ * state

    if verbose:
        print("Bob finished entangling.\n")

    # ---------------------------------------------------------
    # 3. Interactive measurement
    # ---------------------------------------------------------

    measurement_outcomes = [None] * num_qubits
    s_bits = [0] * num_qubits
    log = []

    for x in range(n):
        for y in range(m):
            q = idx_col_major(x, y, n, m)

            phi_p = phi_prime_list[q]
            r = random.choice([0, 1])         # Alice's random bit
            delta = (phi_p + theta_list[q] + pi * r) % (2*pi)

            # Build measurement projectors
            Pp, Pm = measurement_projectors(delta)

            Mplus = embed_projector_on_qubit(Pp, num_qubits, q)
            Mminus = embed_projector_on_qubit(Pm, num_qubits, q)

            # Compute probabilities (QuTiP 5 returns a plain complex scalar)
            p_plus = complex(state.dag() * (Mplus * state)).real
            p_minus = complex(state.dag() * (Mminus * state)).real

            # Normalize and clamp
            p_plus = max(0.0, min(1.0, p_plus))
            p_minus = max(0.0, min(1.0, p_minus))

            # Bob samples outcome
            if random.random() < p_plus / (p_plus + p_minus):
                outcome = 0
                M = Mplus
            else:
                outcome = 1
                M = Mminus

            # Collapse
            new_state = M * state
            norm = new_state.norm()
            if norm != 0:
                new_state = new_state / norm
            state = new_state

            # Bob sends outcome to Alice
            measurement_outcomes[q] = outcome

            # Alice flips bit if r=1
            if r == 1:
                s_bits[q] ^= 1

            if verbose:
                print(f"q={q}, (x={x},y={y}) r={r} delta={delta:.4f}, "
                      f"outcome={outcome}, p+={p_plus:.4f}, p-={p_minus:.4f}")

            log.append({
                "qubit": q, "x": x, "y": y, "r": r,
                "delta": float(delta),
                "p_plus": p_plus,
                "p_minus": p_minus,
                "outcome": outcome
            })

    # End of execution
    if verbose:
        print("\nFinal measurement outcomes:", measurement_outcomes)
        print("Final s bits:", s_bits)

    return {
        "final_state": state,
        "outcomes": measurement_outcomes,
        "s_bits": s_bits,
        "theta": theta_list,
        "phi_prime": phi_prime_list,
        "edges": edges,
        "log": log
    }


# ---------------------------------------------------------
# Main execution (example)
# ---------------------------------------------------------

if __name__ == "__main__":
    result = run_ubqc_qutip(n=2, m=2, seed=5, verbose=True)
