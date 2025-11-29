import numpy as np
from math import pi, sqrt
import random

# --- Utility functions ---

def plus_theta_state(theta):
    """|+_theta> = (|0> + e^{i theta}|1>)/sqrt(2)"""
    return np.array([1.0, np.exp(1j * theta)]) / sqrt(2.0)

def kron_n(mats):
    """Kronecker product of a list of matrices/vectors"""
    res = mats[0]
    for m in mats[1:]:
        res = np.kron(res, m)
    return res

def cz_diag_operator(num_qubits, i, j):
    """Construct diagonal CZ(i,j) on n qubits"""
    dim = 2**num_qubits
    diag = np.ones(dim, dtype=complex)
    for idx in range(dim):
        b = format(idx, f'0{num_qubits}b')
        if int(b[i]) & int(b[j]):
            diag[idx] = -1
    return np.diag(diag)

def apply_cz_edges(state, num_qubits, edges):
    """Apply a list of CZ gates to a density matrix"""
    rho = state
    for (i, j) in edges:
        U = cz_diag_operator(num_qubits, i, j)
        rho = U @ rho @ U.conj().T
    return rho

def plus_minus_projectors(delta):
    """Measurement projectors |+_delta>, |- _delta>"""
    ket_plus = np.array([1, np.exp(1j * delta)]) / sqrt(2)
    ket_minus = np.array([1, -np.exp(1j * delta)]) / sqrt(2)
    return (
        np.outer(ket_plus, ket_plus.conj()),
        np.outer(ket_minus, ket_minus.conj()),
    )

def measure_qubit(state, num_qubits, qubit, delta):
    """Measure one qubit in |±_delta> basis"""
    Pp, Pm = plus_minus_projectors(delta)

    ops_p = [np.eye(2) for _ in range(num_qubits)]
    ops_m = [np.eye(2) for _ in range(num_qubits)]
    ops_p[qubit] = Pp
    ops_m[qubit] = Pm

    M_plus = kron_n(ops_p)
    M_minus = kron_n(ops_m)

    p_plus = float(np.clip(np.real(np.trace(M_plus @ state)), 0, 1))
    p_minus = float(np.clip(np.real(np.trace(M_minus @ state)), 0, 1))

    # Sample outcome
    if p_plus + p_minus == 0:
        outcome = 0
    else:
        outcome = 0 if random.random() < p_plus / (p_plus + p_minus) else 1

    # Post-measurement state
    M = M_plus if outcome == 0 else M_minus
    new_rho = M @ state @ M
    norm = np.trace(new_rho)
    if norm > 0:
        new_rho /= norm

    return outcome, new_rho, p_plus, p_minus


# --- UBQC Demo Parameters (2×2) ---

n, m = 2, 2
num_qubits = n * m
idx = lambda x, y: x*m + y  # column-major

# Alice chooses secret random θ from multiples of π/4
theta_choices = [k * pi/4 for k in range(8)]
theta = [random.choice(theta_choices) for _ in range(num_qubits)]

print("Alice's secret θ:", theta)

# Initial product state
psi = kron_n([plus_theta_state(t) for t in theta])
rho = np.outer(psi, psi.conj())

# Bob's entangling edges (brickwork-like)
edges = []
for x in range(n):
    for y in range(m-1):
        edges.append((idx(x,y), idx(x,y+1)))
for x in range(n-1):
    for y in range(m):
        edges.append((idx(x,y), idx(x+1,y)))

edges = sorted(list(set(edges)))
print("CZ edges:", edges)

rho = apply_cz_edges(rho, num_qubits, edges)
print("Entanglement applied.\n")

# --- Measurement phase ---

phi_prime = [0]*num_qubits
s = [0]*num_qubits
results = [None]*num_qubits

for x in range(n):
    for y in range(m):
        q = idx(x,y)

        r = random.choice([0,1])
        delta = (phi_prime[q] + theta[q] + pi*r) % (2*pi)

        outcome, rho, p_plus, p_minus = measure_qubit(rho, num_qubits, q, delta)
        results[q] = outcome

        if r == 1:
            s[q] ^= 1

        print(f"Qubit q={q} (x={x},y={y}): r={r}, δ={delta:.3f}, outcome={outcome}, p+={p_plus:.3f}, p-={p_minus:.3f}")

print("\nFinal measurement outcomes:", results)
print("Alice's final s bits:", s)
