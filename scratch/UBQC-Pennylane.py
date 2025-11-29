import pennylane as qml
import numpy as np
import random

# -----------------------------
# Parameters
# -----------------------------
n, m = 2, 2  # 2x2 grid
num_qubits = n * m
seed = 42
random.seed(seed)
np.random.seed(seed)

# Alice's secret theta (random from {k*pi/4})
theta_list = [random.choice([k*np.pi/4 for k in range(8)]) for _ in range(num_qubits)]

# φ' list (here just zeros for simplicity)
phi_prime_list = [0.0] * num_qubits

# Define brickwork edges (2x2 example)
edges = [(0,1),(0,2),(1,3),(2,3)]

# Alice random r bits
r_list = [random.choice([0,1]) for _ in range(num_qubits)]

# -----------------------------
# PennyLane device
# -----------------------------
# Use shots=1 so qml.sample returns a single 0 or 1 per qubit
dev = qml.device("default.qubit", wires=num_qubits, shots=1)

# -----------------------------
# UBQC quantum node
# -----------------------------
@qml.qnode(dev)
def ubqc_circuit(theta_list, delta_list):
    # 1. Alice prepares |+_θ> states
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(theta_list[i], wires=i)

    # 2. Bob applies CZ entanglement (brickwork)
    for (i,j) in edges:
        qml.CZ(wires=[i,j])

    # 3. Bob measures qubits in rotated bases ±δ
    # Implemented via RY rotation + standard Z measurement
    for i in range(num_qubits):
        # Rotate basis so that measuring Z is equivalent to measuring ±δ
        qml.RY(-2*delta_list[i], wires=i)

    # Return measurement samples for all qubits
    return qml.sample(wires=range(num_qubits))

# -----------------------------
# Run UBQC
# -----------------------------
delta_list = [(phi_prime_list[i] + theta_list[i] + np.pi*r_list[i]) % (2*np.pi)
              for i in range(num_qubits)]

# Run the circuit
raw_results = ubqc_circuit(theta_list, delta_list)

# Alice applies r corrections
measurement_outcomes = []
s_bits = []
for i in range(num_qubits):
    corrected = int(raw_results[i]) ^ r_list[i]
    measurement_outcomes.append(corrected)
    s_bits.append(corrected)

# -----------------------------
# Display results
# -----------------------------
print("Alice's secret theta:", np.round(theta_list,3))
print("Alice's r bits:", r_list)
print("Delta values:", np.round(delta_list,3))
print("Raw outcomes from Bob:", raw_results)
print("Corrected outcomes (Alice's s bits):", measurement_outcomes)
