# client_Alice_fixed.py
import pennylane as qml
import numpy as np
import random
import json
import socket
import time
import math
import os  # NEW

# --------------------------
# Basic MBQC parameters
# --------------------------
n, m = 2, 5  # original grid dims
data_num_qubits = n * m

# Dependency lists for your brickwork flow (these should match your original)
# Example sets (matching your previous file); adjust if your original differs
D_X = [[], [], [0], [1], [2], [3], [4], [5], [6], [7]]
D_Z = [[], [], [], [], [0,3], [1,2], [2], [3], [4,7], [5,6]]
phi_list = [np.pi/4, 0, np.pi/4, 0, np.pi/4, 0, 0, 0, 0, 0]

seed = 30
random.seed(seed)
np.random.seed(seed)

# Alice's secret thetas for data qubits
theta_list = [random.choice([k*np.pi/4 for k in range(8)]) for _ in range(data_num_qubits)]

# helper
def wrap_2pi_floor(x):
    return x - math.floor(x / (2*math.pi)) * (2*math.pi)

# helper prepare single-qubit |+_theta>
def make_plus_theta_state(theta):
    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev)
    def qubit_state():
        qml.Hadamard(wires=0)
        qml.RZ(theta, wires=0)
        return qml.state()
    return qubit_state()

def state_zero():
    return np.array([0.707+0j, 0.707+0j])

def state_one():
    return np.array([0.707+0j, -0.707+0j])

# --------------------------
# Verification params
# --------------------------
num_traps = int(os.getenv("NUM_TRAPS", "3"))   # traps per run
s_reps = int(os.getenv("S_REPS", "4"))         # repetitions (security parameter)

HOST = socket.gethostbyname(socket.gethostname())
PORT = 5050

def recv_json(sock):
    buffer = ""
    while True:
        data = sock.recv(4096).decode()
        if not data:
            raise ConnectionError("Connection closed")
        buffer += data
        if "\n" in buffer:
            msg, buffer = buffer.split("\n", 1)
            return json.loads(msg)

# Brickwork base edges (data indices). This should match the edges Bob expects for the data graph.
def build_base_data_edges():
    base_edges = []
    for i in range(2*m-2):
        base_edges.append((i, i+2))
    base_edges += [(4,5), (8,9)]
    return base_edges

all_run_data_outputs = []
caught_any = False

for run_idx in range(s_reps):
    print("\n=== Starting verification run", run_idx, "===")
    total_qubits = data_num_qubits + num_traps

    # Pick trap positions (physical positions among total_qubits)
    trap_positions = random.sample(range(total_qubits), num_traps)
    trap_positions_set = set(trap_positions)
    print("Trap positions:", trap_positions)

    # data_positions are physical positions reserved for data qubits
    data_positions = [i for i in range(total_qubits) if i not in trap_positions_set]
    assert len(data_positions) == data_num_qubits

    # Map data index -> global position
    data_index_to_pos = {data_idx: data_positions[data_idx] for data_idx in range(data_num_qubits)}

    # Build single-qubit states in global order
    single_qubit_states = [None] * total_qubits
    trap_expected = {}
    for d_idx in range(data_num_qubits):
        pos = data_index_to_pos[d_idx]
        single_qubit_states[pos] = make_plus_theta_state(theta_list[d_idx])
    for tpos in trap_positions:
        chosen = random.choice([0,1])
        single_qubit_states[tpos] = state_zero() if chosen == 0 else state_one()
        trap_expected[tpos] = chosen

    # Convert states to JSON-able list-of-[real,imag]
    data_to_send = []
    for state in single_qubit_states:
        data_to_send.append([[float(amp.real), float(amp.imag)] for amp in state])

    # Build edges mapping data-index edges -> global positions
    base_edges_data_index = build_base_data_edges()
    edges = []
    for (a, b) in base_edges_data_index:
        ga = data_index_to_pos[a]
        gb = data_index_to_pos[b]
        edges.append((ga, gb))

    # Build the interleaved measurement sequence:
    # Start with logical data-order entries, then insert traps at random positions into that sequence.
    # Each sequence entry is a tuple (kind, data_idx_or_None, global_pos)
    sequence = [("data", d_idx, data_index_to_pos[d_idx]) for d_idx in range(data_num_qubits)]
    # Insert each trap at a random insertion point
    for tpos in trap_positions:
        insert_at = random.randint(0, len(sequence))
        sequence.insert(insert_at, ("trap", None, tpos))

    # Connect to Bob and execute sequence
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        # send qubits and edges
        s.sendall((json.dumps(data_to_send) + "\n").encode())
        time.sleep(0.02)
        s.sendall((json.dumps(edges) + "\n").encode())
        time.sleep(0.02)

        cum_results_global = {}  # map physical position -> corrected measurement bit

        # Execute each measurement in the interleaved sequence
        for entry in sequence:
            kind, data_idx, gpos = entry
            if kind == "trap":
                # For a trap: ask Bob to measure in Z basis, but blind with random r
                r = random.choice([0,1])
                # delta = 0 + pi*r (pi*r flips the measurement; will be corrected by r)
                delta = wrap_2pi_floor(math.pi * r)
                # send measurement request as dict {'delta': float, 'target': int}
                s.sendall((json.dumps({'delta': delta, 'target': int(gpos)}) + "\n").encode())
                # receive result (0/1)
                raw = recv_json(s)
                measured = int(raw)
                corrected = r ^ measured
                cum_results_global[gpos] = corrected
                print(f"[TRAP measured] pos {gpos}: blinded r={r}, raw={measured}, corrected={corrected}")
            else:
                # Data qubit measured in logical order: compute phi' from dependencies referring to data indices
                s_x = 0
                s_z = 0
                for dep in D_X[data_idx]:
                    dep_global = data_index_to_pos[dep]
                    s_x ^= cum_results_global[dep_global]
                for dep in D_Z[data_idx]:
                    dep_global = data_index_to_pos[dep]
                    s_z ^= cum_results_global[dep_global]
                phi_prime = ((-1) ** s_x) * phi_list[data_idx] + s_z * np.pi
                # keep phi_prime correctly (removed debug override that set phi_prime to 0.0)
                phi_prime = wrap_2pi_floor(phi_prime)

                r = random.choice([0,1])
                delta = wrap_2pi_floor(theta_list[data_idx] + phi_prime + math.pi * r)
                s.sendall((json.dumps({'delta': delta, 'target': int(gpos)}) + "\n").encode())
                raw = recv_json(s)
                measured = int(raw)
                corrected = r ^ measured
                cum_results_global[gpos] = corrected
                print(f"[DATA measured] data_idx {data_idx} @ pos {gpos}: corrected={corrected}")

        # signal end of run to server (optional)
        s.sendall((json.dumps({'stop': True}) + "\n").encode())

        # Extract classical data outputs (in original data ordering)
        data_output = [cum_results_global[data_index_to_pos[i]] for i in range(data_num_qubits)]
        print("This run's data output:", data_output)

        # Check traps
        trap_fail = False
        for tpos, expected in trap_expected.items():
            measured = cum_results_global[tpos]
            if measured != expected:
                print(f"Trap at pos {tpos} FAILED: expected {expected} got {measured}")
                trap_fail = True
        if trap_fail:
            print("Alice detected cheating this run.")
            caught_any = True
        else:
            print("All traps OK this run.")

        all_run_data_outputs.append(data_output)

# After s_reps: final decision
print("\n=== Protocol complete ===")
if caught_any:
    print("Alice rejects: cheating detected in at least one repetition.")
else:
    all_equal = all(out == all_run_data_outputs[0] for out in all_run_data_outputs)
    if all_equal:
        print("Alice accepts: no traps failed and all outputs identical.")
        print("Accepted classical output:", all_run_data_outputs[0])
    else:
        print("Alice rejects: outputs differ across repetitions.")
        for i, out in enumerate(all_run_data_outputs):
            print(f"  Run {i}: {out}")
