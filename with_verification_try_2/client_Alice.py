import pennylane as qml
import numpy as np
import random
import json
import socket
import time
import math

# Basic parameters (original data-grid)
n, m = 2, 5  # 2x5 grid for the original data qubits
data_num_qubits = n * m

# Existing MBQC dependency lists for data qubits (unchanged)
D_X = [[], [], [0], [1], [2], [3], [4], [5], [6], [7]]  # X dependencies per data-position index
D_Z = [[], [], [], [], [0,3], [1,2], [2], [3], [4,7], [5,6]]  # Z dependencies per data-position index
phi_list = [np.pi/4, 0, np.pi/4, 0, np.pi/4, 0, 0, 0, 0, 0]
seed = 30

def wrap_2pi_floor(x):
    return x - math.floor(x / (2*math.pi)) * (2*math.pi)

random.seed(seed)
np.random.seed(seed)

# Alice's secret thetas for data qubits (kept fixed across runs in this demo)
theta_list = [random.choice([k*np.pi/4 for k in range(8)]) for _ in range(data_num_qubits)]

# Utility: prepare single-qubit state vector for |+_theta> or |0>/<1>
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

# Parameters for verification
num_traps = 3     # number of trap wires per run (you can change)
s_reps = 4        # number of repetitions (security parameter s)

HOST = socket.gethostbyname(socket.gethostname())
PORT = 5050

# Run the whole repetition protocol s_reps times
all_run_data_outputs = []  # store data outputs for each run
caught_any = False

for run_idx in range(s_reps):
    print("\n=== Starting verification run", run_idx, "===")

    # Build a combined list of qubits: data + traps.
    total_qubits = data_num_qubits + num_traps

    # Decide trap positions randomly among the total_qubits
    trap_positions = random.sample(range(total_qubits), num_traps)
    trap_positions_set = set(trap_positions)
    print("Chosen trap positions (this run):", trap_positions)

    # Fill positions: we need to place the data qubits in the remaining positions
    data_positions = [i for i in range(total_qubits) if i not in trap_positions_set]
    # data_positions length should equal data_num_qubits
    assert len(data_positions) == data_num_qubits

    # Map data index -> global position
    data_index_to_pos = {data_idx: data_positions[data_idx] for data_idx in range(data_num_qubits)}

    # Build the single-qubit state list in global order 0..total_qubits-1
    single_qubit_states = [None] * total_qubits
    # store traps info: expected Z measurement result for each trap index
    trap_expected = {}
    for data_idx in range(data_num_qubits):
        pos = data_index_to_pos[data_idx]
        single_qubit_states[pos] = make_plus_theta_state(theta_list[data_idx])

    for tpos in trap_positions:
        # choose randomly |0> or |1>
        chosen = random.choice([0, 1])
        single_qubit_states[tpos] = state_zero() if chosen == 0 else state_one()
        trap_expected[tpos] = chosen
    # convert states to sendable JSON format
    data_to_send = []
    for state in single_qubit_states:
        state_list = [[amp.real, amp.imag] for amp in state]
        data_to_send.append(state_list)

    # Build an edges list for entanglement that includes only edges between **data** qubits mapped into their global positions.
    # We convert the brickwork edges (for the data grid) into edges in the global indexing.
    edges = []
    # recreate the edges for the original data grid (same logic as your original server's edges building)
    # Here we assume the brickwork edges as in your original server: edges between certain pairs of data indices.
    # We'll map those data-index edges into the permuted global positions.
    base_edges_data_index = []
    # replicate construction used earlier in server_Bob.py: (this is the same brickwork as before)
    for i in range(2*m-2):
        base_edges_data_index.append((i, i+2))
    base_edges_data_index += [(4,5), (8,9)]
    # map them
    for (a, b) in base_edges_data_index:
        ga = data_index_to_pos[a]
        gb = data_index_to_pos[b]
        edges.append((ga, gb))

    # Connect to Bob and run MBQC for this run
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        # Send qubits
        s.sendall((json.dumps(data_to_send) + "\n").encode())
        time.sleep(0.05)

        # Send edges spec
        s.sendall((json.dumps(edges) + "\n").encode())
        time.sleep(0.05)

        # Now follow the MBQC measurement order over the grid **in the original ordering of data qubits**
        # But we must issue a measurement (delta) for every global position 0..total_qubits-1 in consistent order.
        # For data qubits we compute phi_prime using dependencies in terms of original data-index ordering,
        # but need to translate cum_results from global positions.
        cum_results_global = {}  # map global position -> classical corrected measurement (0/1)
        # We will iterate through the original data order (x,y as before) to ensure deterministic dependencies
        c = 0  # data index counter for the original grid order
        # We'll also need to make sure we send measurement deltas for trap wires at the appropriate times:
        # A simple approach: measure in increasing global index order 0..total_qubits-1
        # To keep your determinism and dependency structure close to original, we'll do this:
        for global_index in range(total_qubits):
            # Check whether this global index corresponds to a data qubit or trap
            if global_index in trap_positions_set:
                # For a trap: we ask Bob to measure in Z basis.
                # To request a Z-basis measurement in MBQC using the ±δ basis trick,
                # set delta = 0 (then rotating by -2*delta = 0 does nothing) and Bob measures Z directly.
                r = random.choice([0, 1])
                delta = 0.0 + math.pi * r  # adding pi*r is equivalent to a one-time flip and is corrected by r later
                delta = wrap_2pi_floor(delta)
                # send delta to Bob
                s.sendall((json.dumps(delta) + "\n").encode())
                # receive measurement result
                result = s.recv(4096).decode()
                result = json.loads(result)
                corrected = r ^ int(result)
                cum_results_global[global_index] = corrected
                print(f"[TRAP] pos {global_index}: expected {trap_expected[global_index]}, measured {corrected}")
            else:
                # This is a data/qubit; find its data-index c
                # find data_idx such that data_index_to_pos[data_idx] == global_index
                data_idx = None
                # (small linear search; data_num_qubits is small)
                for ddx, pos in data_index_to_pos.items():
                    if pos == global_index:
                        data_idx = ddx
                        break
                if data_idx is None:
                    raise RuntimeError("Couldn't find data index for position", global_index)
                # compute s_x and s_z from previously-corrected classical results of dependencies
                s_x = 0
                s_z = 0
                for i_dep in D_X[data_idx]:
                    # dependency i_dep refers to a data-index earlier in the original ordering
                    # get its global pos and look up cum_results
                    dep_global_pos = data_index_to_pos[i_dep]
                    s_x ^= cum_results_global[dep_global_pos]
                for i_dep in D_Z[data_idx]:
                    dep_global_pos = data_index_to_pos[i_dep]
                    s_z ^= cum_results_global[dep_global_pos]
                phi_prime = ((-1) ** s_x) * phi_list[data_idx] + s_z * np.pi
                phi_prime = wrap_2pi_floor(phi_prime)
                r = random.choice([0,1])
                delta = theta_list[data_idx] + phi_prime + np.pi * r
                delta = wrap_2pi_floor(delta)
                # send delta
                s.sendall((json.dumps(delta) + "\n").encode())
                # receive measurement result
                result = s.recv(4096).decode()
                result = json.loads(result)
                corrected = r ^ int(result)
                cum_results_global[global_index] = corrected
                print(f"[DATA] data_idx {data_idx} at pos {global_index}: corrected result {corrected}")

        # After finishing measuring all wires in this run, extract the classical outputs for data qubits
        # The "classical output" is the list of cum_results for the data qubits in original ordering
        data_output = [cum_results_global[data_index_to_pos[i]] for i in range(data_num_qubits)]
        print("This run's data output (original ordering):", data_output)

        # Check traps correctness
        trap_fail = False
        for tpos, expected in trap_expected.items():
            measured = cum_results_global[tpos]
            if measured != expected:
                print(f"Trap at position {tpos} FAILED: expected {expected}, got {measured}")
                trap_fail = True
        if trap_fail:
            print("Alice detected cheating in this run!")
            caught_any = True
            # still record the data output (could be ignored); but breaking is optional. We'll continue repetitions per specification.
        else:
            print("All traps ok in this run.")

        all_run_data_outputs.append(data_output)
        # connection will close automatically at end of with-block

# After s_reps runs: decide acceptance
print("\n=== Protocol complete ===")
if caught_any:
    print("Alice rejects: cheating was detected in at least one repetition.")
else:
    # Check all data outputs are identical across runs
    all_equal = all(out == all_run_data_outputs[0] for out in all_run_data_outputs)
    if all_equal:
        print("Alice accepts: no traps failed and all outputs identical.")
        print("Accepted classical output:", all_run_data_outputs[0])
    else:
        print("Alice rejects: outputs differ across repetitions.")
        print("Run outputs were:")
        for i, out in enumerate(all_run_data_outputs):
            print(f"  Run {i}: {out}")
