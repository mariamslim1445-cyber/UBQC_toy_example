#!/usr/bin/env python3
# bob_server.py
import socket
import json
import numpy as np
import pennylane as qml
import struct
import sys

# -----------------------------
# Helpers: length-prefixed JSON send/recv
# -----------------------------
def send_msg(sock, obj):
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_msg(sock):
    raw_len = recv_all(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    data = recv_all(sock, msg_len)
    return json.loads(data.decode())

def recv_all(sock, n):
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            return None
        out.extend(chunk)
    return bytes(out)

# -----------------------------
# UBQC / circuit parameters
# -----------------------------
n, m = 2, 2
num_qubits = n * m
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]

# Use shots=1 so qml.sample returns one sample
dev = qml.device("default.qubit", wires=num_qubits, shots=1)

def run_circuit(state_vectors, delta_list):
    @qml.qnode(dev)
    def circuit():
        # Load Alice's prepared single-qubit states
        for i in range(num_qubits):
            sv = state_vectors[i]
            state = np.array([sv[0][0] + 1j * sv[0][1],
                              sv[1][0] + 1j * sv[1][1]], dtype=np.complex128)
            # NEW: StatePrep replaced the old QubitStateVector
            qml.StatePrep(state, wires=i)

        # Entangle according to brickwork graph
        for (a, b) in edges:
            qml.CZ(wires=[a, b])

        # Rotate to measurement bases ±δ and measure (via sampling)
        for i in range(num_qubits):
            qml.RY(-2 * float(delta_list[i]), wires=i)

        return qml.sample(wires=range(num_qubits))

    samples = circuit()
    # qml.sample may return an array; convert to a flat list of ints
    samples = np.array(samples)
    # If shape is (1, num_qubits) or (num_qubits,), flatten to 1D
    flat = samples.flatten().astype(int).tolist()
    return flat

# -----------------------------
# Socket server
# -----------------------------
HOST, PORT = "127.0.0.1", 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print("[Bob] Server listening on {}:{}".format(HOST, PORT))
    conn, addr = server.accept()
    with conn:
        print("[Bob] Connected by", addr)
        msg = recv_msg(conn)
        if msg is None:
            print("[Bob] No message received - closing")
            sys.exit(1)

        states = msg.get("states")
        delta_list = msg.get("delta")
        if states is None or delta_list is None:
            print("[Bob] Malformed request")
            sys.exit(1)

        print("[Bob] Received {} states and δ".format(len(states)))
        raw_results = run_circuit(states, delta_list)

        send_msg(conn, {"raw_results": raw_results})
        print("[Bob] Sent raw measurement results, closing connection.")
