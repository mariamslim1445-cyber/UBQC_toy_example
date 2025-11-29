#!/usr/bin/env python3
# alice_client.py
import socket
import json
import numpy as np
import random
import struct

# -----------------------------
# Helpers: length-prefixed JSON send/recv
# -----------------------------
def send_msg(sock, obj):
    data = json.dumps(obj).encode()
    # 4-byte big-endian length prefix
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_msg(sock):
    # read 4-byte length prefix
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
# UBQC Parameters (2x2 brickwork)
# -----------------------------
n, m = 2, 2
num_qubits = n * m

seed = 42
random.seed(seed)
np.random.seed(seed)

theta_list = [random.choice([k * np.pi / 4 for k in range(8)]) for _ in range(num_qubits)]
phi_prime_list = [0.0] * num_qubits
r_list = [random.choice([0, 1]) for _ in range(num_qubits)]

# δ = φ' + θ + π r  (mod 2π)
delta_list = [float((phi_prime_list[i] + theta_list[i] + np.pi * r_list[i]) % (2 * np.pi))
              for i in range(num_qubits)]

# Prepare actual single-qubit statevectors |+_θ> = (|0> + e^{iθ}|1>)/√2
state_vectors = []
for theta in theta_list:
    state = np.array([1 / np.sqrt(2), np.exp(1j * theta) / np.sqrt(2)], dtype=np.complex128)
    # send as [[re0, im0], [re1, im1]]
    state_vectors.append([[state[0].real, state[0].imag],
                          [state[1].real, state[1].imag]])

# -----------------------------
# Socket: send states and δ to Bob, receive raw results
# -----------------------------
HOST, PORT = "127.0.0.1", 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect((HOST, PORT))
    msg = {
        "states": state_vectors,
        "delta": delta_list
    }
    send_msg(client, msg)

    resp = recv_msg(client)
    if resp is None:
        raise RuntimeError("No response from server")
    raw_results = resp.get("raw_results")
    if raw_results is None:
        raise RuntimeError("Malformed response from server")

# -----------------------------
# Alice applies r corrections
# -----------------------------
corrected = [int(raw_results[i]) ^ r_list[i] for i in range(num_qubits)]

print("\n--- UBQC Results (Alice) ---")
print("θ:", np.round(theta_list, 3))
print("r:", r_list)
print("δ:", np.round(delta_list, 3))
print("Raw from Bob:", raw_results)
print("Corrected (Alice s bits):", corrected)
