import pennylane as qml
import numpy as np
import json
import socket

def recv_json(conn):
    """Receive a full JSON message terminated by newline."""
    buffer = ""
    while True:
        data = conn.recv(4096).decode()
        if not data:
            raise ConnectionError("Connection closed")
        buffer += data
        if "\n" in buffer:
            msg, buffer = buffer.split("\n", 1)
            return json.loads(msg)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((socket.gethostbyname(socket.gethostname()), 5050))
server.listen()

print("Bob (server) ready, waiting for Alice...")

while True:
    conn, addr = server.accept()
    print("Connection established. Client's address:", addr)

    # Receive Alice's qubits (list of single-qubit states)
    received_qubits_json = recv_json(conn)
    reconstructed_states = [np.array([complex(r, i) for r, i in state]) for state in received_qubits_json]

    print("Received states from Alice:")
    for i, state in enumerate(reconstructed_states):
        print(f"  Qubit {i}: {state}")

    num_qubits = len(reconstructed_states)
    print("Number of qubits received:", num_qubits)

    # Receive edges specification for entanglement from Alice
    edges = recv_json(conn)
    print("Received edges for entanglement:", edges)

    # Create devices
    dev_prep = qml.device("default.qubit", wires=num_qubits)
    dev_meas = qml.device("default.qubit", wires=num_qubits, shots=1)

    @qml.qnode(dev_prep)
    def prepare_all():
        # initialize each qubit individually from received single-qubit state
        for i, state_vector in enumerate(reconstructed_states):
            alpha, beta = state_vector
            # Convert statevector to RY + RZ rotations (avoid direct state injection to stick with pennylane ops)
            # If alpha is complex or magnitude issues arise, this simple approach still reconstructs the state for normalized single-qubit vectors.
            # Handle edge cases: alpha might be 0 or 1
            amp0 = alpha
            amp1 = beta
            # if amplitude is (a + 0j) real, arccos may get numerical issues; clip
            theta = 2 * np.arccos(np.clip(np.abs(amp0), -1.0, 1.0))
            qml.RY(theta, wires=i)
            # add appropriate phase rotation
            phase = np.angle(amp1) - np.angle(amp0)
            qml.RZ(phase, wires=i)
        # Apply the entangling CZs according to edges provided by Alice
        for (p, q) in edges:
            qml.CZ(wires=[p, q])
        return qml.state()

    _ = prepare_all()  # we only need the device state after preparation and entanglement

    # For each qubit, receive delta and perform measurement in that basis; send back binary measurement result
    for q_index in range(num_qubits):
        delta = recv_json(conn)
        # perform measurement on wire q_index in the basis ±δ (we use the trick RY(-2*delta) then measure Z)
        @qml.qnode(dev_meas)
        def measure_one():
            # NOTE: device is freshly created; we need to re-prepare full state on it
            for i, state_vector in enumerate(reconstructed_states):
                amp0, amp1 = state_vector
                theta = 2 * np.arccos(np.clip(np.abs(amp0), -1.0, 1.0))
                qml.RY(theta, wires=i)
                phase = np.angle(amp1) - np.angle(amp0)
                qml.RZ(phase, wires=i)
            for (p, q) in edges:
                qml.CZ(wires=[p, q])
            # Rotate target wire so Z-measurement gives ±δ outcome
            qml.RZ(-delta, wires=q_index)
            qml.Hadamard(wires=q_index)
            return qml.sample(wires=q_index)

        sample = measure_one()
        # sample is an array-like with a single shot; convert to int 0/1
        s = int(sample[0][0]) if hasattr(sample[0], "__len__") else int(sample[0])
        print(f"Measured qubit {q_index} with delta={delta} -> {s}")

        # send result back to Alice as an integer encoded as JSON with newline termination
        to_send = json.dumps(s) + "\n"
        conn.send(to_send.encode())

    print("Completed run, closing connection.")
    conn.close()
