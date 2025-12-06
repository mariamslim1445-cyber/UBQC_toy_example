# server_Bob_fixed.py
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


def send_json(conn, obj):
    conn.sendall((json.dumps(obj) + "\n").encode())

HOST = socket.gethostbyname(socket.gethostname())
PORT = 5050

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()
print("Bob (server) ready on", HOST, "port", PORT)

while True:
    conn, addr = server.accept()
    print("Connection established from", addr)
    try:
        # Receive Alice's qubit list (list of single-qubit states)
        received_qubits_json = recv_json(conn)
        # reconstruct complex state vectors
        reconstructed_states = [np.array([complex(r, i) for r, i in state]) for state in received_qubits_json]
        num_qubits = len(reconstructed_states)
        print(f"Received {num_qubits} single-qubit states from Alice.")

        # Receive entanglement edges
        edges = recv_json(conn)
        print("Received edges:", edges)

        # Devices
        dev_meas = qml.device("default.qubit", wires=num_qubits, shots=1)

        # Helper: prepare each single-qubit state exactly matching Alice's convention
        # Alice prepares states as: H then RZ(theta) on |0> (i.e. |+_theta> = RZ(theta) H |0>).
        # If the two amplitudes have equal magnitude (within tol), we can recover theta from
        # the phase difference: theta = angle(amp1) - angle(amp0), and prepare with H then RZ(theta).
        # Otherwise, fall back to a general reconstruction via RY and RZ.
        def prepare_state_on_wires(qnode_dev):
            @qml.qnode(qnode_dev)
            def _prepare_and_measure(target, delta, edges_local):
                # Prepare every qubit from reconstructed_states
                for i, state_vector in enumerate(reconstructed_states):
                    amp0, amp1 = state_vector
                    mag0 = np.abs(amp0)
                    mag1 = np.abs(amp1)
                    # tolerance for "equal magnitudes"
                    if np.isclose(mag0, mag1, atol=1e-6):
                        # compute phase difference
                        phase = (np.angle(amp1) - np.angle(amp0))
                        # Prepare |+> then apply RZ(phase)
                        qml.Hadamard(wires=i)
                        qml.RZ(phase, wires=i)
                    else:
                        # General construction: convert state vector to Bloch angles
                        # state = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
                        theta_bloch = 2 * np.arccos(np.clip(mag0, -1.0, 1.0))
                        # relative phase between amplitudes
                        phi_rel = np.angle(amp1) - np.angle(amp0)
                        qml.RY(theta_bloch, wires=i)
                        qml.RZ(phi_rel, wires=i)
                # Apply CZ edges
                for (p, q) in edges_local:
                    qml.CZ(wires=[p, q])

                # Rotate target to Z-basis according to delta and measure
                # Alice expects Bob to perform RZ(-delta) then H then measure Z
                qml.RZ(-delta, wires=target)
                qml.Hadamard(wires=target)
                return qml.sample(wires=target)
            return _prepare_and_measure

        # Create a qnode factory for measurement where we pass target and delta each time
        measure_qnode = prepare_state_on_wires(dev_meas)

        # We'll respond to measurement deltas one-by-one
        while True:
            try:
                delta_msg = recv_json(conn)
            except ConnectionError:
                print("Connection closed by client.")
                break

            # stop flag
            if isinstance(delta_msg, dict) and 'stop' in delta_msg and delta_msg['stop']:
                break

            if not (isinstance(delta_msg, dict) and 'delta' in delta_msg and 'target' in delta_msg):
                print("Invalid measurement request:", delta_msg)
                break

            delta = float(delta_msg['delta'])
            target = int(delta_msg['target'])

            # perform measurement (this qnode will reprepare full state + edges each call)
            sample = measure_qnode(target, delta, edges)

            # normalize result retrieval for pennylane shapes
            try:
                bit = int(np.asarray(sample).flatten()[0])
            except Exception:
                bit = int(sample[0])

            send_json(conn, bit)

    except Exception as e:
        print("Server error:", e)
    finally:
        conn.close()
        print("Connection closed, ready for next client.")
