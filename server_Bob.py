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

while True:
    conn, addr = server.accept()
    print("Connection established. Client's address:", addr)

    # Receive Alice's qubits
    received_qubits_json = recv_json(conn)
    reconstructed_states = [np.array([complex(r, i) for r, i in state]) for state in received_qubits_json]

    print("Received states from Alice:")
    for i, state in enumerate(reconstructed_states):
        print(f"Qubit {i}: {state}")

    num_qubits = len(reconstructed_states)
    print("Number of qubits:", num_qubits)

    n, m = 2, 5  
    edges = []
    for i in range(2*m-2):
        edges.append((i,i+2))
    edges += [(4,5), (8,9)]
    print(edges)


    dev1 = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev1)
    def apply_multi_qubit_ops():
        # Initialize each qubit individually
        for i, state_vector in enumerate(reconstructed_states):
            alpha, beta = state_vector
            qml.RY(2*np.arccos(alpha), wires=i)
            qml.RZ(np.angle(beta)-np.angle(alpha), wires=i)

        # Bob applies CZ entanglement (brickwork)
        for (i,j) in edges:
            qml.CZ(wires=[i,j])
        return qml.state()

    prepared_state = apply_multi_qubit_ops()  # returns state vector

    # Step 2: measurement QNode using the prepared state
    dev2 = qml.device("default.qubit", wires=num_qubits, shots=1000)

    def MBQC(delta, wire, state_vector):
        @qml.qnode(dev2)
        def qnode():
            # Set the qubits to be the qubits from brickwork state
            qml.StatePrep(state_vector, wires=range(num_qubits))
            # Rotate then apply measurement in Z
            qml.RZ(-delta, wires=wire)
            qml.Hadamard(wires=wire)
            return qml.sample(qml.PauliZ(wire))
        return qnode()
    # Note: This way is assuming we can "copy" unknown quantum states, but pennylane does not allow us to work on same qubits with different qnodes
    # Also, this does not take into consideration the collapse of previous qubits.
    '''
    def MBQC_state(delta, wire, state_vector):
        @qml.qnode(dev2)
        def qnode():
            # Set the qubits to be the qubits from brickwork state
            qml.StatePrep(state_vector, wires=range(num_qubits))
            # Rotate then apply measurement in Z
            qml.RZ(-delta, wires=wire)
            qml.Hadamard(wires=wire)
            qml.sample(qml.PauliZ(wire))
            return  qml.state()
        return qnode()
    '''

    # Step 3: apply MBQC measurements
    c = 0
    for x in range(1, n+1):
        for y in range(1, m+1):
            delta = recv_json(conn)
            print(f"delta_{(x,y)} = {delta} received...")
            res = MBQC(delta, c, prepared_state)
            res = res[0]
            s = 0 if res == 1 else 1
            print(f"measurement result: {s}")
            #prepared_state = MBQC_state(delta, c, prepared_state)

            # Bob sends result to Alice
            s_to_send = json.dumps(s)
            conn.send(s_to_send.encode())

            c+=1
   
