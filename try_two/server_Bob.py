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


    # Create a device with num_qubits wires
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
    
    dummy = apply_multi_qubit_ops()
    
    dev2 = qml.device("default.qubit", wires=num_qubits, shots=1000)

    # Receive delta values from Alice
    c = 0
    for x in range(1, n+1):
        for y in range(1, m+1):
            delta = recv_json(conn)
            print(f"delta_{(x,y)} = {delta} received...")
            
            @qml.qnode(dev2)
            def MBQC():
                # Rotate basis so that measuring Z is equivalent to measuring ±δ
                qml.RY(-2*delta, wires=c)
                return qml.sample(wires=c)
            
            # Bob measures in the corresponding basis ...
            s = int(MBQC()[c][0])
            print(f"measurement result: {s}")

            # Bob sends result to Alice
            s_to_send = json.dumps(s)
            conn.send(s_to_send.encode())

            c+=1
   
