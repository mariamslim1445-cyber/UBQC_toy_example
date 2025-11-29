import pennylane as qml
import numpy as np
import random
import json
import socket
import time

n, m = 2, 2  # 2x2 grid
num_qubits = n * m
seed = 30

random.seed(seed)
np.random.seed(seed)


# Alice's secret theta (random from {k*pi/4})
theta_list = [random.choice([k*np.pi/4 for k in range(8)]) for _ in range(num_qubits)]

# Alice prepares and stores |+_θ> states, each qubit separately

single_qubit_states = []
for theta in theta_list:
    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev)
    def qubit_state():
        qml.Hadamard(wires=0)
        qml.RZ(theta, wires=0)
        return qml.state()
    state = qubit_state()
    single_qubit_states.append(state)

'''
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def qubit_state(theta):
    qml.Hadamard(0)
    qml.RZ(theta, 0)
    return qml.state()

single_qubit_states = [qubit_state(theta) for theta in theta_list]'''
    

# Convert states to sendable format
data_to_send = []
for state in single_qubit_states:
    state_list = [[amp.real, amp.imag] for amp in state]
    data_to_send.append(state_list)

HOST = socket.gethostbyname(socket.gethostname())
PORT = 5050

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    
    # Send qubits as JSON terminated by newline
    s.sendall((json.dumps(data_to_send) + "\n").encode())
    print("Qubits sent...")
    time.sleep(0.1)

    cum_results = [] # array to save the results
    c = 0
    for x in range(1, n+1):
        for y in range(1, m+1):

            # Alice prepares measurement angle delta
            phi_prime = 0
            r = random.choice([0,1])
            print(f"random r chosen: {r}")

            delta = theta_list[c] + phi_prime + np.pi*r / (2*np.pi)
            c += 1

            # Alice sends delta to Bob
            s.sendall((json.dumps(delta) + "\n").encode())
            print(f"delta_{(x,y)} = {delta} sent...")

            time.sleep(0.1)

            # Alice receives measurement result from Bob
            result = s.recv(4096).decode()
            result = json.loads(result)
            print(f"corresponding measurement result from server: {result}")

            # Alice corrects result based on her chosen r
            corrected_result = r ^ result
            cum_results.append(corrected_result)
            print(f"Corrected Result ({corrected_result}) saved\n")

