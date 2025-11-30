import pennylane as qml
import numpy as np
import random
import json
import socket
import time
import math


# Initializations
n, m = 2, 5  # 2x5 grid
num_qubits = n * m    # number of qubits
D_X = [[], [], [0], [1], [2], [3], [4], [5], [6], [7]] #List of indices of X-dependencies
D_Z = [[], [], [], [], [0,3], [1,2], [2], [3], [4,7], [5,6]] #List of indices of Z-dependencies
phi_list = [np.pi/4, 0, np.pi/4, 0, np.pi/4, 0, 0, 0, 0, 0]
seed = 30

# Function to keep angles below 2pi
def wrap_2pi_floor(x):
    return x - math.floor(x / (2*math.pi)) * (2*math.pi)

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
    for x in range(1, n+1):         # go through columns (layers)
        for y in range(1, m+1):     # go through rows

            print(f"Handling qubit {c}...")

            # Alice prepares measurement angle delta

            print(f"randomly chonsen theta is: {theta_list[c]}")

                # compute phi_prime for determinism 
            s_x = 0 # initialize the parity measurement for qubits in X_(x,y)
            s_z = 0 # initialize the parity measurement for qubits in Z_(x,y)
            for i in D_X[c]:
                #print(f"Added qubit {i} as an X dependency")
                s_x = s_x ^ cum_results[i]   # apply the formula
            print(f"s_x of qubit {c} is: {s_x} ") 
            for i in D_Z[c]:
                #print(f"Added qubit {i} as a Z dependency")
                s_z = s_z ^ cum_results[i]   # apply the formula
            print(f"s_z of qubit {c} is: {s_z} ")
            phi_prime = ((-1)**(s_x)) * phi_list[c] + s_z * np.pi
            phi_prime = wrap_2pi_floor(phi_prime)
            print(f"corresponding phi_prime: {phi_prime}")

                # generate r
            r = random.choice([0,1])
            print(f"random r chosen: {r}")

                # compute delta
            delta = theta_list[c] + phi_prime + np.pi*r 
            delta = wrap_2pi_floor(delta)

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

            c += 1

