import pennylane as qml
import numpy as np
import random
import json
import socket
import time
import math


# Initializations
n, m = 2, 5  # 2x5 grid of actual data
num_qubits = n * m    # number of actual data qubits
D_X = [[], []] + [[i] for i in range (2*num_qubits-2)] #List of indices of X-dependencies
D_Z = [[], [], [], [], [0,3], [1,2], [2], [3], [4,7], [5,6], [6], [7], [8,11], [9,10], [10], [11], [12,15], [13,14], [14], [15]] #List of indices of Z-dependencies
phi_list = [np.pi/4, 0, np.pi/4, 0, np.pi/4, 0, 0, 0, 0, 0]
seed = 30

# Function to keep angles below 2pi
def wrap_2pi_floor(x):
    return x - math.floor(x / (2*math.pi)) * (2*math.pi)

random.seed(seed)
np.random.seed(seed)


# Parameters for verification
num_traps = num_qubits       # number of trap wires per run
s_reps = 4          # number of repetitions (security parameter s)

###########################################################################################################

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


###########################################################################################################

# Functions to prepare |0> and |1> trap states

def trap_state_zero():
    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev)
    def state_zero():
        qml.Hadamard(wires=0)
        return qml.state()
    return state_zero()

def trap_state_one():
    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev)
    def state_one():
        qml.Hadamard(wires=0)
        qml.PauliZ(0)
        return qml.state()
    return state_one()

trap_qubits = []
trap_expected = []
for i in range(num_traps):
    print(i)
    chosen = random.choice([0,1])
    trap_expected.append(chosen)
    chosen_state = trap_state_zero() if chosen==0 else trap_state_one()
    trap_qubits.append(chosen_state)

print(f"Trapped qubits are: {trap_qubits}\n")
print(f"Which actually are: {trap_expected}\n")

###########################################################################################################

#Build a combined list of qubits: data + traps

total_qubits = num_qubits + num_traps   #total number of qubits to be sent

# Decide trap positions randomly among the total_qubits
trap_positions = random.sample(range(total_qubits), num_traps)
trap_positions_set = set(trap_positions)
print(f"Chosen trap positions (this run): {trap_positions}\n")

# Fill positions: we need to place the data qubits in the remaining positions
data_positions = [i for i in range(total_qubits) if i not in trap_positions_set]
assert len(data_positions) == num_qubits    # data_positions length should equal data_num_qubits

total_states = [] 
t = 0 
d = 0
for i in range (total_qubits):
    if i in trap_positions_set:
        total_states.append(trap_qubits[t])
        t += 1
    else:
        total_states.append(single_qubit_states[d])
        d += 1

print(f"Total qubits are: {total_states}\n")


###########################################################################################################

# Convert states to sendable format
data_to_send = []
for state in total_states:
    state_list = [[amp.real, amp.imag] for amp in state]
    data_to_send.append(state_list)
print(data_to_send,"\n")

###########################################################################################################

# Define the client socket 
HOST = socket.gethostbyname(socket.gethostname())
PORT = 5050

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    
    # Send qubits as JSON terminated by newline
    s.sendall((json.dumps(data_to_send) + "\n").encode())
    print("Qubits sent...")
    time.sleep(0.1)

    cum_results = [] # array to save the results
    caught_any = False # No wrong trap result yet

    def handle_data_qubits(d, c):
        # Alice prepares measurement angle delta
        print(f"This is a data qubit...")

        print(f"randomly chonsen theta is: {theta_list[c]}")

                # compute phi_prime for determinism 
        s_x = 0 # initialize the parity measurement for qubits in X_(x,y)
        s_z = 0 # initialize the parity measurement for qubits in Z_(x,y)
        for i in D_X[d]:
            #print(f"Added qubit {i} as an X dependency")
            s_x = s_x ^ cum_results[i]   # apply the formula
        print(f"s_x of qubit {d} is: {s_x} ") 
        for i in D_Z[d]:
            #print(f"Added qubit {i} as a Z dependency")
            s_z = s_z ^ cum_results[i]   # apply the formula
        print(f"s_z of qubit {d} is: {s_z} ")
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

    
    def handle_trap_qubits(d, t):

        print(f"This is a trap qubit...")

                # compute phi_prime for determinism 
        s_x = 0 # initialize the parity measurement for qubits in X_(x,y)
        s_z = 0 # initialize the parity measurement for qubits in Z_(x,y)
        for i in D_X[d]:
            #print(f"Added qubit {i} as an X dependency")
            s_x = s_x ^ cum_results[i]   # apply the formula
        print(f"s_x of qubit {d} is: {s_x} ") 
        for i in D_Z[d]:
            #print(f"Added qubit {i} as a Z dependency")
            s_z = s_z ^ cum_results[i]   # apply the formula
        print(f"s_z of qubit {d} is: {s_z} ")
        phi_prime = ((-1)**(s_x)) * 0.0 + s_z * np.pi
        phi_prime = wrap_2pi_floor(phi_prime)
        print(f"corresponding phi_prime: {phi_prime}")
        
        #r = random.choice([0,1])
        r = 0
        print(f"random r chosen: {r}")

            # compute delta
        delta =  phi_prime + np.pi*r 
        delta = wrap_2pi_floor(delta)
        s.sendall((json.dumps(delta) + "\n").encode())
        print(f"delta = {delta} sent...")

        time.sleep(0.1)

        result = s.recv(4096).decode()
        result = json.loads(result)
        print(f"corresponding measurement result from server: {result}")

        corrected_result = r ^ result
        cum_results.append(corrected_result)
        print(f"Corrected Result ({corrected_result}) saved")

        print(f"Expected Result: {trap_expected[t]}")

        if corrected_result != trap_expected[t]:
            caught_any = True       # Found a False result!
            print(f"Found a False result! Do not trut the server!\n")
        else:
            print(f"Trap result as expected\n")



    d = 0
    t = 0
    c = 0
    for x in range(1, n+1):         # go through rows
        for y in range(1, 2*m+1):     # go through columns (layers)

            print(f"Handling qubit {d}...")

            if d in data_positions:
                handle_data_qubits(d, c)
                c+=1
            
            else:
                handle_trap_qubits(d, t)
                t+=1
            

            d += 1

