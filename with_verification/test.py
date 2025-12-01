import pennylane as qml
from pennylane import numpy as np

dev2 = qml.device("default.qubit", wires=20, shots = 1)

delta = 0
results = []
for c in range(20):
    @qml.qnode(dev2)
    def MBQC():
    # Rotate basis so that measuring Z is equivalent to measuring ±δ
        qml.Hadamard(wires=c)
        qml.PauliZ(wires=c)
        qml.RZ(-delta, wires=c)
        qml.Hadamard(wires=c)
        return qml.sample(qml.PauliZ(c))
    results.append(MBQC()[0])

print(results)


