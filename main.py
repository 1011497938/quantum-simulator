from circuit import Circuit, CNOT, U3
import time
from numpy import pi
import numpy as np

qubit_numer = 5
qc = Circuit(qubit_numer)
# qc.addGate(CNOT(0, 1))
# qc.addGate(U3(0, 0, pi, 0))
# qc.addGate(U3(1, 0, pi, 0))
# qc.addGate(U3(2, 0, 0, pi))
qc.random(20)

# init_state = np.zeros(2**qubit_numer)
# init_state[0] = 1
# final_state = simulate(qc, init_state)
# print(final_state)

qc_matrix = qc.matrix()


from qiskit.quantum_info import Operator
verified_qc = qc.toQiskit() #.reverse_bits()
mat = Operator(verified_qc).data
# print(mat)
# print(mat @ init_state)
print(verified_qc)
print(np.allclose(mat, qc_matrix))
print(np.round(qc_matrix))
print()
print(np.round(mat))

pass