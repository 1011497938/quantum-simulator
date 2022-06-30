from random import random, randint, choice as rand_choice
import numpy as np
from numpy import cos, pi, sin, exp
# from qiskit.circuit.library import U3Gate, CXGate
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp
from qiskit import QuantumCircuit



class U3():
    def __init__(self, control, *parms):
        assert len(parms) == 3
        self.control = control
        self.target = None
        self.theta, self.lam, self.phi = parms
        self.qubits = [control]

    def matrix(self):
        return np.array([
            [cos(self.theta/2), -exp(1j * self.lam) * sin(self.theta / 2)],
            [exp(1j * self.phi) * sin(self.theta / 2),
             exp(1j * self.lam + 1j * self.phi) * cos(self.theta / 2)]
        ])


class CNOT():
    def __init__(self,  control, target):
        self.control = control
        self.target = target
        self.qubits = [control, target]

    def matrix(self):
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ])

def permute(mat, orders: list):
    '''
        orders中的j比特会被放到新的矩阵的order[j]的位置
    '''
    mat = MatrixOp(mat)
    mat = mat.permute(orders).to_matrix()
    return mat

def tensor(*ms):
    sm = ms[0]
    for m in ms[1:]:
        sm = np.kron(sm, m)
    return sm

class Circuit():
    def __init__(self, qubit_number):
        self.qubit_number = qubit_number
        self.gates = []
        return

    def addGate(self, gate):
        assert isinstance(gate, (U3, CNOT))
        self.gates.append(gate)

        assert gate.control >= 0 and gate.control < self.qubit_number and isinstance(
            gate.control, int)
        if gate.target is not None:
            assert gate.target >= 0 and gate.target < self.qubit_number and isinstance(
                gate.control, int)

    def random(self, gate_number):
        for _ in range(gate_number):
            gate_type = rand_choice([U3, CNOT])
            control = randint(0, self.qubit_number-1)
            if gate_type is U3:
                gate = U3(control,
                          random()*pi, random()*pi, random()*pi)
                self.addGate(gate)
            else:
                target = rand_choice([qubit for qubit in range(
                    self.qubit_number) if qubit != control])
                self.addGate(CNOT(control, target))
        return self

    def toQiskit(self):
        verified_qc = QuantumCircuit(self.qubit_number)
        for gate in self.gates:
            if isinstance(gate, U3):
                verified_qc.u3(gate.theta, gate.phi, gate.lam, gate.control)
            elif isinstance(gate, CNOT):
                verified_qc.cnot(gate.control, gate.target)
        return verified_qc


    def getGateMatrix(self, gate):
        qubit_numer = self.qubit_number
        other_qubits = [qubit for qubit in range(
            qubit_numer) if qubit not in gate.qubits]  # 得到除了作用的门的比特
        
        matrix = gate.matrix()  # 得到门的矩阵
        matrix = tensor(np.identity(2**len(other_qubits)), matrix)  # [N, ..., 0], matirx 在0
        matrix = permute(matrix, gate.qubits +
                            other_qubits)  #  通过把matrix放到比特实际要放的位置, 得到作用在整个系统的矩阵
        return matrix

    def matrix(self) -> np.array:
        '''计算电路的矩阵，需要加速的部分'''

        qubit_numer = self.qubit_number

        circuit_matrix = np.identity(2**qubit_numer)
        for gate in self.gates:
            matrix =  self.getGateMatrix(gate)
            circuit_matrix = matrix @ circuit_matrix  # U = U2*U1

        return circuit_matrix
