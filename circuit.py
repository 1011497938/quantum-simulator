from random import random, randint, choice as rand_choice
import numpy as np
from numpy import cos, pi, sin, exp
from qiskit.circuit.library import U3Gate, CXGate
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp

class Gate():
    def __init__(self, type, control, target = None):
        self.type = type

        self.control = control
        self.target = target
       
        if target is not None:
            self.qubits = [target, control]
        else:
            self.qubits = [control]
        
    def __str__(self):
        return f'Gate: {self.type}, {self.qubits}'


class U3(Gate):
    def __init__(self, control, *parms):
        super().__init__('U3',control)
        assert len(parms) == 3
        self.theta, self.lam, self.phi = parms

    def matrix(self):
        return  np.array([
            [cos(self.theta/2), -exp(1j * self.lam) * sin(self.theta / 2)],
            [exp(1j * self.phi) * sin(self.theta / 2), exp(1j * self.lam + 1j * self.phi) * cos(self.theta / 2)]
        ])


class CNOT(Gate):
    def __init__(self,  control, target):
        super().__init__('CNOT', control, target)

    def matrix(self):
        return np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0],
        ])


# 调用了一个现有的函数
def permute(mat, orders: list):
    mat = MatrixOp(mat)
    mat = mat.permute(orders).to_matrix()
    return mat

def tensor(*ms):
    ms = list(ms)
    ms.reverse()
    sm = ms[0]
    for m in ms[1:]:
        sm = np.kron(sm ,m)
    return sm

class Circuit():
    def __init__(self, qubit_number):
        self.qubit_number = qubit_number
        self.gates = []
        return


    def addGate(self, gate: Gate):
        self.gates.append(gate)
        
        assert gate.control >= 0 and gate.control < self.qubit_number and isinstance(gate.control, int)
        if gate.target is not None:
            assert gate.target >= 0 and gate.target < self.qubit_number and isinstance(gate.control, int)

    def random(self, gate_number):
        for _ in range(gate_number):
            gate_type = rand_choice([U3, CNOT])
            if gate_type is U3:
                gate = U3(randint(0, self.qubit_number-1), random()*pi, random()*pi, random()*pi)
                self.addGate(gate)
            else:
                control = randint(0, self.qubit_number-1)
                target = rand_choice([qubit for qubit in range(self.qubit_number) if qubit != control])
                self.addGate(CNOT(control, target))
        return self

    
    def toQiskit(self):
        from qiskit import QuantumCircuit
        
        verified_qc = QuantumCircuit(self.qubit_number)
        for gate in self.gates:
            if isinstance(gate, U3):
                verified_qc.u3(gate.theta, gate.phi, gate.lam, gate.control) 
            elif isinstance(gate, CNOT):
                verified_qc.cnot(gate.control, gate.target)
        # verified_qc = verified_qc.reverse_bits() # qiskt是小端, 我们这里是大端
        return verified_qc


    def matrix(self) -> np.array:
        qubit_numer = self.qubit_number

        circuit_matrix = np.identity(2**qubit_numer)
        for gate in self.gates:
            matrix = gate.matrix()  # 得到门的矩阵

            other_qubits = [qubit for qubit in range(qubit_numer) if qubit not in gate.qubits]  # 得到除了作用的门的比特

            matrix = tensor(matrix, np.identity(2**len(other_qubits)))  #[0, ..., N], matirx 在0
            matrix = permute(matrix, gate.qubits + other_qubits)  # 得到作用在整个系统的矩阵

            circuit_matrix = matrix @ circuit_matrix  # B = UA

        return circuit_matrix