from math import log2
import numpy as np

from circuit import Circuit
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp
# from qutip import tensor
# from qutip.cy.spmath import zcsr_kron

# 调用了一个现有的函数
def permute(mat, orders: list):
    # permutation_matrix = np.zeros(mat.shape)
    # for source, target in enumerate(orders):
    #     permutation_matrix[source][target] = 1
    # orders = list(orders)
    # orders.reverse()
    mat = MatrixOp(mat)
    mat = mat.permute(orders).to_matrix()
    return mat

def tensor(m1, m2):
    return MatrixOp(m2).tensor(MatrixOp(m1)).to_matrix()

def simulate(circuit: Circuit, init_state: np.array) -> np.array:
    qubit_numer =  int(log2(init_state.shape[0]))

    assert qubit_numer == circuit.qubit_number

    now_state = np.array(init_state)
    for gate in circuit.gates:
        matrix = gate.matrix()  # 得到门的矩阵

        other_qubits = [qubit for qubit in range(qubit_numer) if qubit not in gate.qubits]  # 得到除了作用的门的比特

        # matrix = tensor(matrix, np.identity(2**len(other_qubits)))
        matrix =  np.kron(matrix, np.identity(2**len(other_qubits)))  # [source, target, 0, 1, 3 ····]

        now_order = gate.qubits + other_qubits
        # reconnect_order = list(range(qubit_numer))
        # for index, target in enumerate(now_order):
        #     reconnect_order[target] = index

        # reconnect_order = [0,2,1]  # 2: 201, 021
        matrix = permute(matrix, now_order)  # 得到作用在整个系统的矩阵
        # matrix = permute(matrix, other_qubits + gate.qubits )  # 得到作用在整个系统的矩阵

        now_state = matrix @ now_state  # B = UA

    return now_state
