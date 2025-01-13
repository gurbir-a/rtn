import numpy as np
from scipy.stats import unitary_group
import matplotlib.pyplot as plt

# np.random.seed(seed=233423)

def generate_random_unitary_tensor(N):
    unitary_matrix = unitary_group.rvs(N**2)
    unitary_tensor = unitary_matrix.reshape((N, N, N, N))
    return unitary_tensor

def sample_ket(shape):
    unitary = unitary_group.rvs(dim=2**len(shape))
    tensor = unitary.reshape(*shape)
    
    norm = np.linalg.norm(tensor)
    normalized_tensor = tensor / norm
    
    return normalized_tensor

def is_unitary(matrix, tol=1e-10):
    matrix_dagger = np.conjugate(matrix.T)
    product = np.dot(matrix,matrix_dagger)
    identity = np.eye(matrix.shape[0])
    return np.allclose(product, identity, atol=tol)

def gram_schmidt(vectors):
    orthogonalized_vectors = []
    for v in vectors:
        w = v - sum(np.vdot(q,v) * q/np.linalg.norm(q) for q in orthogonalized_vectors)
        if np.linalg.norm(w) > 1e-10:
            orthogonalized_vectors.append(w / np.linalg.norm(w))
        else:
            print("FAIL")
    return np.array(orthogonalized_vectors)

DIM_PER_BIT=2
MAX_OPS = range(1,10)
N_BITS_PER_OP = 2

norms = []
for N_OPS in MAX_OPS:
    print("N_OPS = ", N_OPS)
    ops_list = []
    for i in range(N_OPS):
        op = generate_random_unitary_tensor(DIM_PER_BIT)
        ops_list.append(op)

    ancilla_qubit = np.array([1, 0])
    old_op = ops_list[0]
    old_op = np.tensordot(old_op, ancilla_qubit, axes=0) #add ancilla qubit via tensor product. This doesnt affect unitarity. Converts 4x4 unitary matrix to 4x8, so the 4x8 * 8x4 product still equals I

    for i in range(1, len(ops_list)): #for next_op in ops_list[1:]:
        # print(next_op.shape, old_op.shape)
        next_op = ops_list[i]
        old_op = np.tensordot(next_op, old_op, axes=(3, 0))
        old_op = np.moveaxis(old_op,2,i+2)


    # post_selected_tensor = old_op[0,...]
    post_selected_tensor = old_op
    post_selected_tensor[1, ...]=0 #set first row to zero to post-select
    # zero_state = np.array([1,0])
    # post_selected_tensor = np.tensordot(old_op, zero_state, axes=(len(old_op.shape)-1,0))
    # post_selected_tensor = np.tensordot(zero_state, old_op, axes=(0,0))
    # post_selected_tensor = post_selected_tensor[..., 0]
    # post_selected_conj = np.conjugate(post_selected_tensor.T)

    shape = post_selected_tensor.shape
    flattened_tensor = post_selected_tensor.reshape(2**(N_OPS+1), 2**(N_OPS+2))
    flattened_tensor_conj = np.conjugate(flattened_tensor.T)
    # print(is_unitary(np.matmul(flattened_tensor, flattened_tensor_conj)))

    # gs_unitary = gram_schmidt(flattened_tensor)
    # norms.append(np.linalg.norm(flattened_tensor - gs_unitary, 'fro'))
    # gs_unitary_dagger = np.conjugate(gs_unitary.T)
    norms.append(np.linalg.norm(np.matmul(flattened_tensor, flattened_tensor_conj) - np.eye(flattened_tensor.shape[0]), 'fro')/2**(N_OPS))
    # norms.append(np.linalg.norm(np.matmul(post_selected_tensor, post_selected_conj), 'fro')/2**(N_OPS+1))

plt.plot(MAX_OPS, norms)
plt.show()

print(flattened_tensor.shape)

