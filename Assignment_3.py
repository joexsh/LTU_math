#license : MIT license. 
# author : Sheng Leslie Joevenller, 2025
# This code is to show Lecture linear algebra assignment 3.



# Question 3 (e) Eiginvalues of permutation matrices.
import itertools
import numpy as np

# Generate all 6 permutation matrices
perms = list(itertools.permutations([0, 1, 2]))

# Collect unique eigenvalues
eigenvalues = set()
for p in perms:
    # Create permutation matrix
    P = np.zeros((3, 3))
    for row in range(3):
        P[row, p[row]] = 1
    # Compute eigenvalues
    eigvals = np.linalg.eigvals(P)
    for val in eigvals:
        eigenvalues.add(round(val.real, 6) + round(val.imag, 6)*1j)

print("Unique eigenvalues:", eigenvalues)

#### Question 4 (a)

import numpy as np

A = np.array([[4, 1], [2, 3]])
c = 2
A_minus_cI = A - c * np.eye(2)

eigvals_A, eigvecs_A = np.linalg.eig(A)

eigvals_A_minus_cI, eigvecs_A_minus_cI = np.linalg.eig(A_minus_cI)

print("Eigenvalues of A:", eigvals_A)
print("Eigenvalues of A - cI:", eigvals_A_minus_cI)
print("Eigenvectors of A:\n", eigvecs_A)
print("Eigenvectors of A - cI:\n", eigvecs_A_minus_cI)

## Question 4 (b),
import numpy as np

A = np.array([[2, 1], [1, 2]])
A_squared = A @ A  # A²

# Eigenvalues and eigenvectors of A
eigvals_A, eigvecs_A = np.linalg.eig(A)

# Eigenvalues and eigenvectors of A²
eigvals_A_squared, eigvecs_A_squared = np.linalg.eig(A_squared)

print("Eigenvalues of A:", eigvals_A)
print("Eigenvalues of A²:", eigvals_A_squared)
print("Eigenvectors of A:\n", eigvecs_A)
print("Eigenvectors of A²:\n", eigvecs_A_squared)

# if runs, this is the output:
# Eigenvalues of A: [3. 1.]
# Eigenvalues of A²: [9. 1.]
# Eigenvectors of A:
#  [[ 0.70710678 -0.70710678]
#  [ 0.70710678  0.70710678]]
# Eigenvectors of A²:
#  [[ 0.70710678 -0.70710678]
#  [ 0.70710678  0.70710678]]

## Question 4 (c)
import numpy as np

A = np.array([[3, -1], [2, 0]])
A_poly = np.linalg.matrix_power(A, 3) + 2 * np.linalg.matrix_power(A, 2) + 3 * A

# Eigenvalues and eigenvectors of A
eigvals_A, eigvecs_A = np.linalg.eig(A)

# Eigenvalues and eigenvectors of A^3 + 2A^2 + 3A
eigvals_poly, eigvecs_poly = np.linalg.eig(A_poly)

print("Eigenvalues of A:", eigvals_A)
print("Eigenvalues of A³ + 2A² + 3A:", eigvals_poly)
print("Eigenvectors of A:\n", eigvecs_A)
print("Eigenvectors of A³ + 2A² + 3A:\n", eigvecs_poly)

# if runs, this is the output:
# Eigenvalues of A: [1. 2.]
# Eigenvalues of A³ + 2A² + 3A: [ 6. 22.]
# Eigenvectors of A:
#  [[ 0.70710678 -0.4472136 ]
#  [ 0.70710678  0.89442719]]
# Eigenvectors of A³ + 2A² + 3A:
#  [[ 0.70710678 -0.4472136 ]
#  [ 0.70710678  0.89442719]]

## question 4 (d)
import numpy as np

A = np.array([[2, 1], [1, 2]])
A_inv = np.linalg.inv(A)

# Eigenvalues and eigenvectors of A
eigvals_A, eigvecs_A = np.linalg.eig(A)

# Eigenvalues and eigenvectors of A⁻¹
eigvals_A_inv, eigvecs_A_inv = np.linalg.eig(A_inv)

print("Eigenvalues of A:", eigvals_A)
print("Eigenvalues of A⁻¹:", eigvals_A_inv)
print("Eigenvectors of A:\n", eigvecs_A)
print("Eigenvectors of A⁻¹:\n", eigvecs_A_inv)

# if runs, this is the output:
# Eigenvalues of A: [3. 1.]
# Eigenvalues of A⁻¹: [0.33333333 1.        ]
# Eigenvectors of A:
#  [[ 0.70710678 -0.70710678]
#  [ 0.70710678  0.70710678]]
# Eigenvectors of A⁻¹:
#  [[ 0.70710678 -0.70710678]
#  [ 0.70710678  0.70710678]]

## qusetion 4 (e)
import numpy as np

A = np.array([[1, 4], [2, 3]])
A_T = A.T  # Transpose of A

# Eigenvalues and eigenvectors of A
eigvals_A, eigvecs_A = np.linalg.eig(A)

# Eigenvalues and eigenvectors of A^T
eigvals_A_T, eigvecs_A_T = np.linalg.eig(A_T)

print("Eigenvalues of A:", eigvals_A)
print("Eigenvalues of Aᵀ:", eigvals_A_T)
print("Eigenvectors of A:\n", eigvecs_A)
print("Eigenvectors of Aᵀ:\n", eigvecs_A_T)

# if runs, this is the output:
# Eigenvalues of A: [ 5. -1.]
# Eigenvalues of Aᵀ: [ 5. -1.]
# Eigenvectors of A:
#  [[ 0.70710678 -0.89442719]
#  [ 0.70710678  0.4472136 ]]
# Eigenvectors of Aᵀ:
#  [[ 0.89442719 -0.4472136 ]
#  [ 0.4472136   0.89442719]]

## question 4 (f)
import numpy as np

A = np.array([[2, 1], [0, 3]])
M = np.block([[A, np.zeros((2, 2))], [np.zeros((2, 2)), 2 * A]])

# Eigenvalues and eigenvectors of M
eigvals_M, eigvecs_M = np.linalg.eig(M)

print("Eigenvalues of M:", np.sort(eigvals_M))
print("Eigenvectors of M (columns):\n", eigvecs_M)

# if runs, this is the output:
# Eigenvalues of M: [2. 3. 4. 6.]
# Eigenvectors of M (columns):
#  [[ 1.    -0.707  0.     0.   ]
#  [ 0.     0.707  0.     0.   ]
#  [ 0.     0.     1.    -0.707]
#  [ 0.     0.     0.     0.707]]

## question 8 (c)
import numpy as np

# Define the permutation matrix
P = np.array([[0, 1],
              [1, 0]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(P)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors (as columns):")
print(eigenvectors)

##if runs, this is the output:
Eigenvalues: [ 1. -1.]
Eigenvectors (as columns):
[[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]

 ## question 8 (d)
import numpy as np

def cyclic_permutation_matrix(n):
    """Generate the n x n cyclic permutation matrix."""
    P = np.eye(n, k=1)  # Superdiagonal ones
    P[-1, 0] = 1        # Wrap around to complete the cycle
    return P

n = 4  # Example for n=4
P = cyclic_permutation_matrix(n)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(P)

print("Eigenvalues:")
for val in eigenvalues:
    print(f"{val:.3f}")

print("\nEigenvectors (as columns):")
print(np.round(eigenvectors, 3))

# if runs, this is the output:
Eigenvalues:
(1.000+0.000j)
(-0.000+1.000j)
(-1.000+0.000j)
(-0.000-1.000j)

Eigenvectors (as columns):
[[ 0.5+0.j   0.5-0.j   0.5+0.j   0.5+0.j ]
 [ 0.5+0.j   0.0-0.5j -0.5+0.j  -0.0+0.5j]
 [ 0.5+0.j  -0.5-0.j   0.5+0.j  -0.5-0.j ]
 [ 0.5+0.j  -0.0+0.5j -0.5+0.j   0.0-0.5j]]

 ## question 8 (e)
import numpy as np

# Define the permutation matrix
P = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0]
])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(P)

# Print eigenvalues and eigenvectors
print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors (columns as eigenvectors):")
print(eigenvectors)