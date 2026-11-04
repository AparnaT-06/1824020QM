"""
Particle in a Finite Potential Well
-----------------------------------
Numerical solution of a one-dimensional finite square-well problem
using a finite-difference Hamiltonian matrix.

Units: hbar = 1 and particle mass m = 1.
"""

import numpy as np
import matplotlib.pyplot as plt

# Box and grid parameters
L = 1.0
n = 100
dx = L / (n + 1)

# Kinetic-energy constant for hbar = m = 1
C = 1.0 / (2.0 * dx**2)

# Construct the kinetic-energy Hamiltonian matrix
H = np.zeros((n, n))
for i in range(n):
    H[i, i] = 2.0 * C
    if i > 0:
        H[i, i - 1] = -C
    if i < n - 1:
        H[i, i + 1] = -C

# Define the finite potential well
a = 0.2
r = L / 2.0 + a       # Right wall of the well
l = L / 2.0 - a       # Left wall of the well
V0 = 0.1              # Potential outside the well

# Position grid
x = np.linspace(dx, L - dx, n)

# Add the potential energy to the Hamiltonian diagonal
for i in range(n):
    if x[i] < l or x[i] > r:
        H[i, i] += V0

# Find eigenvalues and eigenvectors
energies, wavefunctions = np.linalg.eigh(H)

# Ground-state energy and wavefunction
ground_state_energy = energies[0]
phi_0 = wavefunctions[:, 0]

# Normalize the ground-state wavefunction
phi_0 = phi_0 / np.sqrt(np.sum(np.abs(phi_0)**2) * dx)

print("Ground-state energy:", ground_state_energy)

# Plot the ground-state wavefunction
plt.figure(figsize=(7, 4))
plt.plot(x, phi_0, label="Ground State Wavefunction")
plt.title("Ground-State Wavefunction in a Finite Potential Well")
plt.xlabel("Position, x")
plt.ylabel("Wavefunction Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
