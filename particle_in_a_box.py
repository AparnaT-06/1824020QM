"""
Particle in a One-Dimensional Box
---------------------------------
Numerical solution of the time-independent Schrodinger equation
using the finite-difference method.

Units: hbar = 1 and particle mass m = 1.
The potential is zero inside the box and infinite at the walls.
"""

import numpy as np
import matplotlib.pyplot as plt

# Box and grid parameters
L = 1.0                 # Box length
n = 100                 # Number of interior grid points
dx = L / (n + 1)        # Grid spacing

# Construct the finite-difference Hamiltonian matrix.
# For V(x) = 0 inside the box:
# H = -(hbar^2 / 2m) d^2/dx^2, with hbar = m = 1.
main_diagonal = np.full(n, 1.0 / dx**2)
off_diagonal = np.full(n - 1, -0.5 / dx**2)

H = np.diag(main_diagonal)
H += np.diag(off_diagonal, k=1)
H += np.diag(off_diagonal, k=-1)

# Find eigenvalues and eigenvectors of the Hamiltonian
energies, wavefunctions = np.linalg.eigh(H)

# Ground-state energy and wavefunction
ground_state_energy = energies[0]
print("Ground-state energy:", ground_state_energy)

# Position grid (interior points)
x = np.linspace(dx, L - dx, n)
phi_0 = wavefunctions[:, 0]

# Normalize the wavefunction for plotting
phi_0 = phi_0 / np.sqrt(np.sum(np.abs(phi_0)**2) * dx)

# Plot the ground-state wavefunction
plt.figure(figsize=(6, 3.5))
plt.plot(x, phi_0, label="Ground State Wavefunction")
plt.title("Particle in a Box - Ground State")
plt.xlabel("Box Position (x)")
plt.ylabel("Wavefunction Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
