"""
One-Dimensional Harmonic Oscillator
-----------------------------------
Finds the ground-state energy and wavefunction for a 1D harmonic
oscillator using a finite-difference Hamiltonian.

The potential is V(x) = (1/2) m omega^2 x^2.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Grid and physical parameters
n = 500
L = 8.0
hbar = 1.0
mass = 1.0
omega = 1.0

# Interior grid points with zero boundary conditions at x = +/- L
dx = 2.0 * L / (n + 1)
x = -L + dx * np.arange(1, n + 1)

# Finite-difference kinetic-energy matrix
kinetic_factor = hbar**2 / (2.0 * mass * dx**2)
main_diagonal = np.full(n, 2.0 * kinetic_factor)
off_diagonal = np.full(n - 1, -kinetic_factor)

# Harmonic oscillator potential
potential = 0.5 * mass * omega**2 * x**2

# Hamiltonian and lowest eigenstate
hamiltonian_diagonal = main_diagonal + potential
energies, wavefunctions = eigh_tridiagonal(
    hamiltonian_diagonal,
    off_diagonal,
    select="i",
    select_range=(0, 0),
)

ground_state = wavefunctions[:, 0].copy()
ground_state /= np.sqrt(np.sum(np.abs(ground_state)**2) * dx)

if ground_state[np.argmax(np.abs(ground_state))] < 0:
    ground_state *= -1

plt.figure(figsize=(7, 4))
plt.plot(x, ground_state, label="Ground State Wavefunction")
plt.title("Energy Level for 1D Harmonic Oscillator")
plt.xlabel("Position x")
plt.ylabel("Wavefunction Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Numerical ground-state energy:", energies[0])
print("Exact ground-state energy:", 0.5 * hbar * omega)
