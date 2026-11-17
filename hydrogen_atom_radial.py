"""
Hydrogen Atom: Radial Schrodinger Equation
------------------------------------------
Calculates and plots the ground (1s) and first excited (2s) radial
wavefunctions for a hydrogen atom using a finite-difference Hamiltonian.

Atomic units are used: hbar = m_e = e = 1.
The radial equation is solved for l = 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Grid parameters
n = 5000
a0 = 1.0                 # Bohr radius in atomic units
r_max = 10.0 * a0
dr = r_max / (n + 1)
r = np.arange(1, n + 1) * dr

# Physical constants in atomic units
hbar = 1.0
mass = 1.0
charge = 1.0
ell = 0                 # Orbital angular momentum quantum number

# Finite-difference kinetic-energy Hamiltonian
kinetic_factor = hbar**2 / (2.0 * mass * dr**2)
main_diagonal = np.full(n, 2.0 * kinetic_factor)
off_diagonal = np.full(n - 1, -kinetic_factor)

# Coulomb potential: V(r) = -e^2/r
potential = -(charge**2) / r

# Centrifugal term: l(l+1)hbar^2/(2mr^2)
centrifugal = (
    hbar**2 * ell * (ell + 1) / (2.0 * mass * r**2)
)

# Total Hamiltonian diagonal
hamiltonian_diagonal = main_diagonal + potential + centrifugal

# Calculate the two lowest eigenvalues and eigenvectors
energies, wavefunctions = eigh_tridiagonal(
    hamiltonian_diagonal,
    off_diagonal,
    select="i",
    select_range=(0, 1),
)

# Plot ground state (1s) and first excited radial state (2s)
state_labels = ["Ground State (1s)", "First Excited State (2s)"]

for state_index, label in enumerate(state_labels):
    u = wavefunctions[:, state_index].copy()

    # Normalize the radial wavefunction using the grid spacing
    u /= np.sqrt(np.sum(np.abs(u)**2) * dr)

    # Fix the arbitrary eigenvector sign for a consistent plot
    if u[np.argmax(np.abs(u))] < 0:
        u *= -1

    plt.figure(figsize=(7, 4))
    plt.plot(r, u, label=label)
    plt.title(f"Hydrogen Atom Radial Wavefunction: {label}")
    plt.xlabel("Radial Distance r (a.u.)")
    plt.ylabel("Radial Wavefunction u(r)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    energy_ev = energies[state_index] * 27.2114
    print(f"{label} energy: {energy_ev:.6f} eV")
