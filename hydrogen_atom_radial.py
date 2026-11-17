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


n = 5000
a0 = 1.0                 
r_max = 10.0 * a0
dr = r_max / (n + 1)
r = np.arange(1, n + 1) * dr


hbar = 1.0
mass = 1.0
charge = 1.0
ell = 0                 


kinetic_factor = hbar**2 / (2.0 * mass * dr**2)
main_diagonal = np.full(n, 2.0 * kinetic_factor)
off_diagonal = np.full(n - 1, -kinetic_factor)


potential = -(charge**2) / r


centrifugal = (
    hbar**2 * ell * (ell + 1) / (2.0 * mass * r**2)
)


hamiltonian_diagonal = main_diagonal + potential + centrifugal


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

    # Normalize 
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
