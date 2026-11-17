"""
Hydrogen Atom with a Screened Coulomb Potential
------------------------------------------------
Calculates the ground-state radial wavefunction for a hydrogen-like
system using a Yukawa (screened Coulomb) potential.

Atomic units are used: hbar = m_e = e = 1.
The screening length is set to 3 Angstrom in atomic units.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Grid parameters
n = 5000
a0 = 1.0
r_max = 10.0 * a0
dr = r_max / (n + 1)
r = np.arange(1, n + 1) * dr

# Atomic units
hbar = 1.0
mass = 1.0
charge = 1.0
ell = 0

# Screening length: 1 Angstrom = 1.88973 Bohr radii
screening_length = 3.0 * 1.88973

# Screened Coulomb (Yukawa) potential:
# V(r) = -(e^2/r) * exp(-r/a)
screened_potential = (
    -(charge**2 / r) * np.exp(-r / screening_length)
)

# Finite-difference kinetic-energy Hamiltonian
kinetic_factor = hbar**2 / (2.0 * mass * dr**2)
main_diagonal = np.full(n, 2.0 * kinetic_factor)
off_diagonal = np.full(n - 1, -kinetic_factor)

# Centrifugal term for angular momentum ell
centrifugal = (
    hbar**2 * ell * (ell + 1) / (2.0 * mass * r**2)
)

hamiltonian_diagonal = (
    main_diagonal + screened_potential + centrifugal
)

# Ground-state eigenvalue and eigenvector
energies, wavefunctions = eigh_tridiagonal(
    hamiltonian_diagonal,
    off_diagonal,
    select="i",
    select_range=(0, 0),
)

u0 = wavefunctions[:, 0].copy()
u0 /= np.sqrt(np.sum(np.abs(u0)**2) * dr)

if u0[np.argmax(np.abs(u0))] < 0:
    u0 *= -1

plt.figure(figsize=(7, 4))
plt.plot(r, u0, label="Ground State Wavefunction (1s)")
plt.title("Hydrogen Atom Ground State with Screened Potential")
plt.xlabel("Radial Distance r (a.u.)")
plt.ylabel("Wave Amplitude u(r)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Ground-state energy:", energies[0] * 27.2114, "eV")
