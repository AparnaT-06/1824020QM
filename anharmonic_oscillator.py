"""
Anharmonic Oscillator

Potential:
    V(r) = (1/2) k r^2 + (1/3) b r^3

The parameters follow the values visible in the supplied notebook images:
n = 5000, R_max = 10 fm, m = 940 MeV/c^2, k = 100 MeV/fm^2,
hbar*c = 197.3 MeV fm, and b = 0, 10, 30.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


n = 5000
r_max = 10.0             # fm
dr = r_max / (n + 1)
r = np.arange(1, n + 1) * dr


mass = 940.0             # MeV/c^2
k = 100.0                # MeV/fm^2
hbar_c = 197.3           # MeV fm
ell = 0


kinetic_factor = hbar_c**2 / (2.0 * mass * dr**2)
main_kinetic = np.full(n, 2.0 * kinetic_factor)
off_kinetic = np.full(n - 1, -kinetic_factor)


centrifugal = (
    (hbar_c**2 * ell * (ell + 1)) / (2.0 * mass * r**2)
)


b_values = [0.0, 10.0, 30.0]
case_names = ["Harmonic Oscillator (b = 0)",
              "Moderate Anharmonicity (b = 10)",
              "High Anharmonicity (b = 30)"]

for b, case_name in zip(b_values, case_names):
    potential = 0.5 * k * r**2 + (1.0 / 3.0) * b * r**3
    total_diagonal = main_kinetic + potential + centrifugal

    energies, wavefunctions = eigh_tridiagonal(
        total_diagonal,
        off_kinetic,
        select="i",
        select_range=(0, 0),
    )

    u0 = wavefunctions[:, 0].copy()
    u0 /= np.sqrt(np.sum(np.abs(u0)**2) * dr)

    if u0[np.argmax(np.abs(u0))] < 0:
        u0 *= -1

    plt.figure(figsize=(7, 4))
    plt.plot(r, u0, label="Ground State Wave (1s)")
    plt.title(case_name)
    plt.xlabel("Radial Distance r (fm)")
    plt.ylabel("Wave Amplitude u(r)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"{case_name}: Ground-state energy = {energies[0]:.6f} MeV")
