# Particle in a One-Dimensional Box

This Python program calculates the ground-state energy and wavefunction of a
particle in a one-dimensional infinite potential well (particle in a box).

## Method
The time-independent Schrodinger equation is solved numerically using a
finite-difference approximation to the second derivative. The resulting
Hamiltonian matrix is diagonalized with NumPy.

## Requirements
- Python 3
- NumPy
- Matplotlib

Install the required libraries:

```bash
python -m pip install numpy matplotlib
```

## Run

```bash
python particle_in_a_box.py
```

The program prints the numerical ground-state energy and plots the
ground-state wavefunction.

## Units and assumptions
The code uses dimensionless units with hbar = 1 and particle mass m = 1.
The potential is zero inside the box and infinite at the boundaries.
