# spin-phonon-coupling-in-a-small-Rydberg-Lattice
Separate Codes: Molecular Spectrum, Born-Oppenheimer Surfase, Distortion
# Spin–phonon coupling in a small Rydberg lattice — thesis scripts

This repository contains standalone Python scripts used in my MSc thesis (Advanced Quantum Physics, University of Tübingen) on vibronic (spin–phonon) coupling in a small lattice of trapped Rydberg atoms.

## Contents

### 1) Molecular Spectrum (sparse methods)
Folder: `Molecular Spectrum/`
- `Sparse_Matrix.py`  
  Builds the sparse Hamiltonian, performs diagonalization, tracks eigenvectors, and stores spectra in `.npz` files for different coupling strengths (κ).
- `Plot.py`  
  Loads the saved `.npz` results and produces spectrum plots.
- `energies_vs_omega_kappa_*.npz`  
  Saved spectra for selected κ values (example outputs).
- `spectrum_sparse_*.png`  
  Example figure(s) generated from the saved data.

### 2) Born–Oppenheimer energy surfaces
Folder: `Born-Oppenheimer Energy Surface/`
- `clear_heat_map_different_omegas.py`  
  Generates Born–Oppenheimer ground energy surface plots (heat maps) for different Ω at fixed kappa.
- `BO_surfaces_differnt_Omegas.png`  
  Example output figure.

### 3) Geometric distortion
Folder: `Distortion/`
- `Distortion_Vs_Omega_different_kappa.py`  
  Computes distortion measures versus Ω for different κ.
- `rmin_vs_Omega_for_various_kappa.png`  
  Example output figure.

## Requirements
Python 3.x  
NumPy, SciPy, Matplotlib

## How to run (examples)
```bash
python "Molecular Spectrum/Sparse_Matrix.py"
python "Molecular Spectrum/Plot.py"

python "Born-Oppenheimer Energy Surface/clear_heat_map_different_omegas.py"

python "Distortion/Distortion_Vs_Omega_different_kappa.py"
