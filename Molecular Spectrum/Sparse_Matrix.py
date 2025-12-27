# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:01:48 2025

@author: mwael
"""

import numpy as np
from scipy.sparse import kron, identity, csr_matrix, dok_matrix
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import time
from scipy.optimize import linear_sum_assignment

###############################################################################

# Functions
#---------


def number_operator(dim=4):
    """Returns the number operator matrix truncated to dim levels."""
    return np.diag(np.arange(dim))

# phonon number hamiltonian term
#----------------------------------
def build_phonon_number_term(spin_dim, n_phonon, phonon_dim=4, omega=1.0):
    """Constructs the Hamiltonian hbar*omega*sum_j (a†_j a_j + b†_j b_j)."""
    H = 0
    # Identity
    I_spin = identity(spin_dim, format='csr')
    I_ph = identity(phonon_dim, format='csr')

    # number operators 
    n_op = csr_matrix(number_operator(phonon_dim))

    #    #  a_j† a_j tensor identities
    for j in range(n_phonon//2):
        ops = []   # empty list to fill with local operators and at the end we tensor them
        for k in range(n_phonon):
            if k == j:
                ops.append(n_op)
            else:
                ops.append(I_ph)

        term = I_spin
        for op in ops:
            term = kron(term, op, format='csr')
        H += term

    # 4 phonon modes (indices 4 to 7)
    for j in range(n_phonon//2, n_phonon):
        ops = []
        for k in range(n_phonon):
            if k == j:
                ops.append(n_op)
            else:
                ops.append(I_ph)
        term = I_spin
        for op in ops:
            term = kron(term, op, format='csr')
        H += term

    return omega*H






# Define the d operators 

def d_a(n, kappa):
    """Returns the 8x8 d_a matrix for n = 1..4 as per new definitions."""
    mat = np.zeros((8, 8))
    if n == 1:
        mat[7, 7] = 1    # d_1^a 
    elif n == 2:
        mat[5, 5] = 1    # d_2^a
    elif n == 3:
        mat[5, 5] = -1   # d_3^a
    elif n == 4:
        mat[7, 7] = -1   # d_4^a
    return kappa*mat

def d_b(n, kappa):
    """Returns the 8x8 d_b matrix for n = 1..4 as per new definitions."""
    mat = np.zeros((8, 8))
    if n == 1:
        mat[4, 4] = -1   # d_1^b similarly
    elif n == 2:
        mat[4, 4] = 1
    elif n == 3:
        mat[6, 6] = 1    
    elif n == 4:
        mat[6, 6] = -1   
    return kappa*mat

def build_e_ph_coupling(d_ops, mode_ops, spin_dim=8, phonon_dim=4, n_phonon=8):
  
    I_spin = identity(spin_dim, format='csr')
    I_ph = identity(phonon_dim, format='csr')
    mode_ops = csr_matrix(mode_ops)
    H = 0
    # Only for first 4 modes
    for n in range(8):
        d_sparse = csr_matrix(d_ops[n])
        term = d_sparse
        for i in range(n_phonon):
            if i == n:
                term = kron(term, mode_ops, format='csr')
            else:
                term = kron(term, I_ph, format='csr')
        H += term
    return H


def electronic_coupling(capital_omega, couplings, state_indices, spin_dim=8, phonon_dim=4, n_phonon=8):
    
    # Build spin matrix
    spin_mat = dok_matrix((spin_dim, spin_dim), dtype=np.float64)
    for a, b in couplings:
        i, j = state_indices[a], state_indices[b]
        spin_mat[i, j] += capital_omega
        spin_mat[j, i] += capital_omega  # h.c.

    spin_mat = csr_matrix(spin_mat)  

    term = spin_mat
    I_ph = identity(phonon_dim, format='csr')
    for _ in range(n_phonon):
        term = kron(term, I_ph, format='csr')
    return term

###############################################################################

# parameters
#------------


# Example usage:
spin_dim = 8
phonon_dim = 4
n_phonon = 8
omega = 1.0
kappa =  0.5
n_max = 3
dim = n_max + 1

###############################################################################
# Define a and adag matrices 
a = np.zeros((dim, dim))
adag = np.zeros((dim, dim))
for n in range(1, dim):
    a[n-1, n] = np.sqrt(n)
for n in range(dim-1):
    adag[n+1, n] = np.sqrt(n+1)
mode_ops = a + adag
###############################################################################


# Build the phonon_number term
H_phonon = build_phonon_number_term(spin_dim, n_phonon, phonon_dim, omega)



# for coupling term
#--------------------
state_indices = {str(i): i-1 for i in range(1,9)}

couplings = [
    ('2', '5'),
    ('4', '8'),
    ('1', '5'),
    ('3', '6'),
    ('2', '6'),
    ('4', '7'),
    ('1', '8'),
    ('3', '7'),
]


###############################################################################
#Diagonalization
#---------------

# Omega values
Omega_vals = np.linspace(0, 1.25, 150) #exxcept for 10th script to include 1
num_eigen = 25


# one diagonalization of the big matrix
#----------------------------------------
def diagonalize_for_omega(Omega, kappa):
    
    H_elec = electronic_coupling(
        capital_omega=Omega,
        couplings=couplings,
        state_indices=state_indices,
        spin_dim=8, phonon_dim=4, n_phonon=8
    )
    #  d_ops as a list: d_ops_a = [d_a(1), d_a(2), ....., d_b(4)]
    d_ops = [d_a(n,kappa) for n in range(1, 5)] + [d_b(n,kappa) for n in range(1, 5)]
   
# Build the spin_phonon coupling term
    H_spin_phonon = build_e_ph_coupling(d_ops, mode_ops, spin_dim=8, phonon_dim=4, n_phonon=8)
    H = H_phonon  + H_spin_phonon + H_elec
    eigvals, eigvecs  = eigsh(H, k=num_eigen, which='SA', tol=1e-6, maxiter=10000)
    return eigvals, eigvecs  

###############################################################################

eigvals_init, eigvecs_init = diagonalize_for_omega(Omega_vals[0], kappa)
prev_vecs = eigvecs_init  # Columns are initial reference states

filename = f"energies_vs_omega_kappa_{kappa:.2f}_25.npz"
if os.path.exists(filename):
    print(f"File {filename} exists. Skipping...")

energies = np.zeros((len(Omega_vals), num_eigen))  # each lambda is in a column of len 15
failed_indices = []

# Sweep in Omega with state tracking
for idx, Omega in enumerate(tqdm(Omega_vals, desc=f"κ={kappa}", unit="Ω")):
    try:
        eigvals, eigvecs = diagonalize_for_omega(Omega, kappa)
        if idx == 0:
            energies[0, :] = eigvals
            prev_vecs      = eigvecs
        else:
            overlaps = np.abs(eigvecs.conj().T @ prev_vecs)
            row_ind, col_ind = linear_sum_assignment(-overlaps)

            eigvals_tracked  = np.empty_like(eigvals)
            eigvecs_tracked = np.empty_like(eigvecs)
            for new_i, prev_j in zip(row_ind, col_ind):
                eigvals_tracked[prev_j]      = eigvals[new_i]
                eigvecs_tracked[:, prev_j]   = eigvecs[:, new_i]

            energies[idx, :] = eigvals_tracked
            prev_vecs        = eigvecs_tracked

    except Exception as e:
        print(f"Failed for Omega={Omega:.6f}, kappa={kappa}: {e}")
        energies[idx, :] = np.nan
        failed_indices.append(idx)

# Save results
np.savez(filename, Omega_vals=Omega_vals, energies=energies, kappa=kappa)
print(f"Saved: {filename}")