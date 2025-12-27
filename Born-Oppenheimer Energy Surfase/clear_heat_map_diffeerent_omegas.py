# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 12:12:33 2025

@author: mwael


"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # registers 3-D projection
import matplotlib.cm as cm

# ───────── physical parameters ─────────
omega  = 1.0
x_zpf  = 1.0
kappa  = 0.5
Omega_values = [0.0001, 0.3, 0.5]


sq2 = np.sqrt(2)/2
# ───────── grid ─────────
q_min, q_max, N = -3.0, 3.0, 90       # raise N for smoother blobs
q3 = np.linspace(q_min, q_max, N)
q4 = np.linspace(q_min, q_max, N)
q7 = np.linspace(q_min, q_max, N)
Q3, Q4, Q7 = np.meshgrid(q3, q4, q7, indexing='ij')

def E_bo(q3, q4, q7, Omega):
    harm = (omega/4)*(q3**2+q4**2+q7**2)/x_zpf**2
    e5   = (kappa/x_zpf)*(  q4 - sq2*q7)
    e6   = (kappa/x_zpf)*(  q3 + sq2*q7)
    e7   = (kappa/x_zpf)*(- q4 - sq2*q7)
    e8   = (kappa/x_zpf)*(- q3 + sq2*q7)
    
    # Build the 8x8 Hamiltonian matrix stack
    H = np.zeros(q3.shape + (8, 8))
    H[..., 4, 4] = e5
    H[..., 5, 5] = e6
    H[..., 6, 6] = e7
    H[..., 7, 7] = e8
    # Off-diagonals
    H[..., 1, 4] = Omega
    H[..., 4, 1] = Omega
    H[..., 3, 7] = Omega
    H[..., 7, 3] = Omega
    H[..., 0, 4] = Omega
    H[..., 4, 0] = Omega
    H[..., 2, 5] = Omega
    H[..., 5, 2] = Omega
    H[..., 1, 5] = Omega
    H[..., 5, 1] = Omega
    H[..., 3, 6] = Omega
    H[..., 6, 3] = Omega
    H[..., 0, 7] = Omega
    H[..., 7, 0] = Omega
    H[..., 2, 6] = Omega
    H[..., 6, 2] = Omega
    
    #  eigenvalues 
    eigvals = np.linalg.eigvalsh(H)
    min_eig = np.min(eigvals, axis=-1)
    
    return harm + min_eig

# ───────── visual parameters ─────────
E_cut_above = 0.1        # keep E < Emin + this
gamma       = 2        # opacity fall-off exponent
stride      = 1          # grid sub-sampling (1 = every point)


# ───────── figure & axes set-up ─────────
plt.rcParams.update({               # ← make sure style matches previous figure
    'font.family'     : 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm'     : 'Times New Roman',
    'mathtext.it'     : 'Times New Roman:italic',
    'mathtext.bf'     : 'Times New Roman:bold',                
    'font.size'       : 18,
    'axes.labelsize'  : 16,
    'axes.titlesize'  : 18,
    'xtick.labelsize' : 16,
    'ytick.labelsize' : 16,
    'legend.fontsize' : 16,
    'axes.linewidth'  : 1.5,
    'xtick.direction' : 'in',
    'ytick.direction' : 'in',
    'xtick.top'       : False,
    'ytick.right'     : False,
   
})

# ───────── plotting ─────────
fig = plt.figure(figsize=(22,6))
axes = []

for col, Omega in enumerate(Omega_values, 1):
    E = E_bo(Q3, Q4, Q7, Omega)
    Emin = E.min()

    # mask low-energy region only
    keep = E < Emin + E_cut_above
    if stride > 1:
        keep &= (np.indices(E.shape)[0] % stride == 0)

    q3_pts = Q3[keep];  q4_pts = Q4[keep];  q7_pts = Q7[keep]
    E_pts  = E[keep]

    # normalise for colour mapping
    E_norm = (E_pts - Emin) / E_cut_above         # 0 … 1
    colors = cm.viridis(1.0 - E_norm)             # deep = dark
    colors[:, 3] = (1.0 - E_norm)**gamma          # alpha channel
    
    ax = fig.add_subplot(1,3,col,projection='3d')
    ax.scatter(q3_pts, q4_pts, q7_pts,
               c=colors, s=6, edgecolors='none')

    ax.set_xlim(q_min,q_max); ax.set_ylim(q_min,q_max); ax.set_zlim(q_min,q_max)
    ax.set_box_aspect((1,1,1))
    ax.set_xlabel(r"$q_3/x_{\rm zpf}$")
    ax.set_ylabel(r"$q_4/x_{\rm zpf}$")
    ax.set_zlabel(r"$q_7/x_{\rm zpf}$")
    ax.set_title(rf"$\Omega = {Omega}$ ($\Omega/\kappa = {Omega/kappa:.4f}$)")
    axes.append(ax)

# ───────── shared colour bar ─────────
# ScalarMappable so matplotlib knows the mapping

import matplotlib as mpl
from matplotlib.cm import ScalarMappable

norm = mpl.colors.Normalize(vmin=0, vmax=E_cut_above)
sm = ScalarMappable(norm=norm, cmap=cm.viridis_r)  # matches 1 - E_norm
sm.set_array([])

cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                    fraction=0.06, pad=0.4, aspect=35)
cbar.set_label(r"$E - E_{\min}$")


fig.tight_layout(rect=[0, 0.2, 1, 0.95])


plt.savefig("BO_surfaces_differnt_Omegas.png", dpi=300, bbox_inches='tight')
plt.savefig('BO_surfaces_differnt_Omegas.pdf', bbox_inches='tight', format='pdf')
plt.savefig('BO_surfaces_differnt_Omegas.svg', bbox_inches='tight', format='svg')
plt.show()