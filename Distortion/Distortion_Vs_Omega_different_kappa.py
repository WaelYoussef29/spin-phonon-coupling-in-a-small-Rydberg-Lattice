# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 14:57:12 2025

@author: mwael
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ───────── physical params ─────────
omega  = 1.0
x_zpf  = 1.0
sq2    = np.sqrt(2)/2

def energy(q, Omega):
    q3, q4, q7 = q
    harm = (omega / 4.0) * (q3**2 + q4**2 + q7**2) / x_zpf**2

    e5 = (kappa / x_zpf) * (q4 - sq2 * q7)
    e6 = (kappa / x_zpf) * (q3 + sq2 * q7)
    e7 = (kappa / x_zpf) * (-q4 - sq2 * q7)
    e8 = (kappa / x_zpf) * (-q3 + sq2 * q7)

    H = np.zeros((8, 8))
    H[4,4], H[5,5], H[6,6], H[7,7] = e5, e6, e7, e8
    for (i, j) in [(1,4),(3,7),(0,4),(2,5),
                   (1,5),(3,6),(0,7),(2,6)]:
        H[i,j] = H[j,i] = Omega

    eigvals = np.linalg.eigvalsh(H)
    return harm + eigvals[0]


# ───────── global plotting style ─────────
plt.rcParams.update({
    'font.family'     : 'Times New Roman',
    'font.size'       : 20,
    'axes.labelsize'  : 22,
    'axes.titlesize'  : 24,
    'xtick.labelsize' : 16,
    'ytick.labelsize' : 16,
    'legend.fontsize' : 12,
    'axes.linewidth'  : 2.5,
    'xtick.direction' : 'in',
    'ytick.direction' : 'in',
    'xtick.top'       : False,
    'ytick.right'     : False,
})

# ───────── sweep parameters ─────────
Omegas = np.linspace(0.0, 1.5, 200)
kappa_list = [0.0, 0.3, 0.5, 0.8]

# ─────────  2×2 figure ─────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.ravel()

for ax, kapp in zip(axes, kappa_list):
    kappa = kapp

    # minimize to find r_min(Ω)
    q0 = np.zeros(3)
    r_mins = []
    for O in Omegas:
        res = minimize(energy, q0, args=(O,), method='Nelder-Mead')
        q0 = res.x
        r_mins.append(np.linalg.norm(q0))

    # plot
    ax.plot(Omegas/omega, np.array(r_mins)/x_zpf,
            color='blue', lw=2.5)

    ax.set_title(rf"$\kappa = {kapp}$")
    ax.set_xlabel(r'$\Omega/\omega$')
    ax.set_ylabel(r'$r_{\min}/x_{\rm zpf}$')

    # ticks & spines
    ax.tick_params(
        axis='both', which='major',
        length=8, width=2, pad=6,
        direction='in', top=False, right=False
    )
    ax.tick_params(
        axis='both', which='minor',
        length=4, width=1, pad=6,
        direction='in'
    )
    ax.minorticks_on()
    for spine in ['top','right','left','bottom']:
        ax.spines[spine].set_linewidth(2.5)
    ax.grid(False)

fig.tight_layout()
plt.savefig("rmin_vs_Omega_for_various_kappa.pdf", dpi=300, bbox_inches='tight')
plt.savefig("rmin_vs_Omega_for_various_kappa.png", dpi=300, bbox_inches='tight')
plt.show()
