# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*- 
"""
Created on Thu Jul 31 21:22:15 2025

@author: mwael
"""

import os
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

# --- Analytic functions ------------------------------------------------------
def E_ground(kappa, Omega, omega=1.0):
    term = (kappa**2 / 4.0) * (
        -1.0/(8.0*omega)
        + 1.0/((-2.0+np.sqrt(2))*Omega - omega)
        + 1.0/((-2.0-np.sqrt(2))*Omega - omega)
        + 1.0/(-2.0*Omega - omega)
        + 1.0/(8.0*(-4.0*Omega - omega))
    )
    return -2.0*Omega + term

def E_JT(kappa, Omega, omega=1.0):
    if np.isclose(kappa, 0.0):
        return np.full_like(Omega, np.nan)
    a = 2.0*kappa**2 / omega**2
    vals = []
    for Ω in Omega:
        u  = -2.0*Ω**2/omega * mp.e**(-a) * (-a)**(-a) * mp.gammainc(a, 0, -a)
        u1 = -Ω**2 * omega / (2.0*kappa**2) * mp.e**(-2.0*(kappa/omega)**2)
        vals.append(mp.re(-2.0*kappa**2/omega + (u + 2.0*u1)))
    return np.array(vals, dtype=float)

# --- Plot styling -------------------------------------------------------------
plt.rcParams.update({
    "font.size": 18,           # base font size
    "axes.labelsize": 22,      # x,y label size
    "axes.titlesize": 24,      # subplot title size
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    # Times New Roman (or close serif fallback)
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    # make the math look Times-like
    "mathtext.fontset": "stix",       # STIX = Times-like math
    "mathtext.rm": "Times New Roman", # roman math font
})
plt.rcParams["text.usetex"] = False  # <--- add this line

# --- Ω grid -------------------------------------------------------------------
Omega_vals = np.linspace(0, 1.25, 150)
omega = 1.0

# --- Set up 1×3 figure --------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

# --- Panel (a): κ = 0,  the “fans” -------------------------------------------
fans = [
    (0.0,         "E = 0"),
    (-2.0,        "E = -2\\Omega"),
    ( 2.0,        "E = +2\\Omega"),
    ( np.sqrt(2), "E = \\sqrt{2}\\,\\Omega"),
    (-np.sqrt(2), "E = -\\sqrt{2}\\,\\Omega"),
]

shifts = [0.0, 1.0, 2.0, 3.0, 4.0]
linestyles = ["-", "-", "-", "-", "-"]

ax = axes[0]
for shift, ls in zip(shifts, linestyles):
    for coeff, lbl in fans:
        E = coeff * Omega_vals + shift * omega
        ax.plot(Omega_vals, E,
                color="black", linestyle=ls, linewidth=1.5)
ax.set_title(r"(a)                  $\kappa = 0$", loc="left")  # << panel label + kappa
ax.set_xlabel(r"$\Omega/\omega$")
ax.set_ylabel(r"$E/\omega$")
ax.set_xlim(0, Omega_vals.max())
ax.set_ylim(-4, 5)

# --- Panels (b) and (c): κ = 0.30, 0.50 ---------------------------------------
files_and_kappas = [
    ("energies_vs_omega_kappa_0.30_25.npz", 0.30, "(b)"),
    ("energies_vs_omega_kappa_0.50_25.npz", 0.50, "(c)"),
]

for ax, (fname, kappa, label) in zip(axes[1:], files_and_kappas):
    if not os.path.exists(fname):
        raise FileNotFoundError(f"{fname} not found.")
    data = np.load(fname)
    Ov, energies = data["Omega_vals"], data["energies"]

    # tracked eigen-energies
    for n in range(energies.shape[1]):
        ax.plot(Ov, energies[:, n],
                color="black", linestyle="-", linewidth=1.5)

    # analytic ground state E_{G,κ}
    ax.plot(Ov, E_ground(kappa, Ov),
            color="darkred", linewidth=1.8, linestyle="--",
            label=r"$E_{\mathrm{G},\kappa}$")

    # JT correction E_{JT,Ω}
    ax.plot(Ov, E_JT(kappa, Ov),
            color=(33/255, 95/255, 154/255), linewidth=1.8,
            linestyle="--",
            label=r"$E_{\mathrm{JT},\Omega}$")

    ax.set_title(fr"{label}                  $\kappa = {kappa:.2f}$", loc="left")  # << e.g. "(b) κ = 0.30"
    ax.set_xlabel(r"$\Omega/\omega$")
    ax.set_xlim(0, Ov.max())
    ax.set_ylim(-4, 4)
    ax.legend(loc="best", frameon=False)

plt.tight_layout()
plt.savefig("spectrum_sparse_3_phonons_adjust3_2.png", dpi=300, bbox_inches="tight")
plt.show()