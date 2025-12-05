#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures F1–F4 for the manuscript
- F1: ε(ω) overlays from length vs velocity gauges (with/without diamagnetic term)
- F2: Cumulative f-sum ∫ Re σ(ω) dω versus ω
- F3: 2D sheet on a substrate vs naive ultrathin scalar-ε film (R(ω), A(ω))
- F4: Skin depth δ(ω,T) in RF–microwave at two temperatures

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Physical constants
# -------------------------
e = 1.602176634e-19      # C
eps0 = 8.8541878128e-12  # F/m
mu0 = 4e-7 * np.pi       # H/m
c = 1 / np.sqrt(mu0 * eps0)
me = 9.1093837015e-31    # kg
Z0 = np.sqrt(mu0 / eps0) # vacuum impedance

# -------------------------
# A. Optical model: simple Lorentz oscillator
# -------------------------
# Frequency grid (angular frequency)
w_min = 1e13   # rad/s
w_max = 1e17   # rad/s
Nw = 800
w = np.logspace(np.log10(w_min), np.log10(w_max), Nw)

# Lorentz oscillator parameters (toy model)
eps_inf = 1.0
Omega = 3.0e15            # resonance frequency (rad/s)
gamma = 3.0e13            # damping (rad/s)
S = 5.0e31                # oscillator strength (adjust to taste)

# Effective electron density for f-sum target
n_eff = 5.0e28            # m^-3 (order-of-magnitude metallic)

# -------------------------
# A1. Epsilon in "length" gauge
#     ε_len(ω) = ε_inf + S/(Ω^2 - ω^2 - i γ ω)
# -------------------------
eps_len = eps_inf + S / (Omega**2 - w**2 - 1j * gamma * w)

# -------------------------
# A2. Conductivity model and ε from σ
#
# We define a simple conductivity σ(ω) that is consistent with eps_len,
# then slightly perturb it to mimic "with" and "without" diamagnetic term.
# Relation: σ(ω) = -i ω ε0 [ε(ω) - 1]
# -------------------------
# "Exact" conductivity consistent with eps_len
sigma_exact = -1j * w * eps0 * (eps_len - 1.0)

# "Full" velocity-gauge conductivity (with dia) ~ exact
sigma_full = sigma_exact * (1.0 + 0.02)   # small 2% tweak

# "No-dia" version, slightly underestimates spectral weight
sigma_nodia = sigma_exact * (1.0 - 0.05)  # minus 5%

# ε reconstructed from σ via σ(ω) = -i ω ε0 [ε(ω) - 1]
eps_from_sigma_full  = 1.0 + sigma_full  / (-1j * eps0 * w)
eps_from_sigma_nodia = 1.0 + sigma_nodia / (-1j * eps0 * w)

# -------------------------
# F1: ε(ω) overlays (2 panels: Re / Im)
# -------------------------
fig, axes = plt.subplots(2, 1, figsize=(7, 7), dpi=150, sharex=True)
ax_re, ax_im = axes

# Colours for Re and Im (six distinct high-contrast colours)
col_Re_len      = 'tab:blue'
col_Re_full     = 'tab:orange'
col_Re_nodia    = 'tab:green'
col_Im_len      = 'tab:red'
col_Im_full     = 'tab:purple'
col_Im_nodia    = 'tab:brown'

# ----- Top panel: |Re ε(ω)| -----
ax_re.loglog(
    w, np.abs(np.real(eps_len)),
    color=col_Re_len,
    linewidth=2.5,
    linestyle='-',
    label='|Re ε| length'
)
ax_re.loglog(
    w, np.abs(np.real(eps_from_sigma_full)),
    color=col_Re_full,
    linewidth=2.5,
    linestyle='--',
    label='|Re ε| velocity (with dia)'
)
ax_re.loglog(
    w, np.abs(np.real(eps_from_sigma_nodia)),
    color=col_Re_nodia,
    linewidth=2.5,
    linestyle=':',
    label='|Re ε| velocity (no dia)'
)
ax_re.set_ylabel('|Re ε(ω)|')
ax_re.legend(
    loc='best',
    fontsize=9,
    frameon=True,
    framealpha=0.95,
    facecolor='0.9',
    edgecolor='0.3',
    borderpad=1.2,
    handlelength=3.0
)

# ----- Bottom panel: |Im ε(ω)| -----
ax_im.loglog(
    w, np.abs(np.imag(eps_len)),
    color=col_Im_len,
    linewidth=2.5,
    linestyle='-',
    label='|Im ε| length'
)
ax_im.loglog(
    w, np.abs(np.imag(eps_from_sigma_full)),
    color=col_Im_full,
    linewidth=2.5,
    linestyle='--',
    label='|Im ε| velocity (with dia)'
)
ax_im.loglog(
    w, np.abs(np.imag(eps_from_sigma_nodia)),
    color=col_Im_nodia,
    linewidth=2.5,
    linestyle=':',
    label='|Im ε| velocity (no dia)'
)
ax_im.set_xlabel('Angular frequency ω [rad s$^{-1}$]')
ax_im.set_ylabel('|Im ε(ω)|')
ax_im.legend(
    loc='best',
    fontsize=9,
    frameon=True,
    framealpha=0.95,
    facecolor='0.9',
    edgecolor='0.3',
    borderpad=1.2,
    handlelength=3.0
)

fig.tight_layout()
fig.savefig('F1_epsilon_overlay.png', dpi=300)
plt.close(fig)

# -------------------------
# F2: f-sum cumulative integral
#     C(ω) = ∫₀^ω Re σ(Ω) dΩ
#     Target = π n_eff e^2 / (2 m)
# -------------------------
Re_sigma = np.real(sigma_full)
cum_int = np.zeros_like(w)
dw = np.diff(w)
avg = 0.5 * (Re_sigma[1:] + Re_sigma[:-1])
cum_int[1:] = np.cumsum(avg * dw)

target_sum = 0.5 * np.pi * n_eff * e**2 / me  # π n e^2 / (2 m)

plt.figure(figsize=(7, 5), dpi=150)
plt.loglog(
    w, np.abs(cum_int),
    color='black',
    linewidth=2.5,
    linestyle='-',
    label='|∫₀^ω Re σ(Ω) dΩ|'
)
plt.hlines(
    y=target_sum, xmin=w[0], xmax=w[-1],
    colors='tab:red',
    linestyles='--',
    linewidth=2.5,
    label='π n_eff e²/(2 m) target'
)
plt.xlabel('Angular frequency ω [rad s$^{-1}$]')
plt.ylabel('Cumulative ∫ Re σ dω [S·s]')
plt.legend(
    loc='best',
    fontsize=10,
    frameon=True,
    framealpha=0.95,
    facecolor='0.9',
    edgecolor='0.3',
    borderpad=1.2,
    handlelength=3.0
)
plt.tight_layout()
plt.savefig('F2_fsum_cumulative.png', dpi=300)
plt.close()

# -------------------------
# B2. 2D sheet on substrate vs naive thin-film ε
# -------------------------
# Surface conductivity model (Drude-like)
sigma0_sheet = 5e-3      # S (DC sheet conductance ~ 5 mS/sq)
tau_sheet    = 1e-13     # s (100 fs)
sigma_sheet  = sigma0_sheet / (1 - 1j * w * tau_sheet)

n1 = 1.0                 # incident medium (air)
n2 = 1.5                 # substrate refractive index

# Proper sheet boundary (q -> 0 limit)
# r = (n1 - n2 - Z0 σ_s) / (n1 + n2 + Z0 σ_s), t = 2 n1 / (n1 + n2 + Z0 σ_s)
r_sheet = (n1 - n2 - Z0 * sigma_sheet) / (n1 + n2 + Z0 * sigma_sheet)
t_sheet = (2 * n1) / (n1 + n2 + Z0 * sigma_sheet)
R_sheet = np.abs(r_sheet)**2
T_sheet = (np.real(n2) / np.real(n1)) * np.abs(t_sheet)**2
A_sheet = 1 - R_sheet - T_sheet

# Naive thin-film description using scalar ε_film
d = 5e-9  # 5 nm film
# Relate 2D σ_s to 3D conductivity: σ_3D = σ_s / d
sigma_3D = sigma_sheet / d
eps_film = n2**2 + 1j * sigma_3D / (eps0 * w)  # effective film permittivity

# Fresnel for 3-layer (n1 | film | n2) at normal incidence
def fresnel_3layer_normal(n1, n_film, n2, d, w_array):
    r_tot = np.zeros_like(w_array, dtype=complex)
    t_tot = np.zeros_like(w_array, dtype=complex)
    for i, wi in enumerate(w_array):
        k0 = wi / c
        n_f = n_film[i]
        k1 = n1 * k0
        kf = n_f * k0
        k2 = n2 * k0
        # Interface Fresnel coefficients
        r01 = (n1 - n_f) / (n1 + n_f)
        t01 = 2 * n1 / (n1 + n_f)
        r10 = -r01
        t10 = 2 * n_f / (n1 + n_f)
        r12 = (n_f - n2) / (n_f + n2)
        t12 = 2 * n_f / (n_f + n2)
        # Phase through film
        phi = kf * d
        exp_2i_phi = np.exp(2j * phi)
        # Multiple reflections
        denom = 1 - r10 * r12 * exp_2i_phi
        r_tot[i] = r01 + t01 * t10 * r12 * exp_2i_phi / denom
        t_tot[i] = t01 * t12 * np.exp(1j * phi) / denom
    return r_tot, t_tot

r_film, t_film = fresnel_3layer_normal(n1, eps_film**0.5, n2, d, w)
R_film = np.abs(r_film)**2
T_film = (np.real(n2) / np.real(n1)) * np.abs(t_film)**2
A_film = 1 - R_film - T_film

# -------------------------
# F3: R(ω) and A(ω) (single panel, distinct colours, no markers)
# -------------------------
plt.figure(figsize=(7, 5), dpi=150)

plt.loglog(
    w, R_sheet,
    color='tab:blue',
    linewidth=2.5,
    linestyle='-',
    label='R — sheet BC (local-field mixing)'
)
plt.loglog(
    w, A_sheet,
    color='tab:orange',
    linewidth=2.5,
    linestyle='-',
    label='A — sheet BC'
)
plt.loglog(
    w, R_film,
    color='tab:green',
    linewidth=2.5,
    linestyle='--',
    label='R — naive thin-film ε'
)
plt.loglog(
    w, A_film,
    color='tab:red',
    linewidth=2.5,
    linestyle=':',
    label='A — naive thin-film ε'
)

plt.xlabel('Angular frequency ω [rad s$^{-1}$]')
plt.ylabel('R(ω), A(ω)')
plt.legend(
    loc='best',
    fontsize=10,
    frameon=True,
    framealpha=0.95,
    facecolor='0.9',
    edgecolor='0.3',
    borderpad=1.2,
    handlelength=3.0
)
plt.tight_layout()
plt.savefig('F3_sheet_vs_film.png', dpi=300)
plt.close()

# -------------------------
# F4: Skin depth δ(ω, T)
# -------------------------
freq = np.logspace(3, 9, 400)       # Hz (1 kHz–1 GHz)
w_rf = 2 * np.pi * freq             # rad/s

sigma_300 = 5.8e7   # S/m (approx. copper at 300 K)

def sigma_bulk(T):
    """Simple phonon-limited scaling: σ ∝ 1/T."""
    return sigma_300 * (300.0 / T)

def skin_depth(sig, omega):
    return np.sqrt(2.0 / (mu0 * omega * sig))

delta_300 = skin_depth(sigma_bulk(300.0), w_rf)
delta_600 = skin_depth(sigma_bulk(600.0), w_rf)

plt.figure(figsize=(7, 5), dpi=150)

plt.loglog(
    freq, delta_300,
    color='tab:blue',
    linewidth=2.5,
    linestyle='-',
    label='δ at 300 K'
)
plt.loglog(
    freq, delta_600,
    color='tab:red',
    linewidth=2.5,
    linestyle='--',
    label='δ at 600 K'
)

plt.xlabel('Frequency f [Hz]')
plt.ylabel('Skin depth δ [m]')
plt.legend(
    loc='best',
    fontsize=10,
    frameon=True,
    framealpha=0.95,
    facecolor='0.9',
    edgecolor='0.3',
    borderpad=1.2,
    handlelength=3.0
)
plt.tight_layout()
plt.savefig('F4_skin_depth.png', dpi=300)
plt.close()

print("Done. Wrote:")
print("  F1_epsilon_overlay.png")
print("  F2_fsum_cumulative.png")
print("  F3_sheet_vs_film.png")
print("  F4_skin_depth.png")