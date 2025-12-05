
# tddft-gauge-bridge

Gauge-invariant long-wavelength TDDFT without empty states: from polarizability to Kubo conductivity across heterogeneous materials.

This repository contains the **numerical validation scripts** used to generate Figs. F1–F4 and the parameter table in

> C. Tantardini, Q. Pitteloud, B. Yakobson, M. P. Andersson,  
> *Gauge-Invariant Long-Wavelength TDDFT Without Empty States: From Polarizability to Kubo Conductivity Across Heterogeneous Materials* (AIP, submitted).

The goals of this repo are:

- to make the **gauge checks** (length vs. velocity, equal-time term) fully reproducible,
- to document the **unit and prefactor conventions** (including explicit SI and $(\alpha_{\rm fs}\))$,
- and to provide minimal, transparent examples for people who want to extend or reimplement the workflow.

---

## Contents

- `F1-F4.py`  
  Main Python script. Generates:
  - `F1_epsilon_overlay.png` – length vs velocity gauge $(\varepsilon(\omega))$
  - `F2_fsum_cumulative.png` – optical $(f)$-sum saturation
  - `F3_sheet_vs_film.png` – 2D sheet vs. naïve ultrathin film
  - `F4_skin_depth.png` – RF/microwave skin depth at two temperatures
  - `Table_S1_units_prefactors.csv` – constants and parameters (Table S1)

- `IMG/`  
  Folder where the figures are written (created automatically if missing).

- `README.md`  
  This file.

If you use a different structure in your paper (e.g. `F1.png` instead of `F1_epsilon_overlay.png`), you can just rename the generated files or adapt the script paths.

---

## Requirements

A standard scientific Python stack is enough. For example:

- Python ≥ 3.9
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/) (for CSV table output; optional if you rewrite that part)

You can install these with:

```bash
pip install numpy matplotlib pandas
```

(or using your preferred environment manager).

---

## Quick start

Clone the repository and run the main script:

```bash
git clone https://github.com/Christian48596/tddft-gauge-bridge.git
cd tddft-gauge-bridge

python F1-F4.py
```

After a successful run you should see:

- `F1_epsilon_overlay.png`
- `F2_fsum_cumulative.png`
- `F3_sheet_vs_film.png`
- `F4_skin_depth.png`
- `Table_S1_units_prefactors.csv`

These files are exactly the ones referenced as F1–F4 and Table~\ref{tab:allparams} in the manuscript.

---

## What each figure shows

### F1 – Length vs. velocity gauge \(\varepsilon(\omega)\)

- Single Lorentz/2-level oscillator with:
  - transition energy \(\hbar \Omega = 3.0\) eV,
  - dipole \(\mu = 3.0\) D,
  - dephasing \(\eta = 0.05\,\Omega\) (damping \(\gamma = 2\eta\)),
  - number density \(N = 10^{27}\,\mathrm{m^{-3}}\).

- **Length gauge route**: build \(\alpha(\omega)\) from the two-level expression, then
  \[
    \varepsilon(\omega) = 1 + \frac{N\,\alpha(\omega)}{\varepsilon_0}.
  \]

- **Velocity gauge route**: construct \(\sigma(\omega)\) via the Kubo current–current response, including:
  - paramagnetic current,
  - equal-time (diamagnetic/contact) term,

  then
  \[
    \varepsilon(\omega) = 1 - \frac{i}{\varepsilon_0\omega}\,\sigma(\omega).
  \]

- The script plots:
  - length-gauge \(\varepsilon(\omega)\),
  - full velocity-gauge \(\varepsilon(\omega)\),
  - velocity-gauge without the equal-time term (“no-dia”).

Over RF–UV, the **length** and **full velocity** curves numerically coincide, while the **no-dia** curve exhibits the expected unphysical behaviour at low frequency.

---

### F2 – Optical \(f\)-sum saturation

Using the same \(\sigma(\omega)\) as F1, we compute the cumulative integral

\[
\mathcal{C}(\omega) = \int_0^\omega \mathrm{Re}\,\sigma(\Omega)\,d\Omega,
\]

and compare it against the target value

\[
\lim_{\omega\to\infty} \mathcal{C}(\omega)
= \frac{\pi n e^2}{2 m},
\]

with an effective density \(n_{\mathrm{eff}}\) written to `Table_S1_units_prefactors.csv`.

The plot shows that \(\mathcal{C}(\omega)\) saturates at the analytic value within numerical tolerance, providing a **global unit/prefactor check** of the implementation.

---

### F3 – 2D sheet vs. naïve ultrathin film

We compare two ways of modeling a conductive 2D layer on a substrate:

1. **Sheet model (solid lines)**  
   - Use a 2D Drude conductivity \(\sigma_{2\mathrm{D}}(\omega)\) with:
     - DC conductance \(\sigma_0 = 5\times10^{-3}\,\mathrm{S}\),
     - scattering time \(\tau = 10^{-13}\,\mathrm{s}\).
   - Implement proper **sheet boundary conditions** at the interface (normal incidence).
   - This corresponds to the \(q\to 0\) face of the dielectric-matrix formalism.

2. **Naïve ultrathin film (dashed/dotted)**  
   - Map the same sheet conductance to a bulk \(\sigma(\omega)\) over thickness \(d = 1\) nm.
   - Construct a scalar \(\varepsilon(\omega)\) and treat the system as a three-layer Fresnel stack (air | film | substrate).

The difference between \(R(\omega)\) and \(A(\omega)\) from these two models quantifies the role of **local-field mixing** and microscopic geometry at the interface.

---

### F4 – RF/microwave skin depth

Starting from a phonon-limited Drude conductivity for a bulk metal,

- reference conductivity: \(\sigma(300\,\mathrm{K}) = 5.8\times10^7\,\mathrm{S/m}\),
- simple temperature scaling \(\sigma(T) \propto 1/T\),

we compute the skin depth

\[
\delta(\omega, T) = \sqrt{\frac{2}{\mu_0\,\omega\,\sigma(T)}} 
\]

over a logarithmic frequency range, for two temperatures (e.g., 300 K and 600 K).

The resulting curves show:

- \(\delta \propto \omega^{-1/2}\) at fixed \(T\),
- \(\delta \propto \sigma(T)^{-1/2}\) between temperatures,

as expected from textbook RF–microwave electrodynamics.

---

## Reproducibility and units

All constants and parameters used in the script are written to:

- `Table_S1_units_prefactors.csv`

in a format that mirrors the manuscript’s Table~\ref{tab:allparams}. This includes:

- SI values of physical constants (\(\varepsilon_0\), \(\mu_0\), \(e\), \(m\), \(\hbar\), …),
- oscillator parameters and derived quantities (e.g. \(\Omega\), \(\gamma\), Lorentz strength \(S\), \(n_{\mathrm{eff}}\), \(\alpha(0)\)),
- interface and thin-film descriptors,
- skin-depth parameters,
- numerical grid definitions.

The same units and prefactors are used consistently for F1–F4.

---

## Citing

If you use this repository or adapt the scripts, please cite the associated paper:

```bibtex
@article{Tantardini2025GaugeBridge,
  author  = {Tantardini, Christian and Pitteloud, Quentin and Yakobson, Boris and Andersson, Martin Peter},
  title   = {Gauge-Invariant Long-Wavelength TDDFT Without Empty States: From Polarizability to Kubo Conductivity Across Heterogeneous Materials},
  journal = {To be updated},
  year    = {2025},
}
```

(Replace journal / year / pages once available.)

---

## License

- [MIT License](https://opensource.org/licenses/MIT)  

---

## Contact

For questions, issues, or suggestions:

- **Christian Tantardini** – `christiantantardini@ymail.com`

Feel free to open a GitHub issue or pull request if you spot bugs, want to add more validation cases, or adapt the examples to your own TDDFT/linear-response implementation.
