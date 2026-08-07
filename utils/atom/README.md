<div align="center">
<img src="logo.png" alt="logo" width="250"></img>
</div>


# atom — Atomic DFT (Spectral Finite Elements)

![build](https://img.shields.io/badge/build-passing-brightgreen)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](../../../LICENSE)


[**Features**](#features)
| [**Quick start**](#quick-start)
| [**Package layout**](#package-layout)
| [**Upstream**](#upstream)


## What is this package?

**atom** is a Python library for atomic electronic-structure calculations in
Kohn–Sham DFT using a Spectral Finite Element (SFE) discretization. It supports
all-electron and norm-conserving pseudopotential calculations across local,
semilocal, and nonlocal exchange–correlation approximations (including hybrids
and RPA / OEP).

This directory is the SPARC in-tree copy of the solver sources from
[SPARC-atomSFE](https://github.com/SPARC-X/SPARC-atomSFE) (`src/`). It lives
under `utils/atom` so other utilities (notably **PDOS**) can import
`AtomicDFTSolver` when `utils/` is on `PYTHONPATH`.

```python
from atom import AtomicDFTSolver

solver = AtomicDFTSolver(atomic_number=13, xc_functional="GGA_PBE")
results = solver.solve()
print(results["energy"])
```


## Features

* **Finite-element discretization** — Real-space mesh and operators in `atom.mesh`.
* **Pseudopotentials** — Norm-conserving pseudopotential support (e.g. psp8) in `atom.pseudo`.
* **SCF driver** — Density, Hamiltonian, eigensolver, Poisson, mixing, and convergence in `atom.scf`.
* **Exchange–correlation** — LDA, GGA-PBE, HF / hybrids, meta-GGA, OEP, RPA, and optional ML-XC in `atom.xc`.
* **Helpers** — Occupation states and related utilities in `atom.utils`; data helpers in `atom.data`.


## Quick start

Ensure `utils/` (the parent of this package) is on `PYTHONPATH`, then:

```python
from atom import AtomicDFTSolver

solver = AtomicDFTSolver(atomic_number=29, xc_functional="GGA_PBE")
results = solver.solve()

# For PDOS-style radial orbitals on a uniform grid:
results = solver.solve(evaluate_basis_on_uniform_grid=True)
```

Pseudopotential calculations need a valid `psp_dir_path` / `psp_file_name`
(or the solver defaults when configured). See the upstream project for full
API options.


## Package layout

| Path | Description |
|------|-------------|
| `mesh/` | Grid construction and operators |
| `pseudo/` | Pseudopotential reading and evaluation (local / non-local) |
| `scf/` | SCF loop: density, Hamiltonian, eigensolver, Poisson, mixer |
| `xc/` | XC functionals: LDA, GGA, HF, ML-XC, OEP, RPA, etc. |
| `data/` | Data generation, loading, and processing helpers |
| `utils/` | Occupation states, periodicity / result helpers |
| `solver.py` | `AtomicDFTSolver` entry point |


## Requirements

* Python ≥ 3.8
* NumPy ≥ 1.20
* SciPy ≥ 1.7

Optional (only if you use the corresponding modules): PyTorch / scikit-learn for
`xc.ml_xc`, Matplotlib for plotting helpers, `threadpoolctl` for RPA thread control.


## Upstream

Development, tests, documentation, and packaged releases are maintained in
**[SPARC-atomSFE](https://github.com/SPARC-X/SPARC-atomSFE)**. Please report
issues and contributions there.

If you use this code in research, please cite SPARC-atomSFE (see that
repository for the current citation).


## License

Licensed under **GNU GPLv3** (same as SPARC / SPARC-atomSFE).


## Acknowledgement

* **U.S. Department of Energy (DOE), Office of Science (SC): DE-SC0023445**
