# molecule-benchmarks

> 日本語版READMEは [`README_ja.md`](README_ja.md) にあります。

This repository provides a small benchmark suite to run **classical quantum chemistry algorithms** (HF/CISD/CCSD/CASCI/DMRG) for molecules using PySCF / Block2, and to compare quantities such as energies and spin values under a unified interface.

## Overview

- **Recommended entry point**: run the top-level `main.py` via `python main.py ...` (configuration is managed by Hydra)
- **Notebooks**: run analyses by calling `Orchestrator` directly in `tutorial/`
- **Main outputs**:
  - Energy
  - \(S^2\) (spin squared)
  - \(S_z\) (spin projection)
  - Wavefunction coefficients over Slater determinants (`WFN` in `molbench/wfn.py`)

## Quick Start (CLI: `python main.py`)

### Setup

```bash
git clone <this-repo>
cd molecule-benchmarks

python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
```

### Run (default settings)

```bash
python main.py
```

### 1. Change settings (edit `config/default.yaml`)

Instead of relying on command-line overrides, the intended workflow is to edit `config/default.yaml` and switch settings there.

#### Select a molecule (choose a config group)

Edit `defaults` → `molecule:`. For example, to use N2:

```yaml
defaults:
  - molecule: N2
```

#### Enable/disable algorithms (ON/OFF)

Toggle `algorithms.<method>.run_flag` (HF always runs).

```yaml
algorithms:
  cisd:
    run_flag: true
  ccsd:
    run_flag: false
  casci:
    run_flag: true
  dmrg:
    run_flag: false
```

#### Common global settings (number of roots / cutoff)

```yaml
# how many low-energy roots (excited states) to compute
nroots: 2

# cutoff used when extracting wavefunctions (determinant coefficients)
ci_cutoff: 1e-6
```

#### Change the active space (`nelecas`, `norbcas`)

The active space (`nelecas`, `norbcas`) is **not meant to be fixed**—adjust it freely depending on your goal. You have two options:

- **Edit the molecule YAML directly**: change `nelecas` / `norbcas` in `config/molecule/<NAME>.yaml`
- **Override in `config/default.yaml`**: add `molecule.*` keys to overwrite the selected molecule config

Example (overwrite from `config/default.yaml`):

```yaml
molecule:
  nelecas: 10
  norbcas: 8
```

After editing, just run:

```bash
python main.py
```

### 2. Command-line overrides (optional)

The primary workflow is to edit `config/default.yaml`, but you can also temporarily override settings using Hydra overrides:

```bash
# Example: temporarily switch only the molecule
python main.py molecule=N2

# Example: temporarily disable only DMRG
python main.py algorithms.dmrg.run_flag=false
```

## Algorithms

Methods are executed via `Orchestrator.do_*` in `molbench/orchestrator.py`.

- **HF**: `do_hf()` (reference state)
- **CISD**: `do_cisd()` (configuration interaction singles and doubles)
- **CCSD**: `do_ccsd()` (coupled cluster singles and doubles)
- **CASCI**: `do_casci()` (CI within the chosen active space)
- **DMRG**: `do_dmrg()` (MPS/DMRG via Block2)

Most settings are defined in `config/default.yaml` under `algorithms.*` (e.g., `max_cycle` for CCSD, and `schedule` / `max_mem` / `workdir` for DMRG).

### Commonly edited keys

| method | implementation | ON/OFF | common keys |
|---|---|---|---|
| HF | `do_hf()` | always on | `algorithms.hf.verbose` |
| CISD | `do_cisd()` | `algorithms.cisd.run_flag` | `algorithms.cisd.verbose`, `nroots`, `ci_cutoff` |
| CCSD | `do_ccsd()` | `algorithms.ccsd.run_flag` | `algorithms.ccsd.max_cycle`, `algorithms.ccsd.verbose`, `ci_cutoff` |
| CASCI | `do_casci()` | `algorithms.casci.run_flag` | `algorithms.casci.maxiter`, `nroots`, `ci_cutoff` |
| DMRG | `do_dmrg()` | `algorithms.dmrg.run_flag` | `algorithms.dmrg.schedule`, `algorithms.dmrg.max_mem`, `algorithms.dmrg.workdir`, `algorithms.dmrg.n_threads` |

### DMRG (Block2) parameters (more details)

DMRG approximates the wavefunction in the active space as an MPS (matrix product state) and converges it in stages. You mainly tune `algorithms.dmrg.*` in `config/default.yaml`.

- **`schedule`**: the staged convergence strategy. It is a set of four lists (same format as used in the notebooks):
  - **bond dimensions**: MPS bond dimensions per stage (larger → more accurate / more expensive)
  - **sweeps**: number of sweeps per stage (more sweeps → easier convergence / slower)
  - **noise**: noise values to help convergence (often set to 0.0 in the final stage)
  - **Davidson thresholds**: diagonalization thresholds (smaller → stricter / slower)

Example (a reasonable starting point for smaller active spaces):

```python
schedule = [
    [10, 20, 30],          # bond dimensions
    [5, 5, 5],             # sweeps
    [1e-4, 1e-5, 0.0],     # noise
    [1e-6, 1e-8, 1e-10],   # Davidson thresholds
]
```

- **`dot`**: 1-site / 2-site DMRG (2-site is often more robust)
- **`max_mem`**: memory limit (MB)
- **`n_threads`**: number of threads
- **`workdir`**: working directory for scratch/intermediate files
- **`tol`**: convergence tolerance (e.g., energy convergence)
- **`restart_ket`**: restart configuration (only when needed)
- **`smp_tol`**: threshold related to determinant extraction/sampling (smaller → stricter / slower)

Practical guidelines:
- **Need higher accuracy**: increase bond dimensions → increase sweeps → tighten final Davidson thresholds
- **Divergence / poor convergence**: use 2-site (`dot: 2`), increase intermediate noise slightly, increase sweeps
- **Too slow/heavy**: reduce bond dimensions, reduce the number of stages, reduce sweeps

## Molecules

Molecule definitions live in `config/molecule/*.yaml`.

The active space (`nelecas`, `norbcas`) is **not meant to be fixed**—override it freely from the CLI or notebooks (e.g., `molecule.nelecas=...` / `molecule.norbcas=...`).

**Supported molecules** (current set; you can freely add more):

```text
H2  H4  H24  LiH  CO  N2  CH4  C10H8  Cr2  Fe2S2
```

## Notebooks

- `tutorial/1_calculate_energy.ipynb`
  - loads molecule settings via `load_molecule_cfg()` and computes energies for each method
  - example of scanning bond lengths and plotting energy curves
- `tutorial/2_analyze_wfn.ipynb`
  - uses CASCI as a reference and compares wavefunction overlaps for HF/CISD/CCSD/DMRG
  - demonstrates increasing multi-reference character at stretched geometries
  - overlap is computed by `wf_overlap()` in `molbench/wfn.py`

## Troubleshooting / FAQ

- **DMRG is slow / seems stuck**: setting `algorithms.dmrg.workdir` to a local path (e.g., `/tmp/dmrg_calc_tmp`) can help.
- **Out of memory**: adjust `algorithms.dmrg.max_mem`, reduce `schedule` bond dimensions, and/or tune `n_threads`.
- **MPI import errors**: `mpi4py` depends on your environment; an MPI implementation (e.g., OpenMPI) may be required.

## Roadmap

- Add an HCI algorithm
- Add more molecules (`config/molecule/*.yaml`)
- CLI polish:
  - `pyproject.toml` defines `molbench = "molbench.main:main"`, but the current entry point is `python main.py` (to be aligned in the future).