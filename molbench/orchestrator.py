from pyscf import gto

from molbench.calculator.hf import solve_hf, extract_hf_state
from molbench.calculator.cisd import solve_cisd, extract_cisd_state
from molbench.calculator.ccsd import solve_ccsd, extract_ccsd_state
from molbench.calculator.casci import solve_casci, extract_casci_state
from molbench.calculator.dmrg import solve_dmrg, extract_dmrg_state
from molbench.schema import MolResult

class Orchestrator:
  def __init__(self, geometry, basis, nelecas, norbcas, spin, charge, run_hf: bool = True):
    self.geometry = geometry
    self.basis = basis

    nalpha = (nelecas + spin) // 2
    nbeta = (nelecas - spin) // 2
    self.nelec = (nalpha, nbeta)
    self.norb = norbcas
    self.spin = spin
    self.mol = gto.M(atom=geometry, basis=basis, charge=charge, spin=spin, unit='Angstrom')
    self.hf_solver = None
    self.hf_result = self.do_hf() if run_hf else None
    self.active_indices = self._get_active_indices()
    self.frozen_indices = self._get_frozen_indices()


  def _get_active_indices(self):
    ncore = (self.mol.nelectron - sum(self.nelec)) // 2
    active_indices = list(range(ncore, ncore + self.norb))
    return active_indices
  

  def _get_frozen_indices(self):
    """Get frozen orbital indices based on active space."""
    if self.hf_solver is None:
      raise ValueError("HF solver not found. Run HF calculation first.")
    nmo = self.hf_solver.mo_coeff.shape[1]
    active_set = set(self.active_indices)
    frozen_indices = [i for i in range(nmo) if i not in active_set]
    return frozen_indices


  def do_hf(self, verbose: int = 0, ci_cutoff: float = 1e-6):
    if self.hf_solver is not None:
      return self.hf_result
    solver_hf, e, ss, sz = solve_hf(self.mol, verbose)
    self.hf_solver = solver_hf
    # Extract HF state in active space only (for consistency with other methods)
    wfn = extract_hf_state(solver_hf, n_orb=self.norb, nelec=self.nelec, tol=ci_cutoff)
    return MolResult(algorithm="hf", energy=e[0], spin_squared=ss[0], spin_projection=sz[0], wfn=wfn)
  

  def do_cisd(self, nroots: int = 1, verbose: int = 0, ci_cutoff: float = 1e-6):
    if self.hf_solver is None:
      raise ValueError("HF solver not found. Running HF calculation first.")

    solver_cisd, e, ss, sz = solve_cisd(self.hf_solver, nroots=nroots, frozen_orbs=self.frozen_indices, verbose=verbose)
    results = []
    for i in range(nroots):
      wfn = extract_cisd_state(solver_cisd, i, ci_cutoff)
      results.append(MolResult(algorithm="cisd", energy=e[i], spin_squared=ss[i], spin_projection=sz[i], wfn=wfn))
    return results[0] if nroots == 1 else results
  

  def do_ccsd(self, verbose: int = 0, ci_cutoff: float = 1e-6, max_cycle: int = 200):
    if self.hf_solver is None:
      raise ValueError("HF solver not found. Running HF calculation first.")
    
    solver_ccsd, e, ss, sz = solve_ccsd(
      self.hf_solver,
      max_cycle=max_cycle,
      frozen_orbs=self.frozen_indices,
      verbose=verbose,
    )
    wfn = extract_ccsd_state(solver_ccsd, ci_cutoff)
    return MolResult(algorithm="ccsd", energy=e[0], spin_squared=ss[0], spin_projection=sz[0], wfn=wfn)
  

  def do_casci(self, nroots: int = 1, verbose: int = 0, ci_cutoff: float = 1e-6, maxiter: int = 2000):
    if self.hf_solver is None:
      raise ValueError("HF solver not found. Running HF calculation first.")

    solver_casci, e, ss, sz = solve_casci(
      self.hf_solver,
      ncas=self.norb,
      nelecas=self.nelec,
      nroots=nroots,
      maxiter=maxiter,
      verbose=verbose,
    )
    results = []
    for i in range(nroots):
      wfn = extract_casci_state(solver_casci, i, ci_cutoff)
      results.append(MolResult(algorithm="casci", energy=e[i], spin_squared=ss[i], spin_projection=sz[i], wfn=wfn))
    return results[0] if nroots == 1 else results
  

  def do_dmrg(
    self,
    schedule: list[list[int, int, float, float]],
    nroots: int = 1,
    verbose: int = 0,
    ci_cutoff: float = 1e-6,
    **kwargs
  ):
    if self.hf_solver is None:
      raise ValueError("HF solver not found. Running HF calculation first.")

    solver_dmrg, e, ss, sz = solve_dmrg(
      self.hf_solver, self.norb, self.nelec, schedule,
      nroots=nroots, verbose=verbose, **kwargs
    )
    results = []
    for i in range(nroots):
      wfn = extract_dmrg_state(solver_dmrg, state=i, tol=ci_cutoff, n_orb=self.norb)
      results.append(MolResult(algorithm="dmrg", energy=e[i], spin_squared=ss[i], spin_projection=sz[i], wfn=wfn))
    return results[0] if nroots == 1 else results