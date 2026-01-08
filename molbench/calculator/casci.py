from pyscf import mcscf
from pyscf.fci.cistring import addrs2str
import numpy as np
from scipy.sparse import coo_matrix

from molbench.utils.chem_tools import verify_active_space
from molbench.wfn import WFN


def solve_casci(solver_hf, ncas, nelecas, nroots=1, maxiter=2000, verbose=0):
    r"""Execute PySCF's complete active space configuration interaction (CASCI) solver.

    Args:
        solver_hf (object): PySCF's Hartree-Fock Solver object
        ncas (int): Number of active orbitals
        nelecas (tuple(int, int)): Number of active electrons in spin-up (alpha) and
            spin-down (beta) sectors. A warning is raised if `spin` variable of mf.mol
            disagrees with the number of active electrons.
        nroots (int): Number of low-energy eigenstates to solve for.
        maxiter (int): Maximum allowed number of iterations for the Davidson iteration.
        verbose (int): Integer specifying the verbosity level (passed directly into PySCF's solver).

    Returns:
        solver_casci (object): PySCF CASCI Solver object
        e (array): Energies of `nroots` lowest energy eigenstates.
        ss (array): Spin-squared ($S^2$) of `nroots` lowest energy eigenstates.
        sz (array): Spin projection ($S_z$) of `nroots` lowest energy eigenstates.
    """

    verify_active_space(solver_hf.mol, ncas, nelecas)

    # infer restricted or unrestricted type from the passed HF solver
    hftype = str(solver_hf.__str__)
    if "rhf" in hftype.lower() or "rohf" in hftype.lower():
        solver_casci = mcscf.CASCI(solver_hf, ncas, nelecas)
        solver_casci.fix_spin_(ss=solver_hf.mol.spin)
    elif "uhf" in hftype.lower():
        solver_casci = mcscf.UCASCI(solver_hf, ncas, nelecas)
    else:
        raise ValueError("Only RHF/ROHF and UHF solvers are supported.")

    # make sure input orbitals are same as output -- turn off
    # canonicalization and natural orbitals
    solver_casci.canonicalization = False
    solver_casci.natorb = False
    solver_casci.fcisolver.nroots = nroots
    solver_casci.fcisolver.spin = solver_hf.mol.spin
    solver_casci.fcisolver.max_cycle = maxiter

    solver_casci.run(verbose=verbose)
    if verbose > 2:
        solver_casci.analyze()

    e = np.atleast_1d(solver_casci.e_tot)

    ss, sz = [], []
    try:
        if nroots > 1:
            for ii in range(len(solver_casci.ci)):
                ssval, multval = solver_casci.fcisolver.spin_square(
                    solver_casci.ci[ii], ncas, nelecas
                )
                ss.append(ssval)
                sz.append(multval - 1)
        else:
            ssval, multval = solver_casci.fcisolver.spin_square(solver_casci.ci, ncas, nelecas)
            ss.append(ssval)
            sz.append(multval - 1)
    except ValueError:  # if wavefunction is too big
        ss = ["N/A"] * nroots
        sz = [solver_hf.mol.spin] * nroots

    return solver_casci, e, np.array(ss), np.array(sz)


def extract_casci_state(solver_casci, state=0, tol=1e-15):
    r""" Construct a WFN object representing the wavefunction from the complete
    active space configuration interaction (CASCI) solution in PySCF. 
    
    The generated wavefunction is stored in a WFN object where alpha and beta bitstring integers
    represent configurations (Slater determinants) and coefficients are the CI coefficients.
    The binary representation of these integers correspond to a specific configuration: 
    the first number represents the configuration of the alpha electrons and the second 
    number represents the configuration of the beta electrons. For instance, the Hartree-Fock 
    state :math:`|1 1 0 0 \rangle` will be represented by the flipped binary string ``0011`` 
    which is split to ``01`` and ``01`` for the alpha and beta electrons. The integer 
    corresponding to ``01`` is ``1`` and the WFN representation of the Hartree-Fock state 
    will have alpha=[1], beta=[1], coeff=[1.0].

    Args:
        solver_casci (object): PySCF CASCI Solver object (restricted or unrestricted)
        state (int): which state to do the conversion for, if within CASCI multiple states were solved for (i.e. `nroots > 1`) -- default is 0 (the lowest-energy state)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (WFN): wavefunction in WFN format
    """

    # Get active space dimensions from CASCI solver
    ncas = solver_casci.ncas
    nelecas = solver_casci.nelecas
    if isinstance(nelecas, (list, tuple)):
        nelecas_a, nelecas_b = nelecas
    else:
        nelecas_a = nelecas_b = nelecas // 2

    # extract the CI coeffs from the right state
    if not (state in range(solver_casci.fcisolver.nroots)):
        raise IndexError("State requested has not been solved for. Re-run CASCI with larger nroots.")
    if solver_casci.fcisolver.nroots > 1:
        cascivec = solver_casci.ci[state]
    else:
        cascivec = solver_casci.ci

    # filter determinants with coefficients below tol (for sparse extraction)
    cascivec_filtered = cascivec.copy()
    cascivec_filtered[abs(cascivec_filtered) < tol] = 0
    sparse_cascimatr = coo_matrix(cascivec_filtered, shape=np.shape(cascivec_filtered), dtype=float)
    row, col, dat = sparse_cascimatr.row, sparse_cascimatr.col, sparse_cascimatr.data

    ## turn FCI wavefunction matrix indices into integers representing Fock occupation vectors
    # Use active space dimensions only (no padding for core electrons)
    alpha = addrs2str(ncas, nelecas_a, row)
    beta = addrs2str(ncas, nelecas_b, col)
    coeff = dat

    return WFN.from_arrays(alpha, beta, coeff, n_orb=ncas, tol=0.0)