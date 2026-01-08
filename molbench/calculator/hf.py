from pyscf import scf, gto
import numpy as np

from molbench.utils.chem_tools import get_mol_attrs
from molbench.wfn import WFN


def solve_hf(mol: gto.M, verbose: int = 0):
    r"""Execute PySCF's Hartree-Fock solvers.

    Args:
        mol (object): PySCF Molecule object
        hftype (str): String specifying the type of HF calculation to do. Currently supported are
            "rhf", "rohf" and "uhf".
        verbose (int): Integer specifying the verbosity level (passed directly into PySCF's solver).

    Returns:
        solver_hf (object): PySCF Hartree-Fock Solver object
        e (array): HF energy
        ss (array): Spin-squared ($S^2$) of the output wavefunction
        sz (array): Spin projection ($S_z$) of the output wavefunction
    """
    if mol.spin % 2 == 0:
      solver_hf = scf.RHF(mol).run(verbose=verbose)
    elif mol.spin % 2 == 1:
      solver_hf = scf.UHF(mol).run(verbose=verbose)

    if verbose > 0:
        solver_hf.analyze()
    e = np.atleast_1d(solver_hf.e_tot)
    ss, mult = solver_hf.spin_square()
    ss = np.array([ss])
    sz = np.array([mult - 1])

    return solver_hf, e, ss, sz


def extract_hf_state(solver_hf, n_orb=None, nelec=None, tol=1e-15):
    r""" Construct a WFN object representing the wavefunction from the Hartree-Fock
    solution in PySCF. 
    [This function is inspired by PennyLane's qchem.convert._rcisd_state method 
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L664.]

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
        solver_hf (object): PySCF Hartree-Fock Solver object (restricted or unrestricted)
        n_orb (int, optional): Number of orbitals for WFN storage. If None, uses full molecular orbitals.
            If specified, extracts only the active space.
        nelec (tuple(int, int), optional): Number of electrons in active space (nalpha, nbeta).
            Required if n_orb is specified.
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (WFN): wavefunction in WFN format
    """

    norb_full, nelec_a_full, nelec_b_full = get_mol_attrs(solver_hf.mol)
    
    # If n_orb is specified, extract active space only
    if n_orb is not None:
        if nelec is None:
            raise ValueError("nelec must be specified when n_orb is provided for active space extraction")
        nelec_a, nelec_b = nelec
        norb = n_orb
    else:
        nelec_a = nelec_a_full
        nelec_b = nelec_b_full
        norb = norb_full

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    alpha = np.array([ref_a], dtype=np.uint64)
    beta = np.array([ref_b], dtype=np.uint64)
    coeff = np.array([1.0], dtype=np.float64)

    return WFN.from_arrays(alpha, beta, coeff, n_orb=norb, tol=tol)