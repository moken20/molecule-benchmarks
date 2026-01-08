from itertools import product

from pyscf import cc
from pyscf.ci.cisd import amplitudes_to_cisdvec as amplitudes_to_cisdvec_rhf
from pyscf.fci.spin_op import spin_square0
from pyscf.ci.cisd import to_fcivec as to_fcivec_rhf
import numpy as np

from molbench.utils.chem_tools import excited_configurations
from molbench.wfn import WFN

def solve_ccsd(solver_hf, max_cycle=200, frozen_orbs=None, verbose=0):
    r"""Execute PySCF's coupled cluster with singles and doubles (CCSD) solver.
    Unlike the other solvers, CCSD from PySCF has no capability to return wavefunctions
    for excited states, only energies.

    Args:
        solver_hf (object): PySCF's Hartree-Fock Solver object.
        max_cycle (int): Maximum number of iterations in the CCSD procedure.
        frozen_orbs (list(int)): List of orbital indices to freeze.
        verbose (int): Integer specifying the verbosity level (passed directly into PySCF's solver).

    Returns:
        solver_ccsd (object): PySCF's CCSD Solver object.
        e (array): Energies of `nroots` lowest energy eigenstates.
        ss (array): Spin-squared ($S^2$) of lowest energy eigenstate. If `nroots > 1`, remainder are marked N/A.
        sz (array): Spin projection ($S_z$) of lowest energy eigenstate. If `nroots > 1`, remainder are marked N/A.
    """

    hftype = solver_hf.__class__.__name__.lower()

    if "uhf" in hftype.lower() or "rohf" in hftype.lower():
        # NOTE: `frozen` must be provided at construction time in PySCF
        solver_ccsd = cc.UCCSD(solver_hf, frozen=frozen_orbs).run(verbose=verbose, max_cycle=max_cycle)
        ss, mult = solver_ccsd.spin_square()
    elif "rhf" in hftype.lower():
        assert solver_hf.mol.spin == 0, f"Cannot run RCCSD with nonzero Sz."
        # NOTE: `frozen` must be provided at construction time in PySCF
        solver_ccsd = cc.RCCSD(solver_hf, frozen=frozen_orbs).run(verbose=verbose, max_cycle=max_cycle)
        cisdvec = amplitudes_to_cisdvec_rhf(1.0, solver_ccsd.t1, solver_ccsd.t2)
        # Derive active dimensions from the returned amplitudes (most reliable across PySCF versions
        # when freezing arbitrary orbital subsets).
        active_nocc, active_nvir = solver_ccsd.t1.shape
        active_norb = active_nocc + active_nvir
        # PySCF's to_fcivec expects `nelec` (not nocc). Use closed-shell tuple (na, nb).
        fcivec = to_fcivec_rhf(cisdvec, active_norb, (active_nocc, active_nocc))
        ss, mult = spin_square0(fcivec, active_norb, (active_nocc, active_nocc))
    else:
        raise ValueError(f"Unknown HF reference character: {solver_hf.__class__.__name__}")

    e = [solver_ccsd.e_tot]
    ss = [ss]
    sz = [mult - 1]

    return solver_ccsd, np.array(e), np.array(ss), np.array(sz)


def extract_ccsd_state(solver_ccsd, tol=1e-15):
    r"""Wrapper that constructs a WFN object representing 
    the wavefunction from the restricted or unrestricted coupled cluster 
    with singles and doubles (RCCSD/UCCSD) solution in PySCF. It does so 
    by redirecting the flow to the appropriate constructor function.

    PySCF's implementation of CCSD does not support calculation of excited state wavefunctions,
    so unlike all other state generation methods, the `ccsd_state` method does not support the 
    `state` argument for selection of excited states.

    The generated wavefunction is stored in a WFN object where alpha and beta bitstring integers
    represent configurations (Slater determinants) and coefficients are the CI coefficients.
    The binary representation of these integers correspond to a specific configuration: 
    the first number represents the configuration of the alpha electrons and the second 
    number represents the configuration of the beta electrons. For instance, the Hartree-Fock 
    state :math:`|1 1 0 0 \rangle` will be represented by the flipped binary string ``0011`` 
    which is split to ``01`` and ``01`` for the alpha and beta electrons. The integer 
    corresponding to ``01`` is ``1`` and the WFN representation of the Hartree-Fock state 
    will have alpha=[1], beta=[1], coeff=[1.0].

    In the current version, the exponential ansatz :math:`\exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}}`
    is expanded to second order, with only single and double excitation terms collected and kept.
    In the future this will be amended to also collect terms from higher order. The expansion gives

    .. math::
        \exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}} = \left[ 1 + \hat{T}_1 +
        \left( \hat{T}_2 + 0.5 * \hat{T}_1^2 \right) \right] \ket{\text{HF}}

    The coefficients in this expansion are the CI coefficients used to build the wavefunction
    representation.

    Args:
        solver_ccsd (object): PySCF RCCSD/UCCSD Solver object
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (WFN): wavefunction in WFN format
    """
    hftype = str(solver_ccsd.__str__)
    if "uccsd" in hftype.lower():
        wf = _uccsd_state(solver_ccsd, tol=tol)
    elif "ccsd" in hftype.lower() and not ("uccsd" in hftype.lower()):
        wf = _rccsd_state(solver_ccsd, tol=tol)
    else:
        raise ValueError("Unknown HF reference character. The only supported types are RHF, ROHF and UHF.")

    return wf


def _rccsd_state(solver_ccsd, tol=1e-15):
    r""" Construct a WFN object representing the wavefunction from the restricted
    coupled cluster with singles and doubles (RCCSD) solution in PySCF. 
    [This function is copied on PennyLane's qchem.convert._rccsd_state method
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L760.]

    The generated wavefunction is stored in a WFN object where alpha and beta bitstring integers
    represent configurations (Slater determinants) and coefficients are the CI coefficients.
    The binary representation of these integers correspond to a specific configuration: 
    the first number represents the configuration of the alpha electrons and the second 
    number represents the configuration of the beta electrons. For instance, the Hartree-Fock 
    state :math:`|1 1 0 0 \rangle` will be represented by the flipped binary string ``0011`` 
    which is split to ``01`` and ``01`` for the alpha and beta electrons. The integer 
    corresponding to ``01`` is ``1`` and the WFN representation of the Hartree-Fock state 
    will have alpha=[1], beta=[1], coeff=[1.0].

    In the current version, the exponential ansatz :math:`\exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}}`
    is expanded to second order, with only single and double excitation terms collected and kept.
    In the future this will be amended to also collect terms from higher order. The expansion gives

    .. math::
        \exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}} = \left[ 1 + \hat{T}_1 +
        \left( \hat{T}_2 + 0.5 * \hat{T}_1^2 \right) \right] \ket{\text{HF}}

    The coefficients in this expansion are the CI coefficients used to build the wavefunction
    representation.

    Args:
        solver_ccsd (object): PySCF CCSD Solver object (restricted)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (WFN): wavefunction in WFN format
    """

    mol = solver_ccsd.mol

    # Get active space dimensions from CCSD solver (accounts for frozen orbitals)
    # nmo: number of active MOs, nocc: number of active occupied orbitals
    norb = solver_ccsd.nmo
    nocc = solver_ccsd.nocc
    nelec_a = nocc
    nelec_b = nocc
    
    if not (nelec_a == nelec_b):
        raise ValueError("For RHF-based CCSD the molecule must be closed shell.")

    nvir_a, nvir_b = norb - nelec_a, norb - nelec_b

    # build the full, unrestricted representation of the coupled cluster amplitudes
    t1a = solver_ccsd.t1
    t1b = t1a
    t2aa = solver_ccsd.t2 - solver_ccsd.t2.transpose(1, 0, 2, 3)
    t2ab = solver_ccsd.t2.transpose(0, 2, 1, 3)
    t2bb = t2aa

    # add in the disconnected part ( + 0.5 T_1^2) of double excitations
    t2aa = (
        t2aa
        - 0.5
        * np.kron(t1a, t1a).reshape(nelec_a, nvir_a, nelec_a, nvir_a).transpose(0, 2, 1, 3)
    )
    t2bb = (
        t2bb
        - 0.5
        * np.kron(t1b, t1b).reshape(nelec_b, nvir_b, nelec_b, nvir_b).transpose(0, 2, 1, 3)
    )
    # align the entries with how the excitations are ordered when generated by excited_configurations()
    t2ab = t2ab - 0.5 * np.kron(t1a, t1b).reshape(nelec_a, nvir_a, nelec_b, nvir_b)

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    # Build arrays directly instead of dict
    alpha_list = [ref_a]
    beta_list = [ref_b]
    coeff_list = [1.0]

    # alpha -> alpha excitations
    t1a_configs, t1a_signs = excited_configurations(nelec_a, norb, 1)
    alpha_list.extend(t1a_configs)
    beta_list.extend([ref_b] * len(t1a_configs))
    coeff_list.extend(t1a.ravel() * t1a_signs)

    # beta -> beta excitations
    t1b_configs, t1b_signs = excited_configurations(nelec_b, norb, 1)
    alpha_list.extend([ref_a] * len(t1b_configs))
    beta_list.extend(t1b_configs)
    coeff_list.extend(t1b.ravel() * t1b_signs)

    # alpha, alpha -> alpha, alpha excitations
    if nelec_a > 1 and nvir_a > 1:
        t2aa_configs, t2aa_signs = excited_configurations(nelec_a, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_a, -1)
        vvidx = np.tril_indices(nvir_a, -1)
        t2aa = t2aa[ooidx][:, vvidx[0], vvidx[1]]
        alpha_list.extend(t2aa_configs)
        beta_list.extend([ref_b] * len(t2aa_configs))
        coeff_list.extend(t2aa.ravel() * t2aa_signs)

    if nelec_b > 1 and nvir_b > 1:
        t2bb_configs, t2bb_signs = excited_configurations(nelec_b, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_b, -1)
        vvidx = np.tril_indices(nvir_b, -1)
        t2bb = t2bb[ooidx][:, vvidx[0], vvidx[1]]
        alpha_list.extend([ref_a] * len(t2bb_configs))
        beta_list.extend(t2bb_configs)
        coeff_list.extend(t2bb.ravel() * t2bb_signs)

    # alpha, beta -> alpha, beta excitations
    rowvals, colvals = np.array(list(product(t1a_configs, t1b_configs)), dtype=int).T
    alpha_list.extend(rowvals)
    beta_list.extend(colvals)
    coeff_list.extend(t2ab.ravel() * np.kron(t1a_signs, t1b_signs))

    alpha = np.array(alpha_list, dtype=np.uint64)
    beta = np.array(beta_list, dtype=np.uint64)
    coeff = np.array(coeff_list, dtype=np.float64)

    # renormalize, to get the HF coefficient (CC wavefunction not normalized)
    norm = np.sqrt(np.sum(coeff ** 2))
    coeff = coeff / norm

    return WFN.from_arrays(alpha, beta, coeff, n_orb=norb, tol=tol)

def _uccsd_state(solver_ccsd, tol=1e-15):
    r""" Construct a WFN object representing the wavefunction from the unrestricted
    coupled cluster with singles and doubles (UCCSD) solution in PySCF. 
    [This function is copied on PennyLane's qchem.convert._uccsd_state method
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L893.]

    The generated wavefunction is stored in a WFN object where alpha and beta bitstring integers
    represent configurations (Slater determinants) and coefficients are the CI coefficients.
    The binary representation of these integers correspond to a specific configuration: 
    the first number represents the configuration of the alpha electrons and the second 
    number represents the configuration of the beta electrons. For instance, the Hartree-Fock 
    state :math:`|1 1 0 0 \rangle` will be represented by the flipped binary string ``0011`` 
    which is split to ``01`` and ``01`` for the alpha and beta electrons. The integer 
    corresponding to ``01`` is ``1`` and the WFN representation of the Hartree-Fock state 
    will have alpha=[1], beta=[1], coeff=[1.0].

    In the current version, the exponential ansatz :math:`\exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}}`
    is expanded to second order, with only single and double excitation terms collected and kept.
    In the future this will be amended to also collect terms from higher order. The expansion gives

    .. math::
        \exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}} = \left[ 1 + \hat{T}_1 +
        \left( \hat{T}_2 + 0.5 * \hat{T}_1^2 \right) \right] \ket{\text{HF}}

    The coefficients in this expansion are the CI coefficients used to build the wavefunction
    representation.

    Args:
        solver_ccsd (object): PySCF UCCSD Solver object (unrestricted)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (WFN): wavefunction in WFN format
    """

    mol = solver_ccsd.mol

    # Get active space dimensions from CCSD solver (accounts for frozen orbitals)
    # For UCCSD, nmo and nocc can be tuples (nmo_a, nmo_b) and (nocc_a, nocc_b)
    nmo = solver_ccsd.nmo
    nocc = solver_ccsd.nocc
    if isinstance(nmo, tuple):
        norb = nmo[0]  # Assume same for alpha and beta
        nelec_a, nelec_b = nocc
    else:
        norb = nmo
        nelec_a = nocc
        nelec_b = nocc

    nvir_a, nvir_b = norb - nelec_a, norb - nelec_b

    t1a, t1b = solver_ccsd.t1
    t2aa, t2ab, t2bb = solver_ccsd.t2
    # add in the disconnected part ( + 0.5 T_1^2) of double excitations
    t2aa = (
        t2aa
        - 0.5
        * np.kron(t1a, t1a).reshape(nelec_a, nvir_a, nelec_a, nvir_a).transpose(0, 2, 1, 3)
    )
    t2bb = (
        t2bb
        - 0.5
        * np.kron(t1b, t1b).reshape(nelec_b, nvir_b, nelec_b, nvir_b).transpose(0, 2, 1, 3)
    )
    # align the entries with how the excitations are ordered when generated by excited_configurations()
    t2ab = (
        t2ab.transpose(0, 2, 1, 3)
        - 0.5 * np.kron(t1a, t1b).reshape(nelec_a, nvir_a, nelec_b, nvir_b)
    )

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    # Build arrays directly instead of dict
    alpha_list = [ref_a]
    beta_list = [ref_b]
    coeff_list = [1.0]

    # alpha -> alpha excitations
    t1a_configs, t1a_signs = excited_configurations(nelec_a, norb, 1)
    alpha_list.extend(t1a_configs)
    beta_list.extend([ref_b] * len(t1a_configs))
    coeff_list.extend(t1a.ravel() * t1a_signs)

    # beta -> beta excitations
    t1b_configs, t1b_signs = excited_configurations(nelec_b, norb, 1)
    alpha_list.extend([ref_a] * len(t1b_configs))
    beta_list.extend(t1b_configs)
    coeff_list.extend(t1b.ravel() * t1b_signs)

    # alpha, alpha -> alpha, alpha excitations
    if nelec_a > 1 and nvir_a > 1:
        t2aa_configs, t2aa_signs = excited_configurations(nelec_a, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_a, -1)
        vvidx = np.tril_indices(nvir_a, -1)
        t2aa = t2aa[ooidx][:, vvidx[0], vvidx[1]]
        alpha_list.extend(t2aa_configs)
        beta_list.extend([ref_b] * len(t2aa_configs))
        coeff_list.extend(t2aa.ravel() * t2aa_signs)

    if nelec_b > 1 and nvir_b > 1:
        t2bb_configs, t2bb_signs = excited_configurations(nelec_b, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_b, -1)
        vvidx = np.tril_indices(nvir_b, -1)
        t2bb = t2bb[ooidx][:, vvidx[0], vvidx[1]]
        alpha_list.extend([ref_a] * len(t2bb_configs))
        beta_list.extend(t2bb_configs)
        coeff_list.extend(t2bb.ravel() * t2bb_signs)

    # alpha, beta -> alpha, beta excitations
    rowvals, colvals = np.array(list(product(t1a_configs, t1b_configs)), dtype=int).T
    alpha_list.extend(rowvals)
    beta_list.extend(colvals)
    coeff_list.extend(t2ab.ravel() * np.kron(t1a_signs, t1b_signs))

    alpha = np.array(alpha_list, dtype=np.uint64)
    beta = np.array(beta_list, dtype=np.uint64)
    coeff = np.array(coeff_list, dtype=np.float64)

    # renormalize, to get the HF coefficient (CC wavefunction not normalized)
    norm = np.sqrt(np.sum(coeff ** 2))
    coeff = coeff / norm

    return WFN.from_arrays(alpha, beta, coeff, n_orb=norb, tol=tol)