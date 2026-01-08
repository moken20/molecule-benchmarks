from itertools import product

from pyscf import ci
from pyscf.ci.cisd import to_fcivec as to_fcivec_rhf
from pyscf.ci.ucisd import to_fcivec as to_fcivec_uhf
from pyscf.fci.spin_op import spin_square0, spin_square
import numpy as np

from molbench.utils.chem_tools import get_mol_attrs, excited_configurations
from molbench.wfn import WFN

def solve_cisd(solver_hf, nroots=1, frozen_orbs=None, verbose=0):
    r"""Execute PySCF's configuration interaction with singles and doubles (CISD) solver.

    Args:
        solver_hf (object): PySCF's Hartree-Fock Solver object.
        nroots (int): Number of low-energy eigenstates to solve for.
        frozen_orbs (list(int)): List of orbital indices to freeze.
        verbose (int): Integer specifying the verbosity level (passed directly into PySCF's solver).

    Returns:
        solver_cisd (object): PySCF's CISD Solver object.
        e (array): Energies of `nroots` lowest energy eigenstates.
        ss (array): Spin-squared ($S^2$) of `nroots` lowest energy eigenstates.
        sz (array): Spin projection ($S_z$) of `nroots` lowest energy eigenstates.
    """

    ss = []
    sz = []

    hftype = solver_hf.__class__.__name__.lower()

    if "uhf" in hftype.lower():
        solver_cisd = ci.UCISD(solver_hf)
    elif "rhf" in hftype.lower():
        solver_cisd = ci.RCISD(solver_hf)
    else:
        raise ValueError(f"Unknown HF reference character: {solver_hf.__class__.__name__}")
    solver_cisd.frozen = frozen_orbs
    solver_cisd.nroots = nroots
    solver_cisd.run(verbose=verbose)

    e = np.atleast_1d(solver_cisd.e_tot)

    # Use the CISD solver's *active* dimensions (accounts for freezing both occupied and virtual orbs).
    nmo = solver_cisd.nmo
    nocc = solver_cisd.nocc
    if isinstance(nmo, tuple):
        active_norb = nmo[0]
    else:
        active_norb = nmo

    # PySCF's to_fcivec expects `nelec` (not nocc). For CISD in canonical orbitals,
    # `nelec` equals the number of occupied orbitals for each spin.
    if isinstance(nocc, tuple):
        active_nelec = nocc
    else:
        active_nelec = (nocc, nocc)

    if nroots > 1:
        for ii in range(nroots):
            try:
                if "rhf" in hftype.lower():
                    fcivec = to_fcivec_rhf(solver_cisd.ci[ii], active_norb, active_nelec)
                    ssval, multval = spin_square0(fcivec, active_norb, active_nelec)
                elif "uhf" in hftype.lower():
                    fcivec = to_fcivec_uhf(solver_cisd.ci[ii], active_norb, active_nelec)
                    ssval, multval = spin_square(
                        fcivec,
                        active_norb,
                        active_nelec,
                        mo_coeff=solver_hf.mo_coeff,
                        ovlp=solver_hf.get_ovlp(),
                    )
                ss.append(ssval)
                sz.append(multval - 1)
            except ValueError:  # if wavefunction is too big
                ss.append("N/A")
                sz.append(solver_hf.mol.spin)
    else:
        try:
            if "rhf" in hftype.lower():
                fcivec = to_fcivec_rhf(solver_cisd.ci, active_norb, active_nelec)
                ssval, multval = spin_square0(fcivec, active_norb, active_nelec)
            elif "uhf" in hftype.lower():
                fcivec = to_fcivec_uhf(solver_cisd.ci, active_norb, active_nelec)
                ssval, multval = spin_square(
                    fcivec,
                    active_norb,
                    active_nelec,
                    mo_coeff=solver_hf.mo_coeff,
                    ovlp=solver_hf.get_ovlp(),
                )
            ss.append(ssval)
            sz.append(multval - 1)
        except ValueError:  # if wavefunction too big
            ss.append("N/A")
            sz.append(solver_hf.mol.spin)

    return solver_cisd, e, np.array(ss), np.array(sz)


def extract_cisd_state(solver_cisd, state=0, tol=1e-15):
    r"""Wrapper that constructs a WFN object representing 
    the wavefunction from the restricted or unrestricted configuration interaction 
    with singles and doubles (RCISD/UCISD) solution in PySCF. It does so 
    by redirecting the flow to the appropriate constructor function.

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
        solver_cisd (object): PySCF RCISD/UCISD Solver object
        state (int): which state to extract (default 0)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (WFN): wavefunction in WFN format
    """
    hftype = str(solver_cisd.__str__)

    if 'ucisd' in hftype.lower():
        wf = _ucisd_state(solver_cisd, state=state, tol=tol)
    elif 'cisd' in hftype.lower() and not ("ucisd" in hftype.lower()):
        wf = _rcisd_state(solver_cisd, state=state, tol=tol)
    else:
        raise ValueError("Unknown HF reference character. The only supported types are RHF, ROHF and UHF.")

    return wf


def _rcisd_state(solver_cisd, state=0, tol=1e-15):
    r""" Construct a WFN object representing the wavefunction from the restricted
    configuration interaction with singles and doubles (RCISD) solution in PySCF. 
    [This function is copied on PennyLane's qchem.convert._rcisd_state method 
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
        solver_cisd (object): PySCF RCISD Solver object (restricted)
        state (int): which state to do the conversion for, if within CISD multiple states were solved for (i.e. `nroots > 1`) -- default is 0 (the lowest-energy state)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (WFN): wavefunction in WFN format
    """

    # Get active space dimensions from CISD solver (accounts for frozen orbitals)
    norb = solver_cisd.nmo
    nocc = solver_cisd.nocc
    nvir = norb - nocc

    # extract the CI coeffs from the right state
    if not (state in range(solver_cisd.nroots)):
        raise IndexError("State requested has not been solved for. Re-run CISD with larger nroots.")
    if solver_cisd.nroots > 1:
        cisdvec = solver_cisd.ci[state]
    else:
        cisdvec = solver_cisd.ci 

    c0, c1, c2 = (
        cisdvec[0],
        cisdvec[1 : nocc * nvir + 1],
        cisdvec[nocc * nvir + 1 :].reshape(nocc, nocc, nvir, nvir),
    )

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nocc - 1)
    ref_b = ref_a

    # Build arrays directly instead of dict
    alpha_list = [ref_a]
    beta_list = [ref_b]
    coeff_list = [c0]

    # alpha -> alpha excitations
    c1a_configs, c1a_signs = excited_configurations(nocc, norb, 1)
    alpha_list.extend(c1a_configs)
    beta_list.extend([ref_b] * len(c1a_configs))
    coeff_list.extend(c1 * c1a_signs)
    
    # beta -> beta excitations
    alpha_list.extend([ref_a] * len(c1a_configs))
    beta_list.extend(c1a_configs)
    coeff_list.extend(c1 * c1a_signs)

    # check if double excitations within one spin sector (aa->aa and bb->bb) are possible
    if nocc > 1 and nvir > 1:
        # get rid of excitations from identical orbitals, double-count the allowed ones
        c2_tr = c2 - c2.transpose(1, 0, 2, 3)
        # select only unique excitations, via lower triangle of matrix (already double-counted)
        ooidx, vvidx = np.tril_indices(nocc, -1), np.tril_indices(nvir, -1)
        c2aa = c2_tr[ooidx][:, vvidx[0], vvidx[1]].ravel()

        # alpha, alpha -> alpha, alpha excitations
        c2aa_configs, c2aa_signs = excited_configurations(nocc, norb, 2)
        alpha_list.extend(c2aa_configs)
        beta_list.extend([ref_b] * len(c2aa_configs))
        coeff_list.extend(c2aa * c2aa_signs)
        
        # beta, beta -> beta, beta excitations
        alpha_list.extend([ref_a] * len(c2aa_configs))
        beta_list.extend(c2aa_configs)
        coeff_list.extend(c2aa * c2aa_signs)

    # alpha, beta -> alpha, beta excitations
    # generate all possible pairwise combinations of _single_ excitations of alpha and beta sectors
    rowvals, colvals = np.array(list(product(c1a_configs, c1a_configs)), dtype=int).T
    c2ab = (c2.transpose(0, 2, 1, 3).reshape(nocc * nvir, -1)).ravel()
    alpha_list.extend(rowvals)
    beta_list.extend(colvals)
    coeff_list.extend(c2ab * np.kron(c1a_signs, c1a_signs))

    alpha = np.array(alpha_list, dtype=np.uint64)
    beta = np.array(beta_list, dtype=np.uint64)
    coeff = np.array(coeff_list, dtype=np.float64)

    return WFN.from_arrays(alpha, beta, coeff, n_orb=norb, tol=tol)

def _ucisd_state(solver_cisd, state=0, tol=1e-15):
    r""" Construct a WFN object representing the wavefunction from the unrestricted
    configuration interaction with singles and doubles (UCISD) solution in PySCF. 
    [This function is copied on PennyLane's qchem.convert._ucisd_state method 
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L476.]

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
        solver_cisd (object): PySCF UCISD Solver object (unrestricted)
        state (int): which state to do the conversion for, if within CISD multiple states were solved for (i.e. `nroots > 1`) -- default is 0 (the lowest-energy state)
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        wf (WFN): wavefunction in WFN format
    """

    # Get active space dimensions from CISD solver (accounts for frozen orbitals)
    # For UCISD, nmo and nocc can be tuples (nmo_a, nmo_b) and (nocc_a, nocc_b)
    nmo = solver_cisd.nmo
    nocc = solver_cisd.nocc
    if isinstance(nmo, tuple):
        norb = nmo[0]  # Assume same for alpha and beta
        nelec_a, nelec_b = nocc
    else:
        norb = nmo
        nelec_a = nocc
        nelec_b = nocc

    nvir_a, nvir_b = norb - nelec_a, norb - nelec_b

    size_a, size_b = nelec_a * nvir_a, nelec_b * nvir_b
    size_aa = int(nelec_a * (nelec_a - 1) / 2) * int(nvir_a * (nvir_a - 1) / 2)
    size_bb = int(nelec_b * (nelec_b - 1) / 2) * int(nvir_b * (nvir_b - 1) / 2)
    size_ab = nelec_a * nelec_b * nvir_a * nvir_b

    # extract the CI coeffs from the right state
    if not (state in range(solver_cisd.nroots)):
        raise IndexError("State requested has not been solved for. Re-run CISD with larger nroots.")
    if solver_cisd.nroots > 1:
        cisdvec = solver_cisd.ci[state]
    else:
        cisdvec = solver_cisd.ci 

    sizes = [1, size_a, size_b, size_aa, size_ab, size_bb]
    cumul = np.cumsum(sizes)
    idxs = [0] + [slice(cumul[ii], cumul[ii + 1]) for ii in range(len(cumul) - 1)]
    c0, c1a, c1b, c2aa, c2ab, c2bb = [cisdvec[idx] for idx in idxs]

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    # Build arrays directly instead of dict
    alpha_list = [ref_a]
    beta_list = [ref_b]
    coeff_list = [c0]

    # alpha -> alpha excitations
    c1a_configs, c1a_signs = excited_configurations(nelec_a, norb, 1)
    alpha_list.extend(c1a_configs)
    beta_list.extend([ref_b] * size_a)
    coeff_list.extend(c1a * c1a_signs)

    # beta -> beta excitations
    c1b_configs, c1b_signs = excited_configurations(nelec_b, norb, 1)
    alpha_list.extend([ref_a] * size_b)
    beta_list.extend(c1b_configs)
    coeff_list.extend(c1b * c1b_signs)

    # alpha, alpha -> alpha, alpha excitations
    c2aa_configs, c2aa_signs = excited_configurations(nelec_a, norb, 2)
    alpha_list.extend(c2aa_configs)
    beta_list.extend([ref_b] * size_aa)
    coeff_list.extend(c2aa * c2aa_signs)

    # alpha, beta -> alpha, beta excitations
    rowvals, colvals = np.array(list(product(c1a_configs, c1b_configs)), dtype=int).T
    alpha_list.extend(rowvals)
    beta_list.extend(colvals)
    coeff_list.extend(c2ab * np.kron(c1a_signs, c1b_signs))

    # beta, beta -> beta, beta excitations
    c2bb_configs, c2bb_signs = excited_configurations(nelec_b, norb, 2)
    alpha_list.extend([ref_a] * size_bb)
    beta_list.extend(c2bb_configs)
    coeff_list.extend(c2bb * c2bb_signs)

    alpha = np.array(alpha_list, dtype=np.uint64)
    beta = np.array(beta_list, dtype=np.uint64)
    coeff = np.array(coeff_list, dtype=np.float64)

    return WFN.from_arrays(alpha, beta, coeff, n_orb=norb, tol=tol)