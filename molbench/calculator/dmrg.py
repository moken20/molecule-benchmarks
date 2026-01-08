import os
import tempfile
import uuid

import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes


from molbench.utils.chem_tools import get_mol_attrs, verify_active_space, sitevec_to_fock
from molbench.wfn import WFN


def _resolve_workdir(workdir: str | None, isolate_workdir: bool) -> str:
    """Resolve Block2 scratch directory.

    Notes:
        - Reusing the same scratch directory across many independent DMRG runs (e.g., bond-length sweeps)
          can lead to interference from leftover checkpoint/temporary files and make results look unstable.
        - When `isolate_workdir=True`, we create a unique subdirectory under `workdir`.
    """
    if workdir is None:
        return tempfile.mkdtemp(prefix="dmrg_calc_")

    base = os.path.abspath(workdir)
    if isolate_workdir:
        run_id = uuid.uuid4().hex[:10]
        base = os.path.join(base, f"run_{run_id}")
    os.makedirs(base, exist_ok=True)
    return base


def solve_dmrg(
    solver_hf,
    ncas,
    nelecas,
    schedule,
    max_mem=512,
    dot=2,
    workdir="/tmp/dmrg_calc_tmp",
    isolate_workdir=True,
    nroots=1,
    n_threads=1,
    tol=1e-6,
    restart_ket=None,
    smp_tol=1e-6,
    eshift=0,
    verbose=0,
    occs=None,
    reorder=None,
    return_objects=False,
    mpssym="sz",
    proj_state=None,
    proj_weight=None,
):
    r"""Execute the density-matrix renormalization group (DMRG) solver
    from the Block2 library, wrapping Block2's Python bindings.

    The DMRG method in Overlapper is special. DMRG is also used for the calculation of 
    Hamiltonian moments with respect to a given wavefunction, and for running the 
    resolvent method to obtain the state's energy distribution. For this reason, `do_dmrg()`
    can be run in two modes: a) normal mode to generate an initial state, and b) assessment
    mode (triggered with return_objects=True) where the DMRGDriver, MPS and MPO are returned 
    for further operations (such as computing moments using DMRG or executing the resolvent 
    method).

    Args:
        solver_hf (object): PySCF's Hartree-Fock Solver object.
        ncas (int): Number of active orbitals
        nelecas (tuple(int, int)): Number of active electrons in spin-up (alpha) and
            spin-down (beta) sectors. The `spin` variable of mf.mol is used to override the
            spin-up vs spin-down split if they give an incorrect spin.
        schedule (list(list(int), list(int), list(float), list(float))): Schedule of DMRG calculations: the first array is a list of bond dimensions; the second is a list of the number of sweeps to execute at each corresponding bond dimension; the third is a list of noises to be added to the calculation at each bond dimension; the fourth is a list of tolerances for the Davidson iteration in the sweeps.
        max_mem (int): Total memory allocated to the solver (MB).
        dot (int): Type of MPS to execute the calculation for: could be 1 or 2.
        workdir (path): Path to scratch folder for use during the calculation.
        isolate_workdir (bool): If True, create an isolated subdirectory under `workdir` per call.
        nroots (int): Number of low-energy eigenstates to solve for.
        n_threads (int): Number of threads to use for multiprocessing of the algorithm within a single shared-memory node.
        tol (float): Convergence tolerance criterion for the energy.
        restart_ket (str): If it is desired to restart a previous DMRG calculation, one can specify the tag of the corresponding MPS to restart from.
        eshift (float): Value to shift the constant of the energy. Can be used to adjust against nuclear energy during Hamiltonian moment calculation.
        smp_tol (float): Tolerance for reconstructing the MPS into a list of Slater determinants: all determinants with coefficients below this value will be neglected.
        verbose (int): Integer specifying the verbosity level.
        occs (list(int)): An initial guess for the list of occupancies of the orbitals: entires can be 0 and 1 if specified in terms of spin-orbitals (alpha and beta orbitals alternate), or 0, 1, 2, 3 if specified in terms of spatial orbitals (1 is spin-up, 2 is spin-down, 3 is double occupancy).
        reorder (None or boolean): Specifies whether reordering of the orbitals will be done before DMRG: if True, reordering is done according to the fiedler approach.
        return_objects (boolean): Whether to return the results of the calculation, or the Block2 objects (DriverDMRG, MPS and MPO).
        mpssym (str): Whether to run DMRG in SU(2) symmetry mode ("su2") or SZ symmetry mode ("sz").
        proj_state (list(MPS)): Advanced users only -- an MPS to project against during DMRG. Can be used to stabilize particular spin states
        proj_weight (list(float)): Advanced users only -- weights for the projection.

    Returns:
        wfs (tuple(list[int],array[float]) of list of such tuples): Tuples containing as the first element all the Slater determinants of a given wavefunction, and as the second the corresponding coefficients.
        e (array): Energies of `nroots` lowest energy eigenstates.
        ss (array): Spin-squared ($S^2$) of `nroots` lowest energy eigenstates.
        sz (array): Spin projection ($S_z$) of `nroots` lowest energy eigenstates."""

    verify_active_space(solver_hf.mol, ncas, nelecas)
    norb, nelec_a, nelec_b = get_mol_attrs(solver_hf.mol)
    ncore = (nelec_a + nelec_b - nelecas[0] - nelecas[1]) // 2

    # Robust HF-type detection (RHF/UHF/ROHF) for selecting integral builder
    hftype = solver_hf.__class__.__name__.lower()

    if mpssym == "sz":
        SpinSym = SymmetryTypes.SZ
    elif mpssym == "su2":
        if not ("rhf" in hftype.lower()):
            raise ValueError("SU2 MPS calculation only possible with RHF molecular integrals.")
        SpinSym = SymmetryTypes.SU2
    else:
        raise ValueError(f"Unknown mpssym='{mpssym}'. Use 'sz' or 'su2'.")

    if "rhf" in hftype.lower():
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
            solver_hf, ncore, ncas, g2e_symm=8
        )
    elif "uhf" in hftype.lower() or "rohf" in hftype.lower():
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_uhf_integrals(
            solver_hf, ncore, ncas, g2e_symm=8
        )
    else:
        raise ValueError(f"Unsupported HF solver type: {solver_hf.__class__.__name__}")

    # If user wants to restart from a previous MPS tag, they typically need the same scratch directory.
    if restart_ket is not None:
        isolate_workdir = False
    workdir = _resolve_workdir(workdir, isolate_workdir=isolate_workdir)

    max_mem = max_mem * 1024 * 1024 # Convert MB to bytes
    driver = DMRGDriver(scratch=workdir, symm_type=SpinSym, n_threads=n_threads, stack_mem=max_mem)

    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore+eshift, reorder=reorder, iprint=verbose)

    # Need to reorder the occupations provided (must happen BEFORE creating the initial MPS).
    # Note: reorder indices are set by Block2 when building the QC MPO with `reorder=True`.
    if (occs is not None) and (reorder is not None):
        if len(occs) == ncas:
            occs = np.array(occs)[driver.reorder_idx]
        elif len(occs) == 2 * ncas:
            occs_a = (np.array(occs)[0 : len(occs) : 2])[driver.reorder_idx]
            occs_b = (np.array(occs)[1 : len(occs) : 2])[driver.reorder_idx]
            occs = []
            for ii in range(len(occs_a)):
                occs.append(occs_a[ii])
                occs.append(occs_b[ii])
        else:
            raise ValueError(
                f"Invalid occs length={len(occs)}. Expected ncas={ncas} (spatial) or 2*ncas={2*ncas} (spin-orbital)."
            )

    if restart_ket is None:
        ket = driver.get_random_mps(
            tag="GS", bond_dim=schedule[0][0], occs=occs, nroots=nroots, dot=dot
        )
    else:
        ket = driver.load_mps(restart_ket)

    bond_dims, n_sweeps, noises, thresholds = schedule
    for ii, M in enumerate(bond_dims):
        Mvals = [M] * n_sweeps[ii]
        noisevals = [noises[ii]] * n_sweeps[ii]
        thrdsvals = [thresholds[ii]] * n_sweeps[ii]
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=n_sweeps[ii],
            bond_dims=Mvals,
            noises=noisevals,
            thrds=thrdsvals,
            iprint=verbose,
            tol=tol,
            proj_mpss=proj_state,
            proj_weights=proj_weight,
        )

    wfs = []
    e = []
    ss = []
    sz = []
    smpo = driver.get_spin_square_mpo(iprint=verbose)

    if nroots > 1:
        for ii in range(nroots):
            aux_ket = driver.split_mps(ket, ii, f"state{ii}")
            aux_ket_e = driver.expectation(aux_ket, mpo, aux_ket)
            e.append(aux_ket_e)
            # this part reconstructs the Slater determinants from the GS MPS
            dets, coeffs = driver.get_csf_coefficients(aux_ket, cutoff=smp_tol, iprint=verbose)
            # re-attach the frozen core electrons
            dets = [[3] * ncore + det.tolist() for det in dets]
            wfs.append((dets, coeffs))

            ### compute the S^2 spin number
            ss.append(driver.expectation(aux_ket, smpo, aux_ket))
            sz.append(solver_hf.mol.spin)
    else:
        ket_e = driver.expectation(ket, mpo, ket)
        e.append(ket_e)
        # this part reconstructs the Slater determinants from the GS MPS
        dets, coeffs = driver.get_csf_coefficients(ket, cutoff=smp_tol, iprint=verbose)
        # re-attach the frozen core electrons
        dets = [[3] * ncore + det.tolist() for det in dets]
        wfs.append((dets, coeffs))

        ### compute the S^2 spin number
        ss.append(driver.expectation(ket, smpo, ket))
        sz.append(solver_hf.mol.spin)

    if return_objects:
        return mpo, ket, driver

    return wfs, np.atleast_1d(e), np.array(ss), np.array(sz)

def extract_dmrg_state(solver_dmrg, state=0, tol=1e-15, n_orb=None):
    r"""Construct a WFN object from the DMRG wavefunction obtained from the Block2 library.
    [This function is copied on PennyLane's qchem.convert._dmrg_state method
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/qchem/convert.py#L1023.]

    The generated wavefunction is stored in a WFN object where alpha and beta bitstring integers
    represent configurations (Slater determinants) and coefficients are the CI coefficients.
    The binary representation of these integers correspond to a specific configuration: 
    the first number represents the configuration of the alpha electrons and the second 
    number represents the configuration of the beta electrons. For instance, the Hartree-Fock 
    state :math:`|1 1 0 0 \rangle` will be represented by the flipped binary string ``0011`` 
    which is split to ``01`` and ``01`` for the alpha and beta electrons. The integer 
    corresponding to ``01`` is ``1`` and the WFN representation of the Hartree-Fock state 
    will have alpha=[1], beta=[1], coeff=[1.0].

    The determinants and coefficients are supplied externally. They are calculated with Block2 
    DMRGDriver's `get_csf_coefficients()` method and passed as a tuple in the first argument. 
    If the DMRG calculation was executed in SZ mode, the wavefunction is built in terms of Slater
    determinants (eigenfunctions of the :math:`S_z` operator); if it was in SU(2) mode, the 
    wavefunction is automatically built out of configuration state functions (CSF -- 
    eigenfunctions of the :math:`S^2` operator).

    Args:
        solver_dmrg: tuple(list[int], array[float]): determinants and coefficients in physicist notation, as output by Block2 DMRGDriver's `get_csf_coefficients()` methods
        state (int): which state to extract (default 0)
        tol (float): the tolerance for discarding Slater determinants with small coefficients
        n_orb (int, optional): number of orbitals for WFN storage optimization

    Returns:
        wf (WFN): wavefunction in WFN format
    """

    dets, coeffs = solver_dmrg[state]

    alpha_list = []
    beta_list = []
    coeff_list = []
    
    for ii, det in enumerate(dets):
        stra, strb = sitevec_to_fock(det, format="dmrg")

        # compute and fix parity to stick to pyscf notation
        lsta = np.array(list(map(int, bin(stra)[2:])))[::-1]
        lstb = np.array(list(map(int, bin(strb)[2:])))[::-1]

        # pad the shorter list
        maxlen = max([len(lsta), len(lstb)])
        lsta = np.pad(lsta, (0, maxlen - len(lsta)))
        lstb = np.pad(lstb, (0, maxlen - len(lstb)))

        which_occ = np.where(lsta == 1)[0]
        parity = (-1) ** np.sum([np.sum(lstb[: int(ind)]) for ind in which_occ])
        
        alpha_list.append(stra)
        beta_list.append(strb)
        coeff_list.append(parity * coeffs[ii])

    alpha = np.array(alpha_list, dtype=np.uint64)
    beta = np.array(beta_list, dtype=np.uint64)
    coeff = np.array(coeff_list, dtype=np.float64)

    return WFN.from_arrays(alpha, beta, coeff, n_orb=n_orb, tol=tol)