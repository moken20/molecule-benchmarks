from dataclasses import dataclass
import numpy as np


WfnDict = dict[tuple[int, int], float]

@dataclass
class WFN:
    """
    Memory-efficient wavefunction container.

    Internally stores wavefunction in COO-like form:
      alpha[i], beta[i] (bitstring ints as uint64) and coeff[i].
    """
    alpha: np.ndarray    # uint64, shape (K,)
    beta: np.ndarray     # uint64, shape (K,)
    coeff: np.ndarray    # float64, shape (K,)
    n_orb: int | None = None

    def __post_init__(self) -> None:
        self.alpha = np.atleast_1d(np.asarray(self.alpha, dtype=np.uint64))
        self.beta = np.atleast_1d(np.asarray(self.beta, dtype=np.uint64))
        self.coeff = np.atleast_1d(np.asarray(self.coeff, dtype=np.float64))
        if not (self.alpha.shape == self.beta.shape == self.coeff.shape):
            raise ValueError("alpha, beta, coeff must have the same shape.")

    def __len__(self) -> int:
        return int(self.coeff.shape[0])

    @property
    def n_det(self) -> int:
        return len(self)
    
    @property
    def norm(self) -> float:
        return wf_overlap(self, self)

    @classmethod
    def from_arrays(
        cls,
        alpha,
        beta,
        coeff,
        n_orb: int | None = None,
        tol: float = 0.0,
    ) -> "WFN":
        """
        Construct WFN from alpha, beta, coeff arrays/lists.

        Args:
            alpha: Array-like of alpha bitstring integers.
            beta: Array-like of beta bitstring integers.
            coeff: Array-like of CI coefficients.
            n_orb: Number of orbitals (optional metadata).
            tol: Tolerance for filtering small coefficients (default 0.0 = keep all).

        Returns:
            WFN instance.
        """
        alpha = np.asarray(alpha, dtype=np.uint64)
        beta = np.asarray(beta, dtype=np.uint64)
        coeff = np.asarray(coeff, dtype=np.float64)

        if tol > 0.0:
            mask = np.abs(coeff) > tol
            alpha = alpha[mask]
            beta = beta[mask]
            coeff = coeff[mask]

        return cls(alpha=alpha, beta=beta, coeff=coeff, n_orb=n_orb)
    
    def normalize(self) -> "WFN":
        norm_squred = self.norm
        coeff = self.coeff / np.sqrt(norm_squred)
        return WFN(alpha=self.alpha, beta=self.beta, coeff=coeff, n_orb=self.n_orb)

    def limit_dets(self, N: float, warn: bool = True) -> "WFN":
        """Limit the number of determinants to N."""
        if len(self) < N:
            if warn:
                print(f"Warning: {len(self)} determinants is less than {N}. Returning original WFN.")
            return self
        else:
            sorted_indices = np.argsort(np.abs(self.coeff))[::-1]
            alpha = self.alpha[sorted_indices[:N]]
            beta = self.beta[sorted_indices[:N]]
            coeff = self.coeff[sorted_indices[:N]]
            return WFN(alpha=alpha, beta=beta, coeff=coeff, n_orb=self.n_orb)
    
    def to_dict(self, *, tol: float = 0.0) -> WfnDict:
        """
        Convert to { (alpha_int, beta_int) : coeff }.
        Warning: for large K this is memory-heavy (Python dict overhead).
        """
        out: WfnDict = {}
        for i in range(self.n_det):
            v = float(self.coeff[i])
            if tol > 0.0 and abs(v) <= tol:
                continue
            out[(int(self.alpha[i]), int(self.beta[i]))] = v
        return out
    
    def print_topdets(self, n: int = 10, endian: str = "little") -> None:
        """Print the top N determinants."""
        limited_wf = self.limit_dets(n, warn=False)
        np.printoptions(linewidth=300)

        headstr = "".join([str(ii % 10) for ii in range(1, self.n_orb+1)])
        print(f"{headstr}  |  {headstr}  |  coeff")
        print("-" * 40)
        for i in range(min(n, len(limited_wf))):
            alpha_bitstring = _int_to_bitstring(limited_wf.alpha[i], self.n_orb)
            beta_bitstring = _int_to_bitstring(limited_wf.beta[i], self.n_orb)
            print(f"{alpha_bitstring}  |  {beta_bitstring}  |  {limited_wf.coeff[i]:.6f}")

def _int_to_bitstring(x: int, n_orb: int) -> str:
    """Convert an integer to a bitstring."""
    return "".join([str((x >> i) & 1) for i in range(n_orb)])

def to_jw_bitstring(alpha: int, beta: int, n_orb: int, endian: str = "little") -> str:
    """
    Convert a pair of alpha and beta bitstrings to a single Jordan-Wigner
    transformed bitstring.

    Args:
        alpha: Alpha spin occupation bitstring (uint64 integer).
        beta: Beta spin occupation bitstring (uint64 integer).
        n_orb: Number of spatial orbitals.
        endian: Bit ordering.
            "big": MSB first (highest orbital index at leftmost).
            "little": LSB first (lowest orbital index at leftmost).

    Returns:
        String of '0' and '1' characters representing the combined JW bitstring
        (length = 2 * n_orb).

    Example:
        >>> WFN.to_jw_bitstring(alpha=0b011, beta=0b101, n_orb=3, endian="big")
        '011110'  # orbital2: α=0,β=1 | orbital1: α=1,β=0 | orbital0: α=1,β=1
        >>> WFN.to_jw_bitstring(alpha=0b011, beta=0b101, n_orb=3, endian="little")
        '111001'  # orbital0: α=1,β=1 | orbital1: α=1,β=0 | orbital2: α=0,β=1
    """
    if endian not in ("big", "little"):
        raise ValueError(f"endian must be 'big' or 'little', got {endian!r}")

    bits = []
    for i in range(n_orb):
        bits.append(_int_to_bitstring(alpha >> i, 1))
        bits.append(_int_to_bitstring(beta >> i, 1))

    if endian == "big":
        return ''.join(reversed(bits))
    else:
        return ''.join(bits)

def wf_overlap(wf1: WFN, wf2: WFN) -> float:
    """
    Calculate the overlap between two WFNs.
    """
    wf1 = wf1.to_dict()
    wf2 = wf2.to_dict()
    common_keys = list(set(wf1.keys()).intersection(wf2.keys()))
    overlap = 0.0
    for key in common_keys:
        overlap += wf1[key] * wf2[key]
    return abs(overlap)
