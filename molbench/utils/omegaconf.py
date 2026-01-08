from pathlib import Path
from typing import Any
from omegaconf import DictConfig, OmegaConf


def register_resolvers(*, replace: bool = False) -> None:
    """Register custom OmegaConf resolvers used in config files.

    This is safe to call multiple times.
    """
    resolvers = {
        "mul": lambda a, b: float(a) * float(b),
        "add": lambda a, b: float(a) + float(b),
        "sub": lambda a, b: float(a) - float(b),
    }
    for name, fn in resolvers.items():
        if OmegaConf.has_resolver(name):
            if replace:
                OmegaConf.register_new_resolver(name, fn, replace=True)
            continue
        OmegaConf.register_new_resolver(name, fn)


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up until `config/molecule` is found."""
    p = (start or Path.cwd()).resolve()
    for cand in [p] + list(p.parents):
        if (cand / "config").is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find `config/molecule` by walking up from: {p}. "
        "Run this inside the repository, or pass a suitable `start` path."
    )


def get_molecule_config_path(molecule_name: str, start: Path | None = None) -> Path:
    """Return the config path `config/molecule/<molecule_name>.yaml`.

    Notes:
        - `molecule_name` is assumed to be a bare molecule name (e.g., `"H2"`), not a file path.
        - Matching is case-insensitive against available `*.yaml` stems under `config/molecule`.
    """
    if not isinstance(molecule_name, str) or not molecule_name.strip():
        raise ValueError("molecule_name must be a non-empty string (e.g., 'H2').")

    name = molecule_name.strip()
    root = find_repo_root(start=start)
    mol_dir = root / "config" / "molecule"
    files = sorted(mol_dir.glob("*.yaml"))
    name_map = {p.stem.lower(): p for p in files}
    hit = name_map.get(name.lower())
    if hit is None:
        available = ", ".join(sorted({p.stem for p in files}))
        raise FileNotFoundError(
            f"Molecule config not found for: {name!r} (searched in: {mol_dir}). "
            f"Available: [{available}]"
        )
    return hit.resolve()


def load_molecule_cfg(
    molecule_name: str,
    start: Path | None = None,
    overrides: dict[str, Any] | list[str] | tuple[str, ...] | DictConfig | None = None,
) -> DictConfig:
    """Load molecule config as a root config `{"molecule": ...}`.

    Args:
        molecule_name: Bare molecule name (e.g., "H2").
        start: Optional starting path used to locate the repository root.
        overrides: Optional overrides to apply before resolving and returning.
            - If a mapping (e.g., ``{"distance": 2.0}``), it is applied under ``molecule``.
              If it already contains a top-level ``molecule`` key (e.g., ``{"molecule": {...}}``),
              it is merged as-is.
            - If a sequence of strings, it is treated as an OmegaConf dotlist. Entries that do not
              start with ``molecule.`` are automatically prefixed (e.g., ``"distance=2.0"`` becomes
              ``"molecule.distance=2.0"``).

    Returns:
        A resolved DictConfig with a top-level key `molecule`.
    """
    register_resolvers()
    molecule_cfg_path = get_molecule_config_path(molecule_name=molecule_name, start=start)
    mol_cfg = OmegaConf.load(molecule_cfg_path)
    cfg = OmegaConf.create({"molecule": mol_cfg})

    if overrides:
        if isinstance(overrides, DictConfig):
            override_cfg = overrides
        elif isinstance(overrides, dict):
            override_cfg = (
                OmegaConf.create(overrides)
                if "molecule" in overrides
                else OmegaConf.create({"molecule": overrides})
            )
        else:
            # dotlist (sequence[str]) — assume keys are within `molecule` unless prefixed
            fixed = []
            for item in overrides:
                s = str(item)
                if "=" in s:
                    k, v = s.split("=", 1)
                    k_strip = k.strip()
                    if not (k_strip == "molecule" or k_strip.startswith("molecule.")):
                        s = f"molecule.{k_strip}={v}"
                fixed.append(s)
            override_cfg = OmegaConf.from_dotlist(fixed)

        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.resolve(cfg)
    return cfg
