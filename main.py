from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from molbench.orchestrator import Orchestrator
from molbench.schema import MolResult
from molbench.utils.omegaconf import register_resolvers


def _print_result(label: str, result: MolResult) -> None:
    print(f"[RESULT] {label}:")
    result.print_summary(ndets=0)
    print("-" * 36)


def print_results(results: Any, label: str) -> None:
    if isinstance(results, MolResult):
        _print_result(label=label, result=results)
        return

    for i, res in enumerate(results):
        if not isinstance(res, MolResult):
            raise TypeError(f"Unexpected result type: {type(res)}")
        res.algorithm = f"{res.algorithm}_root{i}"
        _print_result(label=f"{label} / root={i}", result=res)


@hydra.main(version_base=None, config_path="./config", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    `config/default.yaml` を読み込み (Hydra で config group 合成/補間も解決)、
    `molbench/orchestrator.py` の `do_*` を実行して、
    `molbench/schema.py` の `MolResult.save()` で `cfg.output` 配下へ保存する。

    実行例:
      - python -m molbench.main
      - python -m molbench.main molecule=H2 algorithms.dmrg.run_flag=false
      - python -m molbench.main output=outputs/test nroots=3 ci_cutoff=1e-7
    """
    register_resolvers()
    mol = cfg.molecule
    molecule_name = HydraConfig.get().runtime.choices.get("molecule", "<unknown>")
    algorithms = cfg.algorithms
    executor = Orchestrator(
        geometry=mol.geometry,
        basis=mol.basis,
        nelecas=int(mol.nelecas),
        norbcas=int(mol.norbcas),
        spin=int(mol.spin),
        charge=int(mol.charge),
    )

    # HF
    hf_cfg = algorithms.hf
    res_hf = executor.do_hf(verbose=hf_cfg.verbose, ci_cutoff=cfg.ci_cutoff)
    print_results(res_hf, label=f"hf (molecule={molecule_name})")

    # CISD
    if algorithms.cisd.run_flag:
        cisd_cfg = algorithms.cisd
        res = executor.do_cisd(
            nroots=cfg.nroots,
            verbose=cisd_cfg.verbose,
            ci_cutoff=cfg.ci_cutoff,
        )
        print_results(res, label=f"cisd (molecule={molecule_name})")

    # CCSD
    if algorithms.ccsd.run_flag:
        ccsd_cfg = algorithms.ccsd
        res = executor.do_ccsd(
            verbose=ccsd_cfg.verbose,
            ci_cutoff=cfg.ci_cutoff,
            max_cycle=ccsd_cfg.max_cycle,
        )
        print_results(res, label=f"ccsd (molecule={molecule_name})")

    # CASCI
    if algorithms.casci.run_flag:
        casci_cfg = algorithms.casci
        res = executor.do_casci(
            nroots=cfg.nroots,
            verbose=casci_cfg.verbose,
            ci_cutoff=cfg.ci_cutoff,
            maxiter=casci_cfg.maxiter,
        )
        print_results(res, label=f"casci (molecule={molecule_name})")

    # DMRG
    if algorithms.dmrg.run_flag:
        dmrg_cfg = algorithms.dmrg
        kwargs = dict(OmegaConf.to_container(dmrg_cfg, resolve=True) or {})
        kwargs.pop("run_flag", None)
        kwargs.pop("schedule", None)
        res = executor.do_dmrg(
            schedule=dmrg_cfg.schedule,
            nroots=cfg.nroots,
            ci_cutoff=cfg.ci_cutoff,
            **kwargs,
        )
        print_results(res, label=f"dmrg (molecule={molecule_name})")


if __name__ == "__main__":
    main()

