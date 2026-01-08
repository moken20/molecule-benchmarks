import os
import json
from dataclasses import dataclass

from molbench.wfn import WFN, to_jw_bitstring 

@dataclass
class MolResult:
  algorithm: str
  energy: float
  spin_squared: float
  spin_projection: float
  wfn: WFN

  def print_summary(self, ndets: int = 10) -> None:
    print(f"Energy: {self.energy}")
    print(f"Spin squared (S^2): {self.spin_squared}")
    print(f"Spin projection (S_z): {self.spin_projection}")
    print(f"Number of determinants: {self.wfn.n_det}")
    if ndets > 0:
      print(f"Top {ndets} determinants:")
      self.wfn.print_topdets(ndets)
  
  def save(self, directory: str, save_top_n_dets: int = 10) -> None:
    top_n_dets = self.wfn.limit_dets(save_top_n_dets, warn=False).to_dict()
    jw_transoformed_dict = {
      to_jw_bitstring(det[0], det[1], self.wfn.n_orb): float(coeff) for det, coeff in top_n_dets.items()
    }
    summary = {
      "energy": float(self.energy),
      "spin_squared": float(self.spin_squared),
      "spin_projection": float(self.spin_projection),
      "n_det": int(self.wfn.n_det),
      f"top_{save_top_n_dets}_determinants": jw_transoformed_dict
    }
    with open(os.path.join(directory, f"{self.algorithm}.json"), "w") as f:
      json.dump(summary, f)