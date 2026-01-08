# molecule-benchmarks

PySCF / Block2 を用いて、分子ごとの**古典量子化学アルゴリズム**（HF/CISD/CCSD/CASCI/DMRG）を同一インターフェースで実行し、エネルギー・スピン量などを比較するためのベンチマーク用リポジトリです。

## Overview

- **実行入口（推奨）**: リポジトリ直下の `main.py` を `python main.py ...` で実行（Hydraで設定を切替）
- **Notebook**: `tutorial/` で `Orchestrator` を直接呼び出して解析
- **主な出力**:
  - Energy
  - \(S^2\)（spin squared）
  - \(S_z\)（spin projection）
  - 波動関数（Slater determinant の係数; `molbench/wfn.py` の `WFN`）

## Quick Start（CLI: `python main.py`）

### セットアップ

```bash
git clone <this-repo>
cd molecule-benchmarks

python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
```

### 実行（デフォルト設定）

```bash
python main.py
```

### 1. 設定をかえる（`config/default.yaml` を編集）

コマンドラインのoverrideではなく、まずは `config/default.yaml` を書き換えて設定を切り替える使い方を想定しています。

#### どの分子を使うか（config group の選択）

`defaults` の `molecule:` を変更します。例えば N2 を使うなら:

```yaml
defaults:
  - molecule: N2
```

#### どのアルゴリズムを走らせるか（ON/OFF）

`algorithms.<method>.run_flag` を切り替えます（HFは常に実行されます）。

```yaml
algorithms:
  cisd:
    run_flag: true
  ccsd:
    run_flag: false
  casci:
    run_flag: true
  dmrg:
    run_flag: false
```

#### よく触るグローバル設定（root数 / cutoff）

```yaml
# 低エネルギーroot（励起状態）を何本計算するか
nroots: 2

# 波動関数（determinant係数）を取り出すときのカットオフ
ci_cutoff: 1e-6
```

#### active space（`nelecas`, `norbcas`）を変える

active space（`nelecas`, `norbcas`）は固定する前提ではなく、目的に応じて自由に変えてOKです。方法は2通りあります。

- **分子YAMLを直接編集**: `config/molecule/<NAME>.yaml` の `nelecas` / `norbcas` を変更
- **`config/default.yaml` で上書き**: 選んだ分子設定の上から `molecule.*` を追記して上書き

例（`config/default.yaml` 側で上書き）:

```yaml
molecule:
  nelecas: 10
  norbcas: 8
```

編集後はそのまま:

```bash
python main.py
```

### 2. コマンドラインのoverride（任意）

基本は `config/default.yaml` を編集して設定を切り替える運用を想定していますが、Hydraのoverrideで**一時的に**上書きして実行することもできます。

```bash
# 例: 分子だけ一時的に切り替える
python main.py molecule=N2

# 例: DMRGだけ一時的にOFFにする
python main.py algorithms.dmrg.run_flag=false
```

## Algorithms（古典アルゴリズム一覧）

`molbench/orchestrator.py` の `Orchestrator.do_*` を呼び出します。

- **HF**: `do_hf()`（基準状態の生成）
- **CISD**: `do_cisd()`（単・二重励起CI）
- **CCSD**: `do_ccsd()`（結合クラスター singles+doubles）
- **CASCI**: `do_casci()`（指定active space内のCI）
- **DMRG**: `do_dmrg()`（Block2によるMPS/DMRG）

主な設定は `config/default.yaml` の `algorithms.*` を参照してください（例: CCSDの `max_cycle`、DMRGの `schedule` / `max_mem` / `workdir` など）。

### 設定キー（CLIからよく触るもの）

| method | 実装 | 実行ON/OFF | 主な設定例 |
|---|---|---|---|
| HF | `do_hf()` | 常に実行 | `algorithms.hf.verbose` |
| CISD | `do_cisd()` | `algorithms.cisd.run_flag` | `algorithms.cisd.verbose`, `nroots`, `ci_cutoff` |
| CCSD | `do_ccsd()` | `algorithms.ccsd.run_flag` | `algorithms.ccsd.max_cycle`, `algorithms.ccsd.verbose`, `ci_cutoff` |
| CASCI | `do_casci()` | `algorithms.casci.run_flag` | `algorithms.casci.maxiter`, `nroots`, `ci_cutoff` |
| DMRG | `do_dmrg()` | `algorithms.dmrg.run_flag` | `algorithms.dmrg.schedule`, `algorithms.dmrg.max_mem`, `algorithms.dmrg.workdir`, `algorithms.dmrg.n_threads` |

### DMRG（Block2）パラメータの説明（詳しめ）

DMRGは、active space内の波動関数をMPS（Matrix Product State）として近似し、段階的に精度を上げながら収束させます。`config/default.yaml` の `algorithms.dmrg.*` を中心に調整します。

- **`schedule`**: DMRGの“段階的な収束戦略”を指定します。4本のリストを1セットにした形です（Notebookでも同形式）。
  - 1行目 **bond dimensions**: 各ステージのMPSボンド次元（大きいほど高精度・高コスト）
  - 2行目 **sweeps**: 各ステージでのスイープ回数（増やすほど収束しやすいが遅い）
  - 3行目 **noise**: 収束を助けるためのノイズ（最終ステージでは0にするのが一般的）
  - 4行目 **Davidson thresholds**: 対角化（Davidson）しきい値（小さいほど厳密だが遅い）

例（小さめのactive space向けの出発点）:

```python
schedule = [
    [10, 20, 30],          # bond dimensions
    [5, 5, 5],             # sweeps
    [1e-4, 1e-5, 0.0],     # noise
    [1e-6, 1e-8, 1e-10],   # Davidson thresholds
]
```

- **`dot`**: 1-site/2-site DMRGの選択（一般に2-siteの方が安定しやすい）
- **`max_mem`**: 使用メモリ上限（MB）
- **`n_threads`**: スレッド数（環境と計算サイズに応じて調整）
- **`workdir`**: 計算の作業ディレクトリ（中間ファイル等の置き場）
- **`tol`**: 収束判定の閾値（エネルギー収束など）
- **`restart_ket`**: リスタート用（必要な場合のみ）
- **`smp_tol`**: determinant抽出/サンプリング関連のしきい値（小さいほど厳密だが遅い）

運用の目安:
- **精度を上げたい**: `bond dimensions` を上げる → 足りなければ `sweeps` を増やす → 最終ステージの `Davidson thresholds` を下げる
- **発散/収束不良**: 2-site（`dot: 2`）にする、途中ステージの `noise` を少し上げる、`sweeps` を増やす
- **遅い/重い**: `bond dimensions` を下げる、ステージ数を減らす、`sweeps` を減らす

## Molecules（対応分子）

分子定義は `config/molecule/*.yaml` です（表はYAMLと一致するように記載）。

active space（`nelecas`, `norbcas`）は **固定する前提ではありません**。CLI/Notebookから自由にoverrideして使ってください（例: `molecule.nelecas=...` / `molecule.norbcas=...`）。

**対応分子**:
現在は以下の分子のみですが、自由に追加可能です。
```text
H2  H4  H24  LiH  CO  N2  CH4  C10H8  Cr2  Fe2S2
```

## Notebook（チュートリアル）

- `tutorial/1_calculate_energy.ipynb`
  - 分子設定を `load_molecule_cfg()` で読み、各アルゴリズムのエネルギーを計算
  - 結合長をscanしてエネルギーカーブを描画する例
- `tutorial/2_analyze_wfn.ipynb`
  - CASCIを参照（reference）にして、HF/CISD/CCSD/DMRG の波動関数overlapを比較
  - 伸長結合での多参照性の増大を確認する例
  - overlap計算は `molbench/wfn.py` の `wf_overlap()` を使用


## Troubleshooting / FAQ

- **DMRGが遅い/止まる**: `algorithms.dmrg.workdir` をGoogle Drive配下ではなくローカル（例: `/tmp/dmrg_calc_tmp`）にすると改善することがあります。
- **メモリ不足**: `algorithms.dmrg.max_mem` を下げる/上げる、`schedule`（bond dimension）を小さくする、`n_threads` を調整してください。
- **MPI関連のimportエラー**: `mpi4py` は環境依存です。MPI実装（OpenMPI等）が必要な場合があります。

## Roadmap

- **HCIアルゴリズムの搭載**
- **分子の増大（`config/molecule/*.yaml` の追加）**
- **CLI整備**:
  - `pyproject.toml` の `project.scripts` に `molbench = "molbench.main:main"` が定義されていますが、現状は `python main.py` が実行入口です（今後整合を取る予定）。