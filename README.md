# RBDP-IMVC

Official implementation accompanying:

**Risk-Balanced Bidirectional Prediction for Incomplete Multi-View Clustering
under Missingness-Mechanism Shift**

Repository placeholder:
[github.com/USERNAME/RBDP-IMVC](https://github.com/USERNAME/RBDP-IMVC)

RBDP-IMVC learns incomplete multi-view representations when the combinations
of available views can change between training and deployment. Instead of
reducing incompleteness to one missing-rate scalar, it represents each
prediction environment by a missing target view and the set of source views
that remain observable. The implementation includes model training,
controlled mask generation, checkpoint evaluation, paired aggregation,
reliability analysis, and unit tests.

## Method at a glance

For each view, an autoencoder maps observed features to a common latent
dimension. Every directed source-target pair has a predictor that returns a
latent mean and a scalar log variance. Training combines:

1. reconstruction on every observed view;
2. mutual-information consistency for co-observed view pairs;
3. heteroscedastic directed prediction loss;
4. reverse-cycle consistency;
5. scale-normalized cosine-Gram geometry preservation; and
6. exponentiated-gradient environment reweighting.

For a path from source `s` to missing target `t`, inference uses predictive
variance plus a reverse-cycle residual as a target-free risk estimate. When
several sources are available, their predictions can be fused uniformly or
weighted by this estimated path risk. Labels are never passed to the training
routine; they are accessed only after optimization for clustering metrics and
diagnostic evaluation.

The central tensor conventions are:

| Object | Shape | Meaning |
| --- | --- | --- |
| `views[v]` | `[N, D_v]` | Features for view `v` |
| `mask` | `[N, V]` | Binary view availability |
| encoded view | `[N, d]` | View-specific latent representation |
| predictor mean | `[N, d]` | Directed latent completion |
| predictor log variance | `[N, 1]` | Path uncertainty |
| completed representation | `[N, V*d]` | Concatenated latent view blocks |
| path diagnostics | `[N, V, V]` | Source-target risk, variance, cycle, and weight |

## Repository layout

    RBDP-IMVC/
    ├── rbdp/                 Core data, mask, model, loss, training, and metric modules
    ├── configs/
    │   ├── datasets/         Dataset schemas
    │   ├── experiments/      Split, mask, architecture, and optimization settings
    │   ├── ablations/        Baselines and component studies
    │   └── grids/            Expected paired-run grids
    ├── tools/                Training, evaluation, aggregation, and audit CLIs
    ├── scripts/              Reproducible shell entry points
    ├── tests/                Unit and integration tests
    └── data/                 Three quantitative benchmark feature files

## Installation

The reference environment uses Python 3.7.16. The exact package versions are
recorded in `requirements.txt`.

    conda create -n rbdp-imvc python=3.7 -y
    conda activate rbdp-imvc
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

The code runs on CPU. Set `runtime.device` or pass `--device cuda:0` to use
a compatible CUDA-enabled PyTorch installation. PyTorch and CUDA wheel
selection is platform-specific; if the pinned PyTorch wheel is unavailable for
your accelerator, install the matching PyTorch build first and then install
the remaining requirements.

## Datasets

Three datasets required by the paired quantitative study are already included:

| Dataset | Instances | Classes | Selected view indices | Selected dimensions |
| --- | ---: | ---: | --- | --- |
| Caltech101-20 | 2,386 | 20 | `[3, 4, 5]` | `[1984, 512, 928]` |
| Scene-15 | 4,485 | 15 | `[0, 1, 2]` | `[20, 59, 40]` |
| LandUse-21 | 2,100 | 21 | `[1, 2, 0]` | `[59, 40, 20]` |

The upstream source is the
[COMPLETER data directory](https://github.com/Lin-Yijie/2021-CVPR-Completer/tree/main/data).
The same three files are also used by the
[DCP reference implementation](https://github.com/XLearning-SCU/2022-TPAMI-DCP).

NoisyMNIST is optional and is not copied into this release. To run an
additional NoisyMNIST experiment, download
[`NoisyMNIST.mat`](https://drive.google.com/file/d/1b__tkQMHRrYtcCNi_LxnVVTwB-TWdj93/view?usp=sharing)
from the link already provided by the original project and place it at
`data/NoisyMNIST.mat`. The loader uses the tune and test partitions, giving
20,000 instances with two 784-dimensional views.

## Verify the installation

Run the unit and integration tests:

    python -B -m unittest discover -s tests -v

Audit deterministic loading, label-free splits, and all four controlled mask
mechanisms for the three included datasets:

    python -B tools/gate0_audit.py

The convenience entry point performs both checks:

    bash scripts/run_gate0.sh

To use a non-default Python executable with any shell entry point, set
`RBDP_PYTHON`, for example:

    RBDP_PYTHON=/path/to/python bash scripts/run_gate0.sh

## Quick start

The following command runs the final training objective for two epochs on
Caltech101-20 using CPU:

    python -B tools/run_rbdp.py --config configs/experiments/pilot_caltech_shift.json --ablation configs/ablations/a4_geometry_balanced.json --output-root outputs --epochs 2 --training-seed 10 --device cpu --run-tag smoke

The result is stored under:

    outputs/pilot_caltech_shift__a4_geometry_balanced__smoke/Caltech101-20/view_skew/r0.9/seed10/

Each completed run contains:

| File | Contents |
| --- | --- |
| `config.json` | Fully resolved configuration, hashes, software environment, and data summary |
| `status.json` | Atomic running/completed/failed state |
| `split.npz` | Deterministic train/validation/test indices |
| `*_mask.npz` | Binary masks with mechanism metadata and hash |
| `curves.csv` | Per-epoch objectives and optimization diagnostics |
| `environment_risks.csv` | Environment losses and learned GroupDRO weights |
| `embedding.npz` | Completed embedding, clusters, masks, and path tensors |
| `reliability.csv` | Path-level risk and true completion error for evaluation |
| `risk_coverage.csv` | Selective-completion coverage curves |
| `metrics.json` | Clustering, pattern-robustness, and reliability metrics |
| `checkpoint.pt` | Model, optimizer, configuration hash, and GroupDRO state |

A completed directory is reused only when its resolved configuration hash
matches exactly.

## Reproduce the paired quantitative study

The manuscript comparison uses three datasets, two methods, and five training
seeds. All paired cells share the same split and masks.

    bash scripts/run_gate1_pilot.sh outputs

This launches 30 runs at 50 epochs each and then writes:

    outputs/gate1_pilot_summary/results.csv
    outputs/gate1_pilot_summary/summary.csv
    outputs/gate1_pilot_summary/paired_comparisons.csv
    outputs/gate1_pilot_summary/audit.json

The main comparison is:

- `a1_masked_dcp.json`: deterministic Masked-DCP baseline;
- `a4_geometry_balanced.json`: heteroscedastic bidirectional prediction,
  cycle consistency, scale-normalized relational geometry, and risk balancing.

The principal intermediate ablations are:

| Configuration | Components |
| --- | --- |
| `a0_dcp.json` | Complete-sample DCP-compatible control |
| `a1_masked_dcp.json` | Masked deterministic baseline |
| `a2_mean_risk.json` | Heteroscedastic prediction and cycle consistency |
| `a3_risk_balanced.json` | A2 plus environment risk balancing |
| `a4_geometry_balanced.json` | A3 plus normalized cosine-Gram geometry |
| `a5_adaptive_reliability.json` | A4 checkpoint with adaptive inference-time source weighting |
| `a6_conflict_safe.json` | Optional primary-anchored conflict-safe optimization |

Run the staged ablation or GroupDRO-rate sweep with:

    bash scripts/run_ablation.sh
    bash scripts/run_eta_sweep.sh

## Re-evaluate a checkpoint

`tools/evaluate_checkpoint.py` changes only the documented evaluation
surface and rejects overlays that alter the trained model signature. For
example:

    python -B tools/evaluate_checkpoint.py --run-dir outputs/pilot_caltech_shift__a4_geometry_balanced__gate1_paired/Caltech101-20/view_skew/r0.9/seed10 --evaluation-ablation configs/ablations/a5_adaptive_reliability.json --output-root outputs/adaptive --device cpu --run-tag adaptive

To evaluate all paired A4 checkpoints with adaptive reliability:

    bash scripts/run_gate2_pilot.sh outputs outputs/gate2_adaptive

## Aggregate custom runs

One or more output roots can be audited and summarized with:

    python -B tools/collect_results.py outputs --output-dir outputs/summary --baseline a1_masked_dcp

Add `--expected-grid configs/grids/gate1_pilot.json --strict` to fail if a
paired cell is missing or invalid. Aggregation never silently substitutes
failed runs.

## Reproducibility notes

- Split generation accepts no labels and is deterministic for a fixed seed.
- Missingness mechanisms include balanced MCAR, view-skewed, correlated, and
  feature-dependent masks.
- Hidden features are zeroed before model input.
- Training uses no class labels.
- Clustering labels are aligned once with a global Hungarian mapping.
- Completion errors use hidden complete benchmark features only after
  training, as an evaluation diagnostic.
- Dataset, split, mask, configuration, source, and checkpoint hashes are
  recorded in run artifacts.
- The paired grid in `configs/grids/gate1_pilot.json` defines the exact cells
  expected by strict aggregation.

## License and acknowledgement

This release retains the MIT license from the upstream XLearning codebase.
RBDP-IMVC builds on the public
[COMPLETER](https://github.com/Lin-Yijie/2021-CVPR-Completer) and
[DCP](https://github.com/XLearning-SCU/2022-TPAMI-DCP) implementations. Please
cite the accompanying RBDP-IMVC manuscript and the relevant dataset and
baseline papers when using this repository. Final bibliographic metadata will
replace this note after publication.
