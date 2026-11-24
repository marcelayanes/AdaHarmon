# AdaHarmon

Reference implementation of **AdaHarmon**, an adaptive optimizer that
combines the root second moment and a softened ℓ∞ gradient state through
their harmonic mean, while modulating momentum per parameter using a
sign-consistency estimator.

This repository accompanies the manuscript:

> **Yanes-Pérez, M. A., Hernández-Nolasco, J. A., & Wister, M. A.**
> *AdaHarmon: A Harmonic-Mean Adaptive Optimizer for CNN-Based
> Classification of Hematological Microscopic Images.*
> Manuscript NEUCOM-D-26-20991, under revision at **Neurocomputing**, 2026.

**Keywords:** Adaptive optimization · Stochastic gradient methods ·
Adaptive momentum · Harmonic mean · Convolutional neural networks ·
Blood cell classification

---

## Overview

AdaHarmon integrates four mechanisms into a single Adam-type update rule:

1. **Sign-consistency estimator** `s_t` that accumulates the per-parameter
   gradient sign, yielding a stability signal in `[-1, 1]`.
2. **Adaptive momentum coefficient** `β₁⁽ᵗ⁾` driven by `|s_t|`, with a
   bias correction generalized to the variable-momentum case via the
   cumulative product `μ_t = ∏ β₁⁽ⁱ⁾`. Since `β₁⁽ᵗ⁾` is per parameter,
   `μ_t` is also a per-parameter quantity. The telescoping identity behind
   the correction is exact; the unbiasedness of the corrected moment relies
   on a local gradient-stability assumption, as in Adam.
3. **Softened ℓ∞ state** `u_t` with a plateau-detection indicator
   `δ_t` that accelerates forgetting of stale gradient peaks.
4. **Hybrid denominator** `h̃_t` built as the harmonic mean of the
   root second moment `√v̂_t` and `u_t`, with no additional balance
   hyperparameter.

**Theory.** An `O(√T)` regret bound is derived for online convex
optimization with a decreasing learning rate `α/√t`, under bounded
gradients, a box-shaped bounded domain with coordinate-wise projection, a
denominator bounded away from zero, a monotone denominator
(`|g_{t,i}| ≥ u_{t-1,i}`), and bounded variation of the momentum
coefficients (Assumption S5). The result is presented as a proof sketch.
Assumption S5 holds automatically for a fixed `β₁`, but it is not expected
to hold for the adaptive coefficient, so the theorem rigorously covers the
fixed-momentum variant and applies to the adaptive coefficient only
conditionally. The monotone-denominator assumption excludes the soft-decay
mechanism. The guarantee does not cover the full algorithm, non-convex
training, or the fixed learning rate used in the experiments. The full
derivation and proof are in Supplementary Sections S1.1 and S1.2.

---

## Installation

The experiments in the paper were run with Python 3.10 and PyTorch 2.10.0
on an NVIDIA L4 GPU. Other versions have not been tested.

```bash
git clone https://github.com/marcelayanes/adaharmon.git
cd adaharmon
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`,
`scipy`, `matplotlib`, `seaborn`, `torchinfo`, `tqdm`, plus the following
packages used by the baselines, the cost analysis and the dataset download
in the notebooks: `adabelief-pytorch` (AdaBelief), `lion-pytorch` (Lion),
`pynvml` (energy estimate), `thop` (complexity analysis), and `kaggle`
(dataset download).

---

## Quick start

```python
import torch
from src.adaharmon import AdaHarmon

model = MyCNN(num_classes=8)
optimizer = AdaHarmon(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999, 0.9),  # (beta1, beta2, beta_s)
    eps=(1e-8, 1e-6),         # (eps1, eps2)
    lambda_s=1.0,             # momentum range coefficient (lambda)
    soft_decay=0.5,           # soft-decay exponent (gamma)
    tau=0.1,                  # plateau threshold
)

for epoch in range(80):
    for batch in loader:
        x, y = batch
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

The default values (`β₁ = 0.9`, `β₂ = 0.999`, `β_s = 0.9`, `λ = 1.0`,
`γ = 0.5`, `τ = 0.1`, `ε₁ = 10⁻⁸`, `ε₂ = 10⁻⁶`) are the ones used in all
experiments in the paper, without per-dataset or per-architecture tuning.
The only externally selected hyperparameter is the learning rate
`α = 10⁻³`. The class also has an optional `amsgrad` flag (off by default
and not used in any reported experiment).

---

## Reproducing the experiments

### Protocol

- **Datasets:**
  - PBC: 64×64, 8 classes, 11,964 / 2,563 / 2,565 train / validation / test images.
  - Naturalize 2K-PBC: 96×96, 11 classes, 21,234 / 2,650 / 2,650.
  - Raabin-WBC: 128×128, 5 classes, 11,643 / 2,494 / 2,496.
- **Architectures (trained from scratch, no pretrained weights):**
  ResNet18 (11.18 M parameters), EfficientNetV2 (20.19 M), MobileNetV3
  (4.21 M). The classification head is replaced by dropout (`p = 0.5`) and
  a linear layer.
- **Optimizers (10):** AdaHarmon and nine baselines: SGD, RMSprop, Adam,
  AdamW, RAdam, AMSGrad, Adamax, AdaBelief, and Lion.
- **Seeds:** 10 fixed seeds per scenario
  (7, 42, 123, 456, 784, 999, 1337, 2026, 3407, 8888).
- **Schedule:** 80 epochs, batch size 64, cross-entropy loss, fixed
  learning rate, no scheduler.
- **Preprocessing:** identical for all optimizers within a dataset.
  Stochastic augmentation on training data (rotations, flips, random crops,
  photometric jitter); resizing and normalization only on validation and
  test data; ImageNet channel statistics. Class imbalance is handled with
  PyTorch's weighted random sampler; Naturalize 2K-PBC and Raabin-WBC
  additionally use a weighted cross-entropy loss (inverse square root of
  class frequencies).
- **Total runs:** 9 scenarios × 10 optimizers × 10 seeds = 900 runs for the
  main comparison (ablation and sensitivity runs are additional).

> **Note on the Raabin notebooks.** The data-loading cells of the three
> Raabin-WBC notebooks display an earlier 500-image subset (350/75/75)
> from a preliminary execution. The reported results use the full
> 11,643/2,494/2,496 split listed above, as reflected in the result tables
> and in the final summary cell of each notebook.

### Optimizer configurations

All baselines use the default values of their original formulations,
without individual tuning.

| Optimizer | Learning rate | Other settings |
|---|---|---|
| AdaHarmon | 1e-3 | Constants fixed as above |
| SGD | 1e-2 | momentum 0.9 (untuned) |
| RMSprop | 1e-3 | β₂ = 0.99 |
| Adam | 1e-3 | β = (0.9, 0.999) |
| AdamW | 1e-3 | β = (0.9, 0.999), weight decay 0.01 (PyTorch default) |
| RAdam | 1e-3 | β = (0.9, 0.999), automatic rectification |
| AMSGrad | 1e-3 | Adam with `amsgrad=True` |
| Adamax | 2e-3 | Value recommended by its authors |
| AdaBelief | 1e-3 | β = (0.9, 0.999), no weight decay, no rectification |
| Lion | 1e-4 | β = (0.9, 0.99), weight decay 0.1 |

Adaptive methods use `ε = 10⁻⁸`. Adamax and Lion are the two exceptions to
the common learning rate of `10⁻³`.

### Running a scenario

Open the corresponding notebook under `notebooks/` for the scenario you
want to reproduce. Each notebook is self-contained: it defines the
`AdaHarmon` class inline (identical to `src/adaharmon.py`) and trains all
ten optimizers for one (dataset, architecture) pair. The seed is set at the
top of the execution loop (`torch.manual_seed(...)`); run the notebook once
per seed.

Model checkpoints are **not** included in this repository due to size
constraints and must be regenerated by re-running the notebooks.

### Statistical analysis

Accuracy, AUC, Loss, macro-F1, and macro-Recall are compared with the
paired Wilcoxon signed-rank test over the 10 matched seeds, with Holm
correction (`p < 0.05`) within each scenario and metric (nine comparisons
per family). Reported standard deviations are sample standard deviations
(ddof = 1).

---

## Results

All numeric results reported in the paper are reproduced from the CSVs
under `statistical_tests/` and `ablation_study/`:

| Paper element | Repository file |
|---|---|
| Table 2 (multiseed test-set results) | `statistical_tests/summary_*.csv` |
| Table 3 (Wilcoxon–Holm by rival) | `statistical_tests/wilcoxon_adaharmon_vs_rivals.csv` |
| Table 4 (ablation study) | `ablation_study/ablation_*.csv` |
| Table 5 (computational cost) | analytic, see Section 3.5 of the paper |
| Supplementary Material (Sections S1–S7, including Tables S22–S24) | derived from the above CSVs |

### Headline results

- AdaHarmon obtained the **highest mean accuracy in 7 of 9 scenarios**,
  with a mean accuracy rank of 2.22 among the ten optimizers (next:
  AdaBelief, 2.78; Adam: 7.89).
- Against the nine baselines, it obtained **53 significant Accuracy
  improvements out of 81 comparisons** (Wilcoxon signed-rank,
  Holm-corrected, `p < 0.05`), 25 non-significant differences, and 3
  significant decreases.
- The 3 significant decreases all occur in EfficientNetV2–Raabin-WBC,
  against AdamW (−0.80 pp), RAdam (−0.63 pp), and AdaBelief (−0.75 pp).
  In MobileNetV3–Naturalize 2K-PBC, AdamW had the highest mean (91.01% vs.
  90.95%), a difference that is not significant.
- Largest margins on PBC, the lowest-resolution dataset (over Adam:
  +0.90, +2.83, and +5.76 pp on ResNet18, EfficientNetV2, and MobileNetV3).

Mean test accuracy (%), 10 seeds:

| Architecture | Dataset | AdaHarmon | Adam | Highest mean |
|---|---|---|---|---|
| ResNet18 | PBC | **95.05** | 94.15 | AdaHarmon |
| ResNet18 | Naturalize 2K-PBC | **92.52** | 91.48 | AdaHarmon |
| ResNet18 | Raabin-WBC | **96.80** | 96.00 | AdaHarmon |
| EfficientNetV2 | PBC | **93.07** | 90.23 | AdaHarmon |
| EfficientNetV2 | Naturalize 2K-PBC | **93.29** | 92.71 | AdaHarmon |
| EfficientNetV2 | Raabin-WBC | 96.05 | 96.21 | AdamW (96.85) |
| MobileNetV3 | PBC | **93.93** | 88.18 | AdaHarmon |
| MobileNetV3 | Naturalize 2K-PBC | 90.95 | 90.67 | AdamW (91.01) |
| MobileNetV3 | Raabin-WBC | **94.04** | 91.72 | AdaHarmon |

Complementary metrics (significant improvements / non-significant /
significant decreases, out of 81 comparisons): AUC 66/11/4, Loss 65/16/0,
F1 68/10/3, Recall 44/29/8. Recall is the least consistent metric: the
Accuracy advantage does not carry over uniformly to it.

### Ablation (PBC, mean test accuracy %, 10 seeds)

| Variant | ResNet18 | EfficientNetV2 | MobileNetV3 |
|---|---|---|---|
| AdaHarmon (full) | 95.04 | 93.08 | 93.93 |
| No harmonic (ℓ₂ denominator) | 94.66 | 91.79 | 91.34 |
| No adaptive β₁ | 94.73 | 92.08 | 91.92 |
| No soft decay | 94.91 | 92.64 | 93.07 |
| No harmonic + no adaptive β₁ | 94.18 | 90.28 | 88.29 |
| Adam | 94.16 | 90.23 | 88.18 |

Removing either the harmonic denominator or the adaptive momentum lowers
accuracy on all three architectures. Removing both together lowers it by
more than the sum of the individual drops (0.87 vs. 0.71 pp on ResNet18,
2.80 vs. 2.29 on EfficientNetV2, 5.63 vs. 4.59 on MobileNetV3): a
larger-than-additive effect. The excess is small on ResNet18 (0.16 pp,
below the seed-to-seed standard deviation) and no formal test of the
interaction was performed. Soft decay has the smallest individual effect.

### Computational cost

AdaHarmon keeps five per-parameter buffers (`m_t`, `v_t`, `u_t`, `s_t`,
and `μ_t`), which is **2.5× the optimizer state of Adam** and 5× that of
SGD (213.24 / 385.09 / 80.30 MB on ResNet18 / EfficientNetV2 /
MobileNetV3, versus 85.30 / 154.04 / 32.12 MB for Adam), and about
**2.6× the per-step arithmetic** of Adam (37 vs. 14 operations per
parameter). Total training-time ratios relative to Adam ranged from 0.78
to 1.59 across the nine scenarios; these were not timed under controlled
GPU-load conditions and are indicative only.

### Scope and limitations

The conclusions hold for the evaluated setting: three hematological
datasets of low to moderate resolution, three convolutional architectures
trained from scratch for 80 epochs, a fixed learning rate without
schedulers, and no individual hyperparameter tuning of any optimizer.
AdamW used the PyTorch default weight decay and SGD was not tuned, so
different values could shift their margins. Sensitivity analyses of the
learning rate (ResNet18–PBC, 5 seeds, Table S22) and of the internal
constants (Table S24) cover a single scenario. Benchmarks outside
hematological images (e.g., CIFAR-100, ImageNet), transformer or ConvNeXt
architectures, additional baselines (e.g., Yogi, MVN-Grad), per-class
metrics, and memory-efficient optimizer states were not evaluated.

---

## Citation

If you use AdaHarmon in your research, please cite:

```bibtex
@article{YanesPerez2026AdaHarmon,
  title   = {AdaHarmon: A Harmonic-Mean Adaptive Optimizer for
             CNN-Based Classification of Hematological Microscopic Images},
  author  = {Yanes-P{\'e}rez, Marcela de los {\'A}ngeles and
             Hern{\'a}ndez-Nolasco, Jos{\'e} Ad{\'a}n and
             Wister, Miguel A.},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Under revision}
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for
details.

---

## Contact

- **Corresponding author:** José Adán Hernández-Nolasco
  ([adan.hernandez@ujat.mx](mailto:adan.hernandez@ujat.mx))
- **First author:** Marcela de los Ángeles Yanes-Pérez
  ([marcelayanesperez@gmail.com](mailto:marcelayanesperez@gmail.com))
- **Institution:** Academic Division of Information Sciences and
  Technologies (DACyTI), Juarez Autonomous University of Tabasco
  (UJAT), Mexico.

For questions about the paper or the implementation, please open an
issue in this repository or contact the corresponding author.
