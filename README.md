# micro-ataraxos: mechanistic reproduction of scalable decision-making for imperfect information

A single-GPU mechanistic study of **Ataraxos** ([Sokota et al., *Nature*, 2026](https://doi.org/10.1038/s41586-026-11036-y)) in a scaled **4×4 Micro-Stratego** environment.

Full 10×10 Stratego training requires 16 H100 GPUs for a week, amounting to approximately 208 billion environment steps. Rather than matching that scale, this project investigates whether the paper's core algorithmic mechanisms can be implemented and studied on modest compute.

The environment is reduced to a 4×4 board with four pieces per player: `Flag`, `Spy`, `Miner`, and `Marshal`. The implementation retains the interdependent setup and move-learning phases, dynamically damped self-play reinforcement learning, autoregressive hidden-state belief modelling, and test-time update-equivalence search.

**The goal is mechanistic fidelity, not scale replication.** The reduced environment makes the learning and search mechanisms easier to inspect, test, and ablate.

---

## Mechanistic Fidelity & Documented Simplifications

The implementation follows the paper's core design while making explicit reductions to the environment and computational setup. The table distinguishes reproduced mechanisms from changes in scale and implementation.

| Component | Paper specification | Implementation (`src/`) | Fidelity / scope |
| :--- | :--- | :--- | :--- |
| **Interdependent self-play** | Decoder-only setup Transformer trained with Monte Carlo returns, coupled with an encoder-only move Transformer trained with λ-returns | `SetupNetwork` + `MoveNetwork` (`src/networks.py`, `src/rl.py`) | Core architecture reproduced at reduced scale |
| **Spatial move parameterization** | Key-query bilinear product. | `torch.bmm(Q, K.transpose(1, 2)) * scale` | Parameterization reproduced |
| **Advantage magnitude filtering** | Train on moves with sufficiently large estimated advantage magnitudes, \(|\hat{A}_t|\) | Top-50% quantile filter on \(|\hat{A}_t|\) (`src/rl.py`) | Mechanism reproduced with an explicit threshold |
| **Dynamic damping and update control** | Power-law annealing of setup entropy regularization and move-policy regularization, proximal reverse-KL regularization against the data-collection policy, PPO-style clipping, gradient-norm clipping, and a power-law Adam learning-rate schedule | `get_damping_schedules` (`src/rl.py`) | Core mechanism implemented; schedule and loss details documented in code |
| **Autoregressive belief network** | Encoder, hidden-token filtering, and row-major autoregressive decoder | `BeliefNetwork._filter_hidden_tokens` and inventory-conservation masking | Core architecture reproduced at reduced scale |
| **Test-time search** | Depth-limited rollouts over sampled belief worlds, followed by a tabular magnetic mirror-descent update | `ataraxos_search_action` (`src/search.py`) | Search mechanism implemented with reduced game complexity |
| **Environment and horizon** | 10×10 board, 40 pieces per player, and the original movement, chasing, and game-length rules | 4×4 board, four pieces per player, two-square rule, and 24-move draw limit | Deliberate scale reduction |

The implementation-level descriptions above should be read alongside the source code and the documented deviations from the original method. Matching the architecture does not, by itself, establish exact equivalence of every update equation or training detail.

---

## Experimental Questions

The reproduction focuses on a few mechanism-level questions:

1. Does update-equivalence search improve play against a tactically distinct opponent?
2. How much of that improvement depends on learned belief modelling and search depth?
3. How do advantage filtering, learning-rate annealing, and regularization affect self-play learning?
4. How does policy entropy evolve under different regularization configurations?
5. How do the current policy iterate and EMA weights differ during training?

The experiments are designed to study these mechanisms in a reduced environment, not to estimate full-scale Stratego performance.

---

## Results

The current suite includes three independent training seeds, multiple training budgets, search ablations, and belief-model evaluation.

### 1. Self-play learning and regularization

The experiments compare the full training configuration against variants without learning-rate annealing, advantage filtering, or regularization annealing. A separate zero-regularization condition examines policy behaviour when regularization is removed.

The current results show differences in learning curves, final expected scores, and policy entropy across these configurations.

### 2. Test-time update-equivalence search

The search experiments compare:

- **Full search:** learned belief model, depth 6, 48 rollout worlds.
- **Shallow search:** learned belief model, depth 2, 16 rollout worlds.
- **Uniform belief:** full search depth and rollout count with a uniform belief prior.
- **No network KL:** full search with the network-reference KL coefficient removed.
- **Raw policy baseline:** the policy network without test-time search.

Across the three seeds, full search achieves a mean effective win rate of **75.8%**, compared with **51.8%** for the raw-policy baseline.

Shallow search and uniform belief modelling do not show the same improvement in this evaluation.

Removing network KL achieves a mean effective win rate of **74.0%**, close to the full-search result. The current experiments therefore do not establish that network KL is necessary for strong search performance in this environment.

These are results from the reduced environment and evaluation protocol, not claims about performance in full-scale Stratego.

### 3. Belief modelling

The learned belief model achieves a mean classification accuracy of **76.4%** across three seeds, with a mean negative log-likelihood of **0.523**, compared with **1.386** for the uniform baseline.

This provides evidence that the model learns to predict hidden piece identities beyond a uniform prior. Generalization to substantially different opponent policies remains a separate question.

### Experimental protocol

- **Environment:** 4×4 Micro-Stratego.
- **Training budgets:** 2,500 to 200,000 environment steps.
- **Independent training seeds:** 42, 43, and 44.
- **Search evaluation:** 120 games per condition per seed.
- **Uncertainty reporting:** observed variation across independent seeds; shaded regions in the training plots show the minimum-to-maximum seed range.

The number of seeds is intentionally modest. The results are intended as a mechanistic investigation, not a high-powered statistical comparison.

---

## Repository Structure

```text
├── paper_style.py             # Publication-style plotting and typography
├── run_reproduction.py        # Mechanism checks, training, evaluation, CSVs and figures
├── src/
│   ├── env.py                 # Vectorized GPU simulator, movement rules and myopic magnet policy
│   ├── networks.py            # Setup decoder, key-query move encoder and belief network
│   ├── rl.py                  # Damping schedules, λ-returns, EMA and belief training
│   ├── search.py              # Tabular update-equivalence search and evaluation
│   └── mechanism_checks.py    # Invariant and implementation-level checks
└── results/                   # Per-seed CSVs and publication-style figures
```

---

## Reproducing the Experiments

The main entry point is `run_reproduction.py`.

The script runs the mechanism checks, trains the self-play agents, evaluates the ablations, and saves the experimental results and figures.

Install the dependencies listed in `requirements.txt`, then run:

```bash
python run_reproduction.py
```

The experiments generate:

- `rl_ablation_seeds.csv`
- `search_ablation_seeds.csv`
- `belief_calibration_seeds.csv`
- `fig1_ataraxos_mechanisms.png`
- `fig2_search_and_setups.png`

Training and evaluation configurations, random seeds, and implementation details are documented in the source.

---

## Limitations

This is a reduced-scale mechanistic study, not a full-scale reproduction of Ataraxos.

The smaller board, reduced piece inventory, shorter horizon, simplified networks, and limited training budget change the strategic environment and computational demands. Consequently, the results should be interpreted as evidence about the implemented mechanisms within Micro-Stratego, not as estimates of full-scale Stratego performance.

The experiments use three independent seeds. Their observed variability is reported descriptively; it should not be interpreted as a precise estimate of population uncertainty.

The implementation and experiments are intended to make the algorithm's structure inspectable and its components experimentally testable. They do not establish that every implementation choice is optimal or that every observed effect generalizes beyond this environment.

---

## Reference

Sokota et al. (2026). *[Ataraxos: scalable decision-making for imperfect information](https://doi.org/10.1038/s41586-026-11036-y).* Nature.

This repository is an independent implementation and mechanistic study of the published method. Please cite the original paper when referring to Ataraxos itself.

---
