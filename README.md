# mosquito-GAN

$\color{red}{\textsf{\textbf{A personal learning project. Not original research. All methods and code shown here are the work of the Mathieson lab.}}}$

> [!CAUTION]
> **This repository is a personal learning project.** It is my exploration of the pg-gan-mosquito codebase and the ideas in the accompanying paper.
>
> It is **NOT** original research and does not present any new methods or results.
>
> Everything technically interesting here (the demographic model, the GAN architecture, the mosquito adaptations) is the work of the **Mathieson lab and their collaborators**. My contribution is a set of annotated notebooks that walk through their code piece by piece to help me (and anyone else learning this material) understand how it fits together.
>
> If you are looking for the actual scientific work, please go to:
> - The paper: [Eneli et al. 2025 on bioRxiv](https://doi.org/10.1101/2025.06.26.661760)
> - The code: [github.com/mathiesonlab/pg-gan-mosquito](https://github.com/mathiesonlab/pg-gan-mosquito)

## Full credit and attribution

All of the underlying methods, code, and scientific work explored in this repository are the property of their original authors.

**Paper:**
> Eneli AA, Siu PC, Perez MF, Burt A, Fumagalli M, Mathieson S (2025). *On the use of generative models for evolutionary inference of malaria vectors from genomic data*. bioRxiv 2025.06.26.661760. [https://doi.org/10.1101/2025.06.26.661760](https://doi.org/10.1101/2025.06.26.661760)

**Original code (pg-gan-mosquito):**
- Sara Mathieson (Department of Biology, University of Pennsylvania; Department of Computer Science, Haverford College)
- Amelia Adibe Eneli, Pui Chung Siu, Matteo Fumagalli (School of Biological and Behavioural Sciences, Queen Mary University of London)
- Manolo F. Perez (Real Jardín Botánico, CSIC / Imperial College London)
- Austin Burt (Department of Life Sciences, Imperial College London)

Original repository: [github.com/mathiesonlab/pg-gan-mosquito](https://github.com/mathiesonlab/pg-gan-mosquito).

This repository imports their code and calls their functions. I have not modified any of their source. All annotations, notebooks, and explanatory prose in this repo are my own learning notes.

---

## What the paper does (in my own words, for my own learning)

pg-gan-mosquito fits a 9-parameter demographic model (population sizes, split time, migration rate) to mosquito genomes from the Ag1000G Phase 2 dataset. A CNN discriminator tries to distinguish real *Anopheles gambiae* haplotypes from ones simulated by `msprime` under candidate demographic parameters θ; θ is iteratively updated to fool the discriminator, converging on a demographic history consistent with the observed genetic variation.

The paper focuses on a pair of closely-related mosquito populations: **Guinea (GN)** and **Burkina Faso (BF)** *Anopheles gambiae*, 31 and 81 diploid samples respectively (62 + 162 = 224 haplotypes).

### The 9 demographic parameters (θ)

These are the values pg-gan-mosquito is inferring:

| Symbol | Meaning | Units |
|---|---|---|
| `NI` | Initial ancestral population size (deep past) | # individuals |
| `TG` | Time at which the ancestral population starts changing size | years ago |
| `NF` | Final ancestral population size, right before the split | # individuals |
| `TS` | Time of the population split | years ago |
| `NI1` | Initial size of POP1 (Guinea) immediately after the split | # individuals |
| `NF1` | Final size of POP1 today | # individuals |
| `NI2` | Initial size of POP2 (Burkina Faso) immediately after the split | # individuals |
| `NF2` | Final size of POP2 today | # individuals |
| `MG` | Bidirectional migration rate between POP1 and POP2 after the split (scaled: `MG = 2·NF·m`) | dimensionless |

Plus two fixed calibration constants: mutation rate `mut = 3.5×10⁻⁹` per site per generation, and recombination rate `reco = 1.45×10⁻⁸` per site per generation (Ag1000G-derived values).

### The demographic model as a timeline

```
       PAST                                                            PRESENT
        ◄─────────────────────────────────────────────────────────────►

           TG                                     TS                       NOW
           │                                      │                        │
[anc. pop  │                                      │                        │
  size NI ]│──── exponential change ────►[NF]     │                        │
           │                                      │ SPLIT                  │
           │                                      │                        │
           │                                      │ ┌─POP1: NI1─exp─►[NF1]─┤
           │                                      │ │  ◄── migration MG ──►│
           │                                      │ └─POP2: NI2─exp─►[NF2]─┤
```

## The full pipeline

```
1. Data prep       VCF (Ag1000G Phase 2) → HDF5
2. Simulator       θ vector + msprime → synthetic genotype matrix     ← notebook 01
3. Discriminator   CNN scoring real vs simulated regions
4. GAN training    Loop: update discriminator, propose new θ, repeat
5. Model select    Second CNN: classify real regions as mig / no-mig
6. Evaluate        Summary statistics, Wasserstein distance vs real
```

My notebooks walk through individual pipeline stages, starting with the simulator (step 2).

## Notebooks

### `notebooks/01_simulate_theta.ipynb`

**Purpose:** exercise only the simulator side of the GAN, the part that turns candidate demographic parameters θ into synthetic mosquito DNA. No CNN, no training, no real data.

**What each cell shows:**

- **Cell 1: Setup.** Auto-detects the pg-gan-mosquito repo location and adds it to `sys.path` so its modules import as a library.
- **Cell 2: θ as a concrete Python object.** Prints all 11 parameters (9 demographic + 2 calibration constants) with their values, ranges, and plain-English labels. Uses `ParamSet(simulation.dadi_joint_mig)`, whose defaults are pulled from the 2017 Ag1000G paper's Supplementary Table 2 for the BFA vs GNB population pair.
- **Cell 3: The θ update step.** Demonstrates `Parameter.proposal(current_value, multiplier)`, the mechanism by which pg-gan generates candidate new θ values during training. Displays small (multiplier=1.0) vs. big (multiplier=3.0) perturbations side by side.
- **Cell 4: Run the simulator.** Calls `simulation.dadi_joint_mig` with the paper's default θ and sample sizes matching GN + BF (62 + 162 haplotypes). Returns an msprime `TreeSequence`, the object containing all the simulated ancestry (~2,000 coalescent trees) plus SNPs (~700 in a 5 kb region).
- **Cell 4b: SNP-level DataFrame.** Pulls the simulated data into a pandas DataFrame with one row per SNP: position, per-population allele counts, per-population frequencies, and the absolute frequency difference between populations (an F_ST-like signal).
- **Cell 5: The genotype matrix (THE tensor the CNN sees).** Extracts the (SNPs × haplotypes) tensor of 0s and 1s, the exact input the CNN discriminator scores during pg-gan training. Two visualizations follow: a SNP-index heatmap (the first 100 SNPs), and a true bp-position scatter across the full 5 kb region.
- **Cell 6: Signal vs. noise.** Runs the simulator repeatedly, varying (a) the random seed at fixed θ to show the noise floor from coalescent stochasticity, and (b) the split time TS at fixed seed to show how output SNP counts respond to a real demographic-parameter change.

**Key takeaways (for anyone reading the notebook cold):**

- θ is a small (11-value) vector of physically meaningful numbers, not a neural-network weight tensor.
- `simulation.dadi_joint_mig` is a wrapper around `msprime.Demography` + `msprime.sim_ancestry` + `msprime.sim_mutations`. All of msprime's calls happen inside that one function.
- The genotype matrix is essentially a black-and-white image (SNPs × haplotypes, values 0/1). Population-genetics CNNs treat it exactly like that.
- Same θ + different seeds → statistical noise; different θ + same seed → systematic differences. The CNN's job is to distinguish (b) from (a) using many regions at once.

## Setup

### 1. Clone this repo and pg-gan-mosquito as siblings

```bash
git clone https://github.com/PugstaLaney/mosquito-GAN.git
git clone https://github.com/mathiesonlab/pg-gan-mosquito.git
```

Directory layout should be:

```
some_parent_dir/
├── mosquito-GAN/
└── pg-gan-mosquito/
```

The notebook auto-detects the pg-gan-mosquito repo at that relative location.

### 2. Create a Python 3.11 venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate         # Linux/Mac
.venv\Scripts\activate            # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows-specific notes** (small tweaks to pg-gan-mosquito's original `requirements.txt` to get the env to install on Windows):

- `numpy==2.1.3`: pg-gan-mosquito's pinned `numpy==2.3.2` conflicts with TensorFlow 2.19's `numpy<2.2.0` cap.
- `msprime==1.3.4`: pg-gan-mosquito's pinned `msprime==1.3.3` has no Windows binary wheel; `1.3.4` does and is API-compatible.

These are minor patch-level bumps, not changes to their methods.

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name mosquito-gan --display-name "Python 3.11 (mosquito-GAN)"
```

Then open notebooks with that kernel.

## Data

The notebooks in this repo currently do not require the Ag1000G data. Any later notebooks that touch real mosquito genomes will need the [MalariaGEN Ag1000G Phase 2 AR1 data release](http://www.malariagen.net/data-package/ag1000g-phase-2-ar1/), which is publicly available.

## Citation

If any of this material is useful to you, please cite the underlying paper, not this repo:

> Eneli AA, Siu PC, Perez MF, Burt A, Fumagalli M, Mathieson S. *On the use of generative models for evolutionary inference of malaria vectors from genomic data*. bioRxiv 2025.06.26.661760.

## License

My annotations and notebooks in this repository are MIT-licensed. The pg-gan-mosquito code (which this repo imports but does not modify) remains under its original CC-BY-NC-4.0 license. Please refer to their [repository](https://github.com/mathiesonlab/pg-gan-mosquito) for licensing terms on their code.

## Contact

Preston Laney ([github.com/PugstaLaney](https://github.com/PugstaLaney))
