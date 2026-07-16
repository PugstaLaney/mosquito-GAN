# mosquito-GAN

> **A personal learning project.** This repository is my exploration of the pg-gan-mosquito codebase and the ideas in the accompanying paper. It is **not** original research and does not present any new methods or results. Everything technically interesting here — the demographic model, the GAN architecture, the mosquito adaptations — is the work of the Mathieson lab and their collaborators. My contribution is a set of annotated notebooks that walk through their code piece by piece to help me (and anyone else learning this material) understand how it fits together.

## Full credit and attribution

All of the underlying methods, code, and scientific work explored in this repository are the property of their original authors:

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

pg-gan-mosquito fits a 9-parameter demographic model — population sizes, split time, migration rate — to mosquito genomes from the Ag1000G Phase 2 dataset. A CNN discriminator tries to distinguish real *Anopheles gambiae* haplotypes from ones simulated by `msprime` under candidate demographic parameters θ; θ is iteratively updated to fool the discriminator, converging on a demographic history consistent with the observed genetic variation.

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

Exercises only the simulator side of the GAN. Shows:

- What the 9-parameter θ vector looks like as a Python object (`ParamSet`)
- How pg-gan proposes candidate θ updates (`Parameter.proposal`)
- What `msprime`'s `dadi_joint_mig` produces given θ: a `TreeSequence` containing 2,000+ coalescent trees and ~700 SNPs across a 5 kb region
- The genotype matrix (SNPs × haplotypes tensor) — the exact input the CNN discriminator scores
- Two visualizations: SNP-index heatmap and true bp-position scatter
- How output responds to seed changes (noise) vs θ changes (signal)

No CNN, no training, no real data — a "look inside the generator" step, purely for building intuition.

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

- `numpy==2.1.3` — pg-gan-mosquito's pinned `numpy==2.3.2` conflicts with TensorFlow 2.19's `numpy<2.2.0` cap.
- `msprime==1.3.4` — pg-gan-mosquito's pinned `msprime==1.3.3` has no Windows binary wheel; `1.3.4` does and is API-compatible.

These are minor patch-level bumps, not changes to their methods.

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name mosquito-gan --display-name "Python 3.11 (mosquito-GAN)"
```

Then open notebooks with that kernel.

## Data

The notebooks in this repo currently do not require the Ag1000G data. Any later notebooks that touch real mosquito genomes will need the [MalariaGEN Ag1000G Phase 2 AR1 data release](http://www.malariagen.net/data-package/ag1000g-phase-2-ar1/), which is publicly available.

## Citation

If any of this material is useful to you, please cite the underlying paper — not this repo:

> Eneli AA, Siu PC, Perez MF, Burt A, Fumagalli M, Mathieson S. *On the use of generative models for evolutionary inference of malaria vectors from genomic data*. bioRxiv 2025.06.26.661760.

## License

My annotations and notebooks in this repository are MIT-licensed. The pg-gan-mosquito code (which this repo imports but does not modify) remains under its original CC-BY-NC-4.0 license — please refer to their [repository](https://github.com/mathiesonlab/pg-gan-mosquito) for licensing terms on their code.

## Contact

Preston Laney — [github.com/PugstaLaney](https://github.com/PugstaLaney)
