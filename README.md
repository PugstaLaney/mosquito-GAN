# mosquito-GAN

Hands-on exploration of the pg-gan-mosquito pipeline — a generative adversarial network for inferring the demographic history of *Anopheles gambiae* (the malaria vector) from genomic data.

Companion notebooks to Eneli et al. (2025), [*On the use of generative models for evolutionary inference of malaria vectors from genomic data*](https://www.biorxiv.org/content/10.1101/2025.06.26.661760v2) (bioRxiv), and their [pg-gan-mosquito repository](https://github.com/mathiesonlab/pg-gan-mosquito).

## What the paper does

pg-gan-mosquito fits a 9-parameter demographic model — population sizes, split time, migration rate — to mosquito genomes from the Ag1000G Phase 2 dataset. A CNN discriminator tries to distinguish real mosquito haplotypes from ones simulated by `msprime` under candidate demographic parameters θ; θ is iteratively updated to fool the discriminator, converging on a demographic history consistent with the observed genetic variation.

## The full pipeline

```
1. Data prep       VCF (Ag1000G Phase 2) → HDF5
2. Simulator       θ vector + msprime → synthetic genotype matrix     ← notebook 01
3. Discriminator   CNN scoring real vs simulated regions
4. GAN training    Loop: update discriminator, propose new θ, repeat
5. Model select    Second CNN: classify real regions as mig / no-mig
6. Evaluate        Summary statistics, Wasserstein distance vs real
```

Notebooks in this repo cover individual pipeline stages, starting with the simulator (step 2).

## Notebooks

### `notebooks/01_simulate_theta.ipynb`

Exercises only the simulator side of the GAN. Shows:

- What the 9-parameter θ vector looks like as a Python object (`ParamSet`)
- How pg-gan proposes candidate θ updates (`Parameter.proposal`)
- What `msprime`'s `dadi_joint_mig` produces given θ: a `TreeSequence` containing 2,000+ coalescent trees and ~700 SNPs across a 5 kb region
- The genotype matrix (SNPs × haplotypes tensor) — the exact input the CNN discriminator scores
- Two visualizations: SNP-index heatmap and true bp-position scatter
- How output responds to seed changes (noise) vs θ changes (signal)

No CNN, no training, no real data — the "look inside the generator" step.

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

**Windows-specific pins** (fixes issues in pg-gan-mosquito's `requirements.txt`):

- `numpy==2.1.3` — the pinned `2.3.2` in pg-gan-mosquito conflicts with TensorFlow 2.19's `numpy<2.2.0` cap
- `msprime==1.3.4` — the pinned `1.3.3` has no Windows binary wheel; `1.3.4` does and is API-compatible

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name mosquito-gan --display-name "Python 3.11 (mosquito-GAN)"
```

Then open notebooks with that kernel.

## Data

The notebooks in this repo currently do not require the Ag1000G data. Later notebooks that touch real mosquito genomes will need the [MalariaGEN Ag1000G Phase 2 AR1 data release](http://www.malariagen.net/data-package/ag1000g-phase-2-ar1/).

## Citation

If you use this repo, please cite the underlying paper:

> Eneli AA, Siu PC, Perez MF, Burt A, Fumagalli M, Mathieson S. *On the use of generative models for evolutionary inference of malaria vectors from genomic data*. bioRxiv 2025.06.26.661760.

## License

Code in this repo is MIT-licensed. pg-gan-mosquito itself is under CC-BY-NC-4.0.
