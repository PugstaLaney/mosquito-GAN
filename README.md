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

At a glance:

```
1. Data prep       VCF (Ag1000G Phase 2) → HDF5
2. Simulator       θ vector + msprime → synthetic genotype matrix     ← notebook 01
3. Discriminator   CNN scoring real vs simulated regions
4. GAN training    Loop: update discriminator, propose new θ, repeat
5. Model select    Second CNN: classify real regions as mig / no-mig
6. Evaluate        Summary statistics, Wasserstein distance vs real
```

My notebooks walk through individual pipeline stages, starting with the simulator (step 2). Each step expanded below.

### Step 1: Data prep (VCF → HDF5)

**Input:** raw whole-genome VCF files from Ag1000G Phase 2.

**About Ag1000G Phase 2.** The *Anopheles gambiae* **1000 Genomes** Project (a public genomics consortium releasing whole-genome sequences of malaria vector mosquitoes), operated by the MalariaGEN network. The "1000" naming echoes the human 1000 Genomes Project which set the pattern for population-scale variant catalogs. Phase 2 contains **1,142 wild-caught mosquitoes** from 13 sub-Saharan African countries, deep whole-genome sequenced. Covers two species (*An. gambiae*, *An. coluzzii*) plus a few hybrid forms. Free download at [malariagen.net](http://www.malariagen.net/data-package/ag1000g-phase-2-ar1/) after a click-through agreement. Delivered as one VCF per chromosome arm (2L, 2R, 3L, 3R, X, mitochondria).

**What VCF is.** Variant Call Format, the standard text-based bioinformatics format for storing genetic variants. One row per variant, columns for chromosome, position, reference allele, alternate allele(s), quality, and per-sample genotype calls (e.g. `0/1` = heterozygous).

**Two sub-steps in the data prep stage:**

1. **Filter the raw VCF** (done externally with tools like `bcftools`; not in the repo):
   - **Keep only chromosome arms 3L and 3R.** Rationale: chromosome 2 (arms 2L, 2R) has multiple large polymorphic inversions in *An. gambiae* which suppress recombination locally and create genealogical patterns that would confound demographic inference. The X chromosome is hemizygous in males (males have only one copy), so its effective population size is 3/4 that of the autosomes and its coalescent dynamics differ. Chromosome 3's autosomal arms are the "cleanest" surface for standard coalescent-based demographic inference.
   - **Keep only biallelic SNPs.** Pg-gan encodes each site as 0/1 (ancestral vs. derived); triallelic and tetrallelic sites (~26% of *An. gambiae* variants) can't fit that scheme and are dropped.
   - **Keep only sites passing Ag1000G's quality filters** and inside acceptable chromatin regions.
   - **Exclude the *Gste* gene region.** *Gste* (glutathione S-transferase epsilon cluster) is under strong recent selection for insecticide resistance, which would violate pg-gan's neutral-coalescent assumption.
   - **Subset samples to the analysis pair:** 31 Guinea (GN) + 81 Burkina Faso (BF) = 112 diploid individuals.
2. **Convert the filtered VCF to HDF5** (`vcf2hdf5.py` in the repo, ~50 lines wrapping one `scikit-allel` call):
   - Keeps only three fields: `CHROM` (chromosome name), `POS` (position in bp), `GT` (genotype calls).
   - Discards everything else (quality, INFO annotations, sample metadata).
   - HDF5 gives fast random access to arbitrary sample × SNP slices, which pg-gan needs thousands of times during training. VCF's text format would be prohibitively slow for that access pattern.

**Output:** an HDF5 file containing genotype matrices for chromosomes 3L and 3R, restricted to the 112 GN + BF samples.

**Where in the repo:** `vcf2hdf5.py` for the conversion; `real_data_random.py` reads the HDF5 during training.

### Step 2: Simulator (θ + msprime → genotype matrix)

**Input:** a demographic parameter vector θ (see the parameter table above).

**What it does:** runs a coalescent simulation under the demographic model defined by θ. This is the "generator" half of the GAN, but the generator is a physics simulator (`msprime`), not a neural network.

**How it works:**
- `simulation.dadi_joint_mig` builds an `msprime.Demography` object from θ (population sizes, split times, migration rates).
- Calls `msprime.sim_ancestry` to draw a random coalescent tree sequence over a 5 kb region for 224 haplotypes.
- Calls `msprime.sim_mutations` to sprinkle mutations onto the tree at rate μ per generation of branch length.
- Returns a `TreeSequence` object, from which the genotype matrix is extracted.

**Output:** synthetic genotype matrix of shape roughly (600-700 SNPs × 224 haplotypes), values 0/1. Same shape and format as a real data batch, so the discriminator can't tell them apart on structural grounds.

**Where in the repo:** `simulation.py`, `param_set.py`, `global_vars.py`. **Notebook 01 in this repo walks through this stage end-to-end.**

### Step 3: Discriminator (CNN scoring real vs. simulated)

**Input:** a batch of genotype matrices (real or simulated), shape `(batch, 224, 72, 2)`. The two channels are (a) allele values 0/1 and (b) the inter-SNP distances.

**What it is:** a convolutional neural network with roughly a million trainable weights.

**Architecture** (from `discriminator.py`):
- Several convolutional layers scanning across the (haplotypes × SNPs) matrix.
- A permutation-invariant pooling step over the haplotype dimension: mean rather than sum, because the two populations have different sample sizes and summing would let the larger one dominate.
- Dense layers projecting to a single sigmoid output.

**Output:** a probability per region: 0 means "confident this is fake", 1 means "confident this is real".

**How it's trained:** standard supervised binary classification with the AdamW optimizer. It's told which regions are real and which are simulated, and updates via backprop.

**Where in the repo:** `discriminator.py`.

### Step 4: GAN training (interleaved updates)

**What happens:** the main training loop alternates between two update rules on every iteration:

- **Discriminator update.** Take a batch of real regions plus a batch of simulated regions from the current θ. Forward-pass through the CNN. Compute binary cross-entropy loss against the true labels. Backprop. Update the discriminator's weights with AdamW.
- **θ update.** Propose K candidate perturbations to the current θ (Normal draws around each parameter, clipped to `[min, max]`). For each candidate, simulate a fresh batch of regions and score them with the frozen discriminator. Whichever candidate produced regions that scored highest as "real" becomes the new θ. This is a discrete search, not gradient descent, because msprime is not differentiable.

**Why it converges:** early in training, θ is a wild guess and the discriminator can easily tell simulated from real (accuracy near 100% on simulated data). As θ improves under selection pressure to fool the discriminator, the simulated regions become more realistic. At convergence, the discriminator's accuracy on simulated data drops toward 50%. That's the Nash equilibrium: the simulator has become good enough that even a well-trained CNN can't reliably distinguish simulated from real.

**Where in the repo:** `pg_gan.py` (the main training loop).

**Output:** a fitted θ vector (Table 1 in the paper). Also a trained discriminator that could be re-used for other tasks (e.g. as an outlier detector for regions that don't fit the neutral demographic history).

### Step 5: Model select (which demographic model fits?)

**The problem:** you can fit two different demographic models to the same data (e.g. one with post-split migration, one without). Which one better matches reality?

**The novel move in this paper.** Instead of using classical model-selection metrics like AIC or likelihood ratio tests, the authors train a SECOND CNN whose only job is to classify simulated regions as "came from mig model" or "came from no-mig model", and then let it vote on real data.

**How it works:**
1. Simulate many regions under the fitted no-mig parameters. Label as 0.
2. Simulate many regions under the fitted mig parameters. Label as 1.
3. Train a fresh CNN (same architecture as the discriminator) as a binary classifier on this simulated data.
4. Feed real data through the trained classifier and count how it labels each region.

**The paper's finding:** 74.4% of real regions get classified as "mig", 25.6% as "no-mig". Interpretation: the migration model is the better fit for real GN + BF data.

**Why this is nice:** you get a direct fraction rather than a hard-to-interpret likelihood ratio, and the classifier's own training accuracy tells you how *distinguishable* the two models are. If the classifier can't learn to separate them cleanly (as in this paper, where it plateaued around 65-70%), the two models produce fairly overlapping distributions and you shouldn't be overly confident in the winner.

**Where in the repo:** `demographic_selection_oop.py`.

### Step 6: Evaluate (does the fit actually match reality?)

**The question:** does data simulated under the fitted θ look like real data, absolutely? Model-selection tells you which of two variants is better; it doesn't tell you if EITHER is a good absolute fit.

**How:** compare distributions of standard population-genetics summary statistics between real regions and simulated regions.

**Statistics used** (all classical pop-gen metrics):
- **π (pairwise heterozygosity):** how genetically diverse each population is on average, measured as the fraction of sites at which two random haplotypes differ.
- **Watterson's θ:** an alternative diversity estimator based on the count of segregating sites.
- **Site frequency spectrum (SFS):** histogram of derived allele counts across the sample.
- **Inter-SNP distances:** how densely variants are packed along the chromosome.
- **Linkage disequilibrium (LD) decay:** how quickly the correlation between alleles falls off as SNPs get farther apart on the chromosome.
- **Number of distinct haplotypes:** how much haplotype diversity the sample carries.
- **F_ST:** how differentiated the two populations are from each other in allele frequencies.

For each statistic, they compute the distribution over many real regions and over many simulated regions, then measure the mismatch using **Wasserstein distance** (a.k.a. earth-mover's distance): the minimum "work" needed to transform one distribution into the other.

**The paper's headline result:** for the mig model, pg-gan-mosquito simulations match real data as well as (and for F_ST, better than) the incumbent ∂a∂i-based baseline. So the fit is validated.

**Where in the repo:** `summary_stats_multi.py`, `ss_helpers.py`.

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

---

## Companion project: DNA Language Model Benchmarking

A separate line of inquiry, sharing the same *Anopheles gambiae* context but independent of pg-gan-mosquito.

**Motivating question.** DNA foundation models (Nucleotide Transformer, DNABERT-2, HyenaDNA) were pretrained on hundreds of genomes and are increasingly used as feature extractors for downstream genomics ML. *An. gambiae* is a non-model organism relative to human-centric benchmarks. Do these models transfer meaningfully to mosquito data, and if so, uniformly or only for particular sequence types?

**Notebooks in `protein-lm/notebooks/`:**

- `00_setup.ipynb`: venv, path, and GPU sanity checks
- `01_tokenization_comparison.ipynb`: side-by-side of 6-mer (NT-v2), BPE (DNABERT-2), and single-nucleotide (HyenaDNA) tokenization on the same 200 bp synthetic sequence with common promoter motifs
- `02_perplexity_benchmark.ipynb`: masked-LM (NT-v2) and causal-LM (HyenaDNA) perplexity on 50 random 500 bp windows from chromosome 3L of the AgamP4 reference genome
- `03_perplexity_benchmark_coding.ipynb`: same benchmark restricted to spliced protein-coding sequences (CDS) on 3L, sampled per-transcript

**Preliminary findings:**

- **Foundation models trained on 850 diverse genomes show weak transfer to *An. gambiae* 3L.** NT-v2 mean MLM loss on random 3L windows: **8.560**, versus a uniform-baseline of **8.320** (essentially at chance). Correct-token probability is 3.2× uniform on average, but top-1 accuracy is 0 percent across 100 sampled positions.
- **The weakness is not attributable to non-coding sequence content.** Restricting the input to spliced CDS from chromosome 3L produces essentially the same loss (**8.722** for NT-v2, **1.428** for HyenaDNA), differing by less than 0.2 from the random-window numbers.
- **Transfer failure appears species-specific rather than sequence-type-specific.** Neither coding grammar (codon usage, splice signals) nor non-coding grammar is being captured well by pretraining on other species. This motivates domain-adapted pretraining or fine-tuning on Ag1000G data as a specific mitigation.
- **DNABERT-2 was excluded from the benchmark** due to a known compatibility issue between its custom BertConfig and current transformers versions. Fixing this requires either manual state-dict remapping (MosaicBERT to standard BertForMaskedLM layer names) or a transformers downgrade that would risk breaking NT-v2. Left as open follow-up.

**Data used:**

- AgamP4 reference genome, Ensembl Metazoa release-58, chromosomes 3L and (available for extension) 3R
- Ensembl Metazoa release-58 GTF annotation for CDS coordinate parsing

**Setup differs from the pg-gan-mosquito environment** (TensorFlow-based). This companion project uses a separate Python 3.11 venv with PyTorch, HuggingFace transformers, pyfaidx, and biopython. Data paths in the notebooks are currently hard-coded to a local drive layout; adapt them to your machine if reproducing.

Figures saved in `protein-lm/outputs/`:
- `perplexity_comparison.png`: NT-v2 and HyenaDNA loss on random 3L windows (from notebook 02)
- `perplexity_random_vs_coding.png`: grouped bar chart contrasting random windows against spliced CDS (from notebook 03)

---

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
