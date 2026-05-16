# Generative-AI-for-AMPS

# 🧬 AI-Driven Discovery of Novel Antimicrobial Peptides Against Drug Resistance

**Project ID:** HS-BCOM-292  
**Category:** Computational Biology & Bioinformatics  
**Level:** High School — Individual  

> An end-to-end deep learning pipeline that generates novel antimicrobial peptide (AMP) candidates against drug-resistant superbugs using LSTM neural networks — extended with a first-of-its-kind paleogenomics fine-tuning system that mines the immune systems of extinct species (Neanderthal, Woolly Mammoth, *Homo heidelbergensis*) to explore untapped evolutionary sequence space.

---

## 📋 Table of Contents

- [Background & Motivation](#background--motivation)
- [What Are AMPs?](#what-are-amps)
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Training History](#training-history)
- [Paleogenomics Extension](#paleogenomics-extension)
- [Results](#results)
- [MIC Prediction Model](#mic-prediction-model)
- [Web Application](#web-application)
- [Scoring Rubric](#scoring-rubric)
- [Data Sources](#data-sources)
- [Dependencies](#dependencies)
- [Known Issues & Notes](#known-issues--notes)
- [References](#references)
- [Acknowledgements](#acknowledgements)

---

## Background & Motivation

Antimicrobial resistance (AMR) is one of the most serious threats to global public health:

- **700,000 people die per year** from drug-resistant infections — today
- **10 million deaths per year projected by 2050** — surpassing cancer (GBD AMR Collaborators, 2024)
- The **last genuinely new antibiotic class** was discovered in **1987** — a 40+ year gap
- Developing one new antibiotic costs **$1 billion+** and takes **10–15 years**, with most candidates failing

Traditional antibiotic discovery is slow, expensive, and has largely stalled because bacteria continuously evolve resistance to molecular-target-based drugs. This project applies deep learning to the problem of **de novo antimicrobial peptide design** — generating novel peptide candidates computationally, in seconds, with no lab required.

---

## What Are AMPs?

Antimicrobial peptides are short protein chains (10–50 amino acids) found across all life forms — in frog skin, bee venom, human immune cells (defensins), spider venom, and more. They work by a fundamentally different mechanism than traditional antibiotics:

- They are **positively charged** (+2 to +9 net charge), making them electrostatically attracted to bacterial membranes, which carry negative surface charges
- They are **amphipathic** — one face hydrophobic, one face hydrophilic — allowing them to insert into the lipid bilayer
- They **physically rupture bacterial membranes** via barrel-stave or carpet models
- Because this is **mechanical destruction** rather than a molecular key-lock interaction, bacteria have a far harder time evolving resistance
- Human cell membranes lack the anionic lipids that attract AMPs, giving them built-in **selectivity for bacteria over human cells**

---

## Project Overview

This project has four major components:

### 1. Modern AMP Generator (v5)
An LSTM neural network trained on 17,836 real antimicrobial peptide sequences using the Müller 2018 concatenated corpus method. Achieves **87% validation accuracy**. Generates novel peptide candidates via temperature-based autoregressive sampling.

### 2. Paleogenomics Extension (v6)
A novel independent contribution: the v5 model is fine-tuned on ~2,000 AMP-like fragments mined from the proteomes of three extinct species — Neanderthal, Woolly Mammoth, and *Homo heidelbergensis*. These species evolved immune systems under completely different bacterial selection pressures, providing access to sequence space no living organism has explored. The paleo model generates peptides that are simultaneously **more novel and more active** than the modern model.

### 3. MIC Prediction Model
A GradientBoostingRegressor trained on 786 sequences with experimentally measured Minimum Inhibitory Concentration (MIC) values from DRAMP, using 450 engineered features. Pearson r = 0.55 (honest 10-fold cross-validation).

### 4. Interactive Web Application
A Streamlit app with two generator tabs (Modern + Paleo) and full analysis including 3D structure prediction via Meta ESMFold API, superbug targeting prediction, scoring breakdown, and CSV export.

---

## Repository Structure

```
amp-generator/
│
├── README.md                          # This file
│
├── models/                            # Trained model files (not included in repo — see Setup)
│   ├── best_amp_lstm_v5_finetuned.keras   # Modern model (87% val accuracy)
│   ├── best_amp_lstm_v6_paleo.keras       # Paleo fine-tuned model
│   ├── amp_vocab_v5.pkl                   # Vocabulary dictionary
│   └── mic_predictor.pkl                  # MIC prediction model
│
├── training/
│   ├── AMP_LSTM_v5_Muller_Exact.py    # v5 training script (Müller 2018 method)
│   ├── AMP_LSTM_v4_Research_Backed.py # Earlier architecture (for reference)
│   └── train_mic_model_DRAMP.py       # MIC model training script
│
├── generation/
│   ├── AMP_Generation_v2.py           # Standalone generation + evaluation script
│   └── AMP_Generation_Evaluation.py   # Batch generation with analysis charts
│
├── paleo/
│   └── paleo_amp_mining.ipynb         # Paleogenomics mining + v6 fine-tuning notebook
│
├── app/
│   └── streamlit_app.py               # Full Streamlit web application
│
├── data/
│   ├── generated_amps.csv             # 100 generated peptides with scores (modern v5)
│   └── ancient_amp_candidates.fasta   # ~2,000 ancient fragments used for v6 fine-tuning
│
└── results/
    ├── generated_amp_analysis.png     # Score distribution + property charts
    ├── training_curves.png            # v5 training accuracy/loss curves
    └── paleo_vs_modern_comparison.png # 3-panel paleo vs modern comparison
```

---

## Installation & Setup

### Requirements

- **Python 3.11.x** — TensorFlow does not support Python 3.12 on Windows as of this writing
- **pip** package manager

### Step 1 — Clone or download the repository

```bash
git clone https://github.com/yourusername/amp-generator.git
cd amp-generator
```

### Step 2 — Install dependencies

```bash
pip install tensorflow==2.15.0
pip install streamlit numpy pandas matplotlib scikit-learn biopython requests
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### Step 3 — Download model files

The trained `.keras` and `.pkl` model files are not included in the repository due to size. Download them from [Google Drive link] and place them in the project root directory (same folder as `streamlit_app.py`):

```
best_amp_lstm_v5_finetuned.keras
best_amp_lstm_v6_paleo.keras
amp_vocab_v5.pkl
mic_predictor.pkl
```

### Step 4 — Verify installation

```python
python -c "import tensorflow as tf; print(tf.__version__)"
# Expected: 2.15.0 or similar

python -c "from tensorflow import keras; m = keras.models.load_model('best_amp_lstm_v5_finetuned.keras'); print('Model loaded, input shape:', m.input_shape)"
# Expected: Model loaded, input shape: (None, 51, 21)
```

---

## How to Run

### Run the Web Application

```bash
# Standard Python
streamlit run streamlit_app.py

# If you have multiple Python versions (Windows)
py -3.11 -m streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Run Generation Standalone

```bash
python AMP_Generation_v2.py
```

This generates 100 peptides, scores them, prints the results, and saves `generated_amps.csv` and `generated_amp_analysis.png`.

### Retrain the Model from Scratch

```bash
# Train v5 from scratch (requires your AMP sequence data files)
python AMP_LSTM_v5_Muller_Exact.py

# Retrain MIC predictor
python train_mic_model_DRAMP.py
```

### Run the Paleo Mining Pipeline

Open and run all cells in `paleo_amp_mining.ipynb` in Jupyter or VS Code. Requires the ancient proteome FASTA files downloaded from NCBI (see notebook for exact accession numbers).

---

## Model Architecture

### v5 Modern LSTM

```
Input: (batch_size, 51, 21)  — 51 timesteps, one-hot over 21 characters
│
├── LSTM(256, return_sequences=True, dropout=0.1, kernel_initializer=RandomNormal)
│
├── LSTM(256, return_sequences=True, dropout=0.2)
│
└── TimeDistributed(Dense(21, activation='softmax'))

Output: (batch_size, 51, 21)  — probability distribution at every timestep
```

**Key design decisions:**

| Decision | Why |
|----------|-----|
| `return_sequences=True` on both LSTM layers | Generates a prediction at every timestep, not just the last — 50× more training signal per sequence |
| Concatenated corpus (no padding) | Every token is a real amino acid — zero wasted capacity on pad tokens |
| TimeDistributed Dense | Applies the same classification head at every position independently |
| Vocab size = 21 | 20 standard amino acids + `B` start/end marker |
| SEQ_LEN = 51 | Maximum sequence length (50 AA) + 1 start token = 51 |
| Dropout 0.1 / 0.2 | Prevents memorization; layer 2 gets higher dropout as it's closer to output |
| RandomNormal initializer | As specified in Müller 2018 |

### Generation Algorithm

```python
def generate_peptide(model, vocab, seed='', temperature=0.8, max_len=30):
    # 1. Initialize context with B start tokens
    context = [c2i['B']] * SEQ_LEN

    # 2. Autoregressive loop
    for _ in range(max_len):
        x_oh = to_categorical(context[-SEQ_LEN:], num_classes=VOCAB_SIZE)
        preds = model.predict(x_oh)[0, -1]   # prediction at last timestep only

        # 3. B-TOKEN COLLAPSE FIX — critical, without this generation fails
        preds[c2i['B']] = 0.0

        # 4. Temperature scaling
        preds = log(preds + 1e-10) / temperature
        preds = exp(preds - preds.max())
        preds /= preds.sum()

        # 5. Sample and append
        next_aa = i2c[choice(VOCAB_SIZE, p=preds)]
        peptide += next_aa

    return peptide
```

**The B-Token Collapse (novel discovery):** After fine-tuning, the model assigns ~99.99% probability to the B start token when the context is full of B tokens. Without zeroing `preds[B]` before sampling, generation produces infinite B characters. This failure mode is not documented in Müller 2018 and was discovered independently through debugging.

---

## Training History

| Version | Method | Val Accuracy | Notes |
|---------|--------|--------------|-------|
| v1 | Zero padding | 45–52% | Model learns padding patterns, not AMP biology |
| v2 | Zero padding + deeper | 55–60% | Marginal improvement, same fundamental problem |
| v3 | Zero padding + regularization | ~58% | Still limited by padding waste |
| v4 | Sliding window | ~52% | Model memorizes local context, doesn't generalize |
| v5 phase 1 | Concatenated corpus, lr=0.001 | **81%** | Breakthrough — 65 epochs |
| v5 phase 2 | Fine-tuning, lr=0.0001 | **87%** | 50 more epochs, saved as `best_amp_lstm_v5_finetuned.keras` |
| v6 | Paleo fine-tuning from v5, lr=0.00005 | Not recorded* | Saved as `best_amp_lstm_v6_paleo.keras` |

*v6 val accuracy was not recorded — the meaningful comparison metrics are novelty and activity scores from the generation comparison (see Results).

---

## Paleogenomics Extension

### Scientific Rationale

Living organisms' AMPs evolved under modern bacterial pressures — bacteria that have existed under antibiotic selection since penicillin (1928). Extinct species like Neanderthals and Woolly Mammoths evolved immune systems facing completely different ice-age pathogens, with no exposure to modern antibiotics. Their peptides represent **40,000–700,000 years of untapped evolutionary history**.

Inspiration: Maasch et al. (2023) Cell Host & Microbe demonstrated that directly mining archaic human genomes for AMP-like sequences ("molecular de-extinction") yields peptides active against drug-resistant bacteria. This project extends that concept by using ancient fragments as **fine-tuning data for a generative AI model** rather than directly testing mined sequences.

### Species Mined

| Species | Extinct Since | Genome Source | Why Interesting |
|---------|--------------|---------------|-----------------|
| *Homo neanderthalensis* | ~40,000 years ago | NCBI (Pääbo et al. 2010) | Closest extinct relative; Eurasian ice age immune system |
| *Mammuthus primigenius* | ~4,000 years ago | NCBI | Survived last ice age; different mammalian lineage |
| *Homo heidelbergensis* | ~200,000 years ago | NCBI | Common ancestor of Neanderthals + modern humans; rarest data |

### Mining Pipeline

```
Ancient FASTA proteomes
        │
        ▼
Sliding window (10–35 AA, all positions, all proteins)
        │
        ▼
Strict 5-rule filter:
  ✓ AMP score ≥ 90        (not 70 — only top-tier enters training)
  ✓ ≥ 2 cationic residues (K, R, or H)
  ✓ Net charge ≥ 3
  ✓ Hydrophobicity ≥ 30%
  ✓ No single AA > 60%    (removes low-complexity repeats)
        │
        ▼
Novelty check: remove exact matches to modern training data
        │
        ▼
~2,000 novel ancient AMP candidates
        │
        ▼
Fine-tune v5 at lr=0.00005, EarlyStopping patience=15, 30 epochs max
        │
        ▼
best_amp_lstm_v6_paleo.keras
```

---

## Results

### Modern v5 Model (100 peptides generated)

| Metric | Value |
|--------|-------|
| Peptides scoring ≥ 70 (active) | **87 / 100** |
| Average AMP score | **85.5 / 100** |
| Average net charge | **+4.3** |
| Average hydrophobicity | **37.9%** |
| Top peptide | **LLCRIPKYCKRKS** (score 100/100, charge +5, hydro 46.2%) |

**Emergent motifs** (not programmed, learned from data):
- GIG — signature of magainin (Zasloff 2002), a natural AMP from African frogs
- LL — found in LL-37, a major human defensin
- KK / KR — paired cationic motifs strongly associated with membrane binding

### Paleo v6 vs Modern v5 Comparison

| Metric | Modern v5 | Paleo v6 | Δ |
|--------|-----------|----------|---|
| Active peptides (score ≥ 70) | 87 / 100 | **98 / 100** | +11 |
| Average AMP score | 80.7 | **85.7** | +5.0 |
| Average novelty score | 0.553 | **0.576** | +0.023 |
| High-novelty peptides (novelty > 0.6) | 10 / 100 | **17 / 100** | +70% |
| Average net charge | ~4.2 | ~4.6 | +0.4 |
| Average hydrophobicity | ~38% | ~41% | +3% |

**Key finding:** The paleo model generates peptides that are simultaneously more novel AND more active — higher on both metrics at once. This is significant because there is typically a tradeoff between novelty and activity. Ancient evolutionary pressure found productive sequence space that modern databases have not catalogued.

**Novelty measurement:** For each generated peptide, pairwise sequence alignment (BioPython `pairwise2.align.globalxx`) against 300 randomly sampled training sequences. Novelty = 1 − max(normalized similarity score). Higher = more different from all known AMPs.

### Top 5 Generated Peptides (Modern v5)

| Sequence | AMP Score | Net Charge | Hydro % | Superbugs Targeted |
|----------|-----------|------------|---------|-------------------|
| LLCRIPKYCKRKS | 100 | +5 | 46.2% | MRSA, E. coli, K. pneumoniae, A. baumannii |
| KKLLFKKILKGL | 97 | +5 | 58.3% | MRSA, E. coli, P. aeruginosa, K. pneumoniae |
| RRLKIWLRKLK | 95 | +6 | 36.4% | MRSA, E. coli, P. aeruginosa, K. pneumoniae, A. baumannii |

---

## MIC Prediction Model

**MIC (Minimum Inhibitory Concentration)** is the gold-standard experimental measure of antibiotic potency — the lowest concentration needed to stop bacterial growth, in µg/mL. Lower = more potent.

### Data
- Source: DRAMP `general_amps.txt`, MIC values parsed from Target_Organism field
- Regex: `r'MIC[≤=]+\s*([\d]+\.?[\d]*)\s*[μµ]g/ml'`
- Final dataset: **786 sequences** with valid µg/mL MIC values

### Features (450 total)
- **Amino acid composition** (20): fraction of each standard AA
- **Physicochemical** (8): net charge, hydrophobicity, cationic fraction, anionic fraction, aromatic fraction, charge density, charge×hydrophobicity interaction, length
- **Dipeptide composition** (400): fraction of all 20×20 = 400 possible 2-AA combinations
- **Motifs** (9): KK, RR, GIG, LL, FF, KR, RK, LK, KL
- **Terminal residues** (2): is N-terminal cationic? is C-terminal cationic?
- **Structural scores** (4): helix score, amphipathic score, max cationic run, max hydrophobic run
- **Diversity** (1): unique AA fraction

### Model & Performance

```python
GradientBoostingRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.75,
    min_samples_leaf=3,
    random_state=42
)
# Predicts log10(MIC)
```

| Metric | Value |
|--------|-------|
| Best test Pearson r | 0.666 |
| Honest 10-fold CV Pearson r | **0.553** |
| RMSE | 0.331 log units (~2× error in real MIC units) |

**Honest limitation:** Published MIC prediction models achieving r ≥ 0.75 use 5,000–20,000 sequences. This model had 786. Additional data from ChEMBL AMP MIC assays would likely push performance to 0.70+.

---

## Web Application

Built with **Streamlit** (Python). Run with:

```bash
py -3.11 -m streamlit run streamlit_app.py
```

### Tab 1 — Modern AMP Generator
- Seed sequence input (optional — e.g. `KK`, `GIG`, `RR`, or blank for fully AI-driven)
- Temperature slider (0.5–1.5)
- Number of peptides to generate (1, 3, 5, or 10)
- 3D structure toggle
- Real-time generation → scoring → superbug prediction
- Live 3D ESMFold structure (spinning, interactive, downloadable as PDB)
- Detailed score breakdown + amino acid composition chart
- Batch comparison table + CSV export

### Tab 2 — Paleo-Inspired Generator
- Same interface with brown/gold color scheme
- Loads `best_amp_lstm_v6_paleo.keras`
- Labels generated peptides as paleo-inspired with explanation of source species
- PDB download labeled `PaleoAMP_[sequence].pdb`

### External APIs Used
- **ESMFold**: `https://api.esmatlas.com/foldSequence/v1/pdb/` — POST with sequence, returns PDB
- **3Dmol.js**: CDN-loaded JavaScript library for interactive 3D rendering in browser

---

## Scoring Rubric

The 100-point AMP scoring system, designed based on published AMP literature:

| Criterion | Ideal Range | Points | Biological Justification |
|-----------|-------------|--------|--------------------------|
| Length | 10–35 AA | +25 | Short enough to synthesize cheaply; long enough to span a membrane |
| Net charge | +2 to +9 | +30 | Electrostatic attraction to anionic bacterial membrane lipids |
| Hydrophobicity | 30–60% | +25 | Membrane insertion without non-specific toxicity |
| Cationic residues | ≥ 10% K/R/H | +20 | Density of positive charge ensures sustained membrane attraction |
| Bonus: KK/RR/KR motif | present | +5 | Paired cationic motifs dramatically boost membrane binding |
| Bonus: GIG/LL/FF motif | present | +5 | Motifs from known highly active natural AMPs |

Score capped at 100. Threshold for "likely active": **≥ 70**.

### Superbug Targeting Rules

| Superbug | Condition | Why |
|----------|-----------|-----|
| MRSA | charge ≥ 3 AND hydro ≥ 35% | Gram-positive; needs charge to reach thick peptidoglycan wall |
| E. coli MDR | charge ≥ 2 AND hydro ≥ 30% | Gram-negative; lower bar due to accessible outer membrane |
| P. aeruginosa MDR | charge ≥ 4 AND hydro ≥ 35% | Efflux pumps require higher charge to overcome |
| K. pneumoniae CRE | charge ≥ 3 AND 30% ≤ hydro ≤ 60% | Carbapenem-resistant; hydro ceiling prevents non-specificity |
| A. baumannii PDR | charge ≥ 5 OR (charge ≥ 3 AND hydro ≥ 40%) | Pan-resistant; needs high charge or charge+hydro combination |
| S. epidermidis | (KK/RR/LL/FF present) AND charge ≥ 2 | Biofilm-former; motif-dependent membrane penetration |

---

## Data Sources

| Database | Citation | URL | Sequences Used |
|----------|----------|-----|----------------|
| APD3 | Wang et al. (2016). *Nucleic Acids Research*, 44(D1), D1087–D1093 | https://aps.unmc.edu | ~2,600 |
| DRAMP 3.0 | Shi et al. (2022). *Nucleic Acids Research*, 50(D1), D488–D496 | http://dramp.cpu-bioinfor.org | ~12,000 |
| DBAASP v3 | Pirtskhalava et al. (2021). *Nucleic Acids Research*, 49(D1), D288–D297 | https://dbaasp.org | ~3,200 |
| **Total after dedup/filter** | | | **17,836** |

| Ancient Genome | NCBI Accession | Species |
|----------------|---------------|---------|
| Neanderthal proteome | Various — see notebook | *Homo neanderthalensis* |
| Woolly Mammoth proteome | Various — see notebook | *Mammuthus primigenius* |
| H. heidelbergensis | Various — see notebook | *Homo heidelbergensis* |

---

## Dependencies

```
tensorflow>=2.13.0,<2.16.0   # Must be 2.x; TF 3.x not tested
keras                         # Bundled with TF 2.x
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
biopython>=1.81              # For pairwise2 novelty measurement (paleo notebook)
streamlit>=1.28.0
matplotlib>=3.7.0
requests>=2.31.0
joblib>=1.3.0                # For loading mic_predictor.pkl
```

**requirements.txt:**
```
tensorflow==2.15.0
numpy==1.24.4
pandas==2.0.3
scikit-learn==1.3.2
biopython==1.81
streamlit==1.28.1
matplotlib==3.7.3
requests==2.31.0
joblib==1.3.2
```

---

## Known Issues & Notes

### Python Version
**Use Python 3.11.x.** TensorFlow does not support Python 3.12 on Windows. If you have multiple Python versions:
```bash
py -3.11 -m streamlit run streamlit_app.py
py -3.11 AMP_LSTM_v5_Muller_Exact.py
```

### ESMFold API
The ESMFold API (`https://api.esmatlas.com`) is a free public endpoint maintained by Meta. It can be slow (~30 seconds) or occasionally unavailable. If it returns an error, try again in a few minutes. Very long sequences (>40 AA) may time out.

### The B-Token Collapse
If you ever modify the generation script and start seeing output like `BBBBBBBBBBB...`, you have accidentally removed or broken the B-token fix. The fix is:
```python
preds[c2i['B']] = 0.0  # This line MUST be present before sampling
```
Without this line, generation is broken. See [Model Architecture](#model-architecture) for a full explanation.

### Model Files Are Not in the Repo
The `.keras` and `.pkl` files are too large for GitHub. Download them separately and place them in the same directory as `streamlit_app.py`. The app will tell you if it can't find them.

### MIC Predictor Feature Order
The `mic_predictor.pkl` must be used with the exact same `extract_features()` function from `train_mic_model_DRAMP.py`. If you change the feature extraction code in any way, you must retrain the model — the feature order is baked into the sklearn pipeline.

### v6 Paleo Model Accuracy
The final validation accuracy of `best_amp_lstm_v6_paleo.keras` was not recorded — the training output was interrupted. Do not cite a training accuracy for v6. The correct metrics to report for the paleo model are the generation comparison results: 0.576 vs 0.553 avg novelty, 17 vs 10 high-novelty peptides, 98 vs 87 active peptides.

---

## References

1. **Müller, A. T., Hiss, J. A., & Schneider, G.** (2018). Recurrent Neural Network Model for Constructive Peptide Design. *Journal of Chemical Information and Modeling*, 58(2), 472–479. https://doi.org/10.1021/acs.jcim.7b00414

2. **Wang, G., Li, X., & Wang, Z.** (2016). APD3: The antimicrobial peptide database as a tool for research and education. *Nucleic Acids Research*, 44(D1), D1087–D1093. https://doi.org/10.1093/nar/gkv1278

3. **Shi, G., et al.** (2022). DRAMP 3.0: An enhanced data repository of antimicrobial peptides. *Nucleic Acids Research*, 50(D1), D488–D496. https://doi.org/10.1093/nar/gkab651

4. **Pirtskhalava, M., et al.** (2021). DBAASP v3: Database of antimicrobial/cytotoxic activity and structure of peptides. *Nucleic Acids Research*, 49(D1), D288–D297. https://doi.org/10.1093/nar/gkaa991

5. **Lin, Z., et al.** (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130. https://doi.org/10.1126/science.ade2574 [ESMFold]

6. **Zasloff, M.** (2002). Antimicrobial peptides of multicellular organisms. *Nature*, 415(6870), 389–395. https://doi.org/10.1038/415389a

7. **GBD 2021 Antimicrobial Resistance Collaborators.** (2024). Global burden of bacterial antimicrobial resistance 1990–2021. *The Lancet*, 404(10459), 1199–1226. https://doi.org/10.1016/S0140-6736(24)01867-1

8. **Maasch, J. R. M. A., et al.** (2023). Molecular de-extinction of ancient antimicrobial peptides enabled by machine learning. *Cell Host & Microbe*, 34(5), 699–711. https://doi.org/10.1016/j.chom.2023.07.001 [Paleo AMP inspiration]

9. **Pääbo, S., et al.** (2010). A Draft Sequence of the Neandertal Genome. *Science*, 328(5979), 710–722. https://doi.org/10.1126/science.1188021

10. **Hochreiter, S., & Schmidhuber, J.** (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

---

## Acknowledgements

All data collection, model architecture design, training, debugging, paleogenomics pipeline construction, scoring rubric design, web application development, and analysis were completed independently.

**Databases:** APD3 (University of Nebraska Medical Center), DRAMP (China Pharmaceutical University), DBAASP (Ivane Javakhishvili Tbilisi State University)

**Tools:** TensorFlow/Keras (Google), Streamlit (Snowflake), ESMFold API (Meta AI), 3Dmol.js (University of Pittsburgh), BioPython, scikit-learn, NCBI GenBank

**Key inspiration:** Müller et al. (2018) for the LSTM architecture; Maasch et al. (2023) for the paleogenomics concept

---

*This project was completed as an independent science fair entry. All results are computational predictions and require wet-lab validation (peptide synthesis + MIC testing) before any clinical conclusions can be drawn.*
