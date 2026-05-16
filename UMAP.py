!pip install biopython umap-learn -q

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pickle
import random
from Bio import SeqIO
from tensorflow import keras
import umap
import pandas as pd

print("="*65)
print("UMAP Distribution Shift Visualization")
print("="*65)

with open('amp_vocab_v5.pkl', 'rb') as f:
    vocab = pickle.load(f)

c2i     = vocab['c2i']
i2c     = vocab['i2c']
SEQ_LEN = vocab['SEQ_LEN']
VOCAB   = vocab['VOCAB']

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
HYDROPHOBIC = set('VILMFYWAC')
POSITIVE    = set('KRH')
NEGATIVE    = set('DE')

model_v5 = keras.models.load_model('best_amp_lstm_v5_finetuned.keras')
model_v6 = keras.models.load_model('best_amp_lstm_v6_paleo.keras')
print("Both models loaded")

AA_PROPS = {
    'A': (1.8,  0, 1), 'C': (2.5,  0, 2), 'D': (-3.5,-1, 2),
    'E': (-3.5,-1, 3), 'F': (2.8,  0, 4), 'G': (-0.4, 0, 1),
    'H': (-3.2, 0, 3), 'I': (4.5,  0, 3), 'K': (-3.9, 1, 3),
    'L': (3.8,  0, 3), 'M': (1.9,  0, 3), 'N': (-3.5, 0, 2),
    'P': (-1.6, 0, 2), 'Q': (-3.5, 0, 3), 'R': (-4.5, 1, 4),
    'S': (-0.8, 0, 2), 'T': (-0.7, 0, 2), 'V': (4.2,  0, 2),
    'W': (-0.9, 0, 5), 'Y': (-1.3, 0, 4),
}

def featurize(seq):
    if not seq or len(seq) == 0:
        return np.zeros(26)
    length       = len(seq)
    net_charge   = sum(1 for aa in seq if aa in POSITIVE) - sum(1 for aa in seq if aa in NEGATIVE)
    hydro_pct    = sum(1 for aa in seq if aa in HYDROPHOBIC) / length
    cationic_pct = sum(1 for aa in seq if aa in POSITIVE) / length
    anionic_pct  = sum(1 for aa in seq if aa in NEGATIVE) / length
    norm_length  = length / 50.0
    avg_hydro    = np.mean([AA_PROPS.get(aa, (0,0,0))[0] for aa in seq])
    AA_ORDER     = sorted(STANDARD_AA)
    aa_fracs     = [seq.count(aa) / length for aa in AA_ORDER]
    return np.array([net_charge, hydro_pct, cationic_pct,
                     anionic_pct, norm_length, avg_hydro] + aa_fracs,
                    dtype=np.float32)

print("\nLoading sequences...")

FASTA_FILES = ['DBAASPS_peptides.fasta', 'DRAMP_antibacterial.fasta',
               'DRAMP_antimicrobial.fasta', 'APD_natural.fasta']
modern_all = []
for f in FASTA_FILES:
    try:
        for r in SeqIO.parse(f, 'fasta'):
            s = str(r.seq).upper()
            if 10 <= len(s) <= 50 and all(c in STANDARD_AA for c in s):
                modern_all.append(s)
    except FileNotFoundError:
        pass
modern_sample = random.sample(modern_all, min(500, len(modern_all)))
print(f"  Modern training: {len(modern_sample)} sequences sampled")

ancient_all = []
try:
    for r in SeqIO.parse('ancient_amp_candidates.fasta', 'fasta'):
        s = str(r.seq).upper()
        if all(c in STANDARD_AA for c in s) and 10 <= len(s) <= 50:
            ancient_all.append(s)
    ancient_sample = random.sample(ancient_all, min(500, len(ancient_all)))
    print(f"  Ancient sequences: {len(ancient_sample)} sequences sampled")
except FileNotFoundError:
    print("  ancient_amp_candidates.fasta not found")
    raise

N_GEN = 500
print(f"\nGenerating {N_GEN} peptides from each model...")

def generate_batch(model, n=500, temperature=0.8):
    peptides = []
    for _ in range(n):
        context = [c2i['B']] * SEQ_LEN
        pep     = ''
        for _ in range(30):
            x     = np.array(context[-SEQ_LEN:])
            x_oh  = keras.utils.to_categorical(x, num_classes=VOCAB)[np.newaxis]
            preds = model.predict(x_oh, verbose=0)[0, -1].astype(float)
            preds[c2i['B']] = 0.0
            preds  = np.log(preds + 1e-10) / temperature
            preds  = np.exp(preds - preds.max())
            preds /= preds.sum()
            next_id = np.random.choice(VOCAB, p=preds)
            pep    += i2c[next_id]
            context.append(next_id)
        pep = ''.join(aa for aa in pep if aa in STANDARD_AA)
        if len(pep) >= 8:
            peptides.append(pep)
    return peptides

v5_generated = generate_batch(model_v5, N_GEN)
print(f"  v5 generated: {len(v5_generated)} peptides")
v6_generated = generate_batch(model_v6, N_GEN)
print(f"  v6 generated: {len(v6_generated)} peptides")

print("\nFeaturizing all sequences...")
feats_modern  = np.array([featurize(s) for s in modern_sample])
feats_ancient = np.array([featurize(s) for s in ancient_sample])
feats_v5      = np.array([featurize(s) for s in v5_generated])
feats_v6      = np.array([featurize(s) for s in v6_generated])

all_feats = np.vstack([feats_modern, feats_ancient, feats_v5, feats_v6])
labels    = (['modern_train'] * len(feats_modern) +
             ['ancient']      * len(feats_ancient) +
             ['v5_generated'] * len(feats_v5) +
             ['v6_generated'] * len(feats_v6))
print(f"Total: {len(all_feats)} sequences featurized (dim=26)")

print("\nRunning UMAP (2-3 minutes)...")
reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_components=2,
    random_state=42,
    metric='euclidean'
)
embedding = reducer.fit_transform(all_feats)
print("UMAP projection complete")

n_m  = len(feats_modern)
n_a  = len(feats_ancient)
n_v5 = len(feats_v5)

emb_modern  = embedding[:n_m]
emb_ancient = embedding[n_m:n_m+n_a]
emb_v5      = embedding[n_m+n_a:n_m+n_a+n_v5]
emb_v6      = embedding[n_m+n_a+n_v5:]

centroid_ancient = emb_ancient.mean(axis=0)
centroid_modern  = emb_modern.mean(axis=0)

dist_v5_to_ancient = np.linalg.norm(emb_v5.mean(axis=0) - centroid_ancient)
dist_v6_to_ancient = np.linalg.norm(emb_v6.mean(axis=0) - centroid_ancient)
dist_v5_to_modern  = np.linalg.norm(emb_v5.mean(axis=0) - centroid_modern)
dist_v6_to_modern  = np.linalg.norm(emb_v6.mean(axis=0) - centroid_modern)

print("\n" + "="*55)
print("CENTROID DISTANCES")
print("="*55)
print(f"v5 centroid -> ancient centroid: {dist_v5_to_ancient:.3f}")
print(f"v6 centroid -> ancient centroid: {dist_v6_to_ancient:.3f}")
print(f"v5 centroid -> modern centroid:  {dist_v5_to_modern:.3f}")
print(f"v6 centroid -> modern centroid:  {dist_v6_to_modern:.3f}")
print()
if dist_v6_to_ancient < dist_v5_to_ancient:
    shift = dist_v5_to_ancient - dist_v6_to_ancient
    print(f"v6 is {shift:.3f} units closer to ancient cluster than v5")
    print("Paleo fine-tuning shifted the generative distribution toward ancient sequence space.")
else:
    print("v5 is closer to ancient cluster on this UMAP projection.")
    print("Check ablation study for more rigorous comparison.")
print("="*55)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('UMAP: Evolutionary Grammar Transfer - Distribution Shift',
             fontsize=14, fontweight='bold')

ax = axes[0]
ax.scatter(emb_modern[:,0],  emb_modern[:,1],
           s=6, alpha=0.3, c='steelblue',    label='Modern training seqs')
ax.scatter(emb_ancient[:,0], emb_ancient[:,1],
           s=6, alpha=0.3, c='saddlebrown',  label='Ancient fine-tune seqs')
ax.scatter(emb_v5[:,0], emb_v5[:,1],
           s=14, alpha=0.6, c='royalblue',   label='v5 generated (modern)')
ax.scatter(emb_v6[:,0], emb_v6[:,1],
           s=14, alpha=0.6, c='firebrick',   label='v6 generated (paleo)')
ax.scatter(*emb_modern.mean(0),  s=200, c='steelblue',   marker='*', zorder=5)
ax.scatter(*emb_ancient.mean(0), s=200, c='saddlebrown', marker='*', zorder=5)
ax.scatter(*emb_v5.mean(0),      s=200, c='royalblue',   marker='*', zorder=5)
ax.scatter(*emb_v6.mean(0),      s=200, c='firebrick',   marker='*', zorder=5)
ax.annotate('', xy=centroid_ancient, xytext=emb_v5.mean(axis=0),
            arrowprops=dict(arrowstyle='->', color='royalblue', lw=2, ls='--'))
ax.annotate('', xy=centroid_ancient, xytext=emb_v6.mean(axis=0),
            arrowprops=dict(arrowstyle='->', color='firebrick', lw=2, ls='--'))
ax.set_title('All four populations in UMAP space\n(* = centroid)', fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.set_xlabel('UMAP dimension 1')
ax.set_ylabel('UMAP dimension 2')
ax.grid(alpha=0.2)

ax2 = axes[1]
ax2.scatter(emb_v5[:,0], emb_v5[:,1],
            s=20, alpha=0.7, c='royalblue', label=f'v5 generated (n={len(emb_v5)})')
ax2.scatter(emb_v6[:,0], emb_v6[:,1],
            s=20, alpha=0.7, c='firebrick', label=f'v6 generated (n={len(emb_v6)})')
ax2.scatter(emb_ancient[:,0], emb_ancient[:,1],
            s=6, alpha=0.25, c='saddlebrown', label='Ancient sequences')
ax2.scatter(*emb_v5.mean(0), s=250, c='royalblue', marker='*', zorder=5,
            label=f'v5 centroid (dist to ancient={dist_v5_to_ancient:.2f})')
ax2.scatter(*emb_v6.mean(0), s=250, c='firebrick', marker='*', zorder=5,
            label=f'v6 centroid (dist to ancient={dist_v6_to_ancient:.2f})')
ax2.scatter(*centroid_ancient, s=300, c='saddlebrown', marker='D', zorder=5,
            label='Ancient centroid')
ax2.set_title('Generated peptide clouds: v5 vs v6\n(closer to ancient = learned from ancient grammar)',
              fontweight='bold')
ax2.legend(fontsize=8, loc='best')
ax2.set_xlabel('UMAP dimension 1')
ax2.set_ylabel('UMAP dimension 2')
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('umap_distribution_shift.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: umap_distribution_shift.png")

print("\nGenerating property-space breakdown figure...")
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('Property Shift: What Did Ancient Training Change?',
              fontsize=13, fontweight='bold')

def compute_props_list(seq_list):
    props_list = []
    for s in seq_list:
        if not s: continue
        l      = len(s)
        charge = (sum(1 for aa in s if aa in POSITIVE)
                - sum(1 for aa in s if aa in NEGATIVE))
        hydro  = sum(1 for aa in s if aa in HYDROPHOBIC) / l * 100
        props_list.append({'charge': charge, 'hydro': hydro, 'length': l})
    return props_list

props_modern  = compute_props_list(modern_sample)
props_ancient = compute_props_list(ancient_sample)
props_v5g     = compute_props_list(v5_generated)
props_v6g     = compute_props_list(v6_generated)

for ax_, key, label in [
    (axes2[0], 'charge', 'Net Charge'),
    (axes2[1], 'hydro',  'Hydrophobicity (%)'),
    (axes2[2], 'length', 'Length (AA)')
]:
    vals = {
        'Modern train': [p[key] for p in props_modern],
        'Ancient':      [p[key] for p in props_ancient],
        'v5 generated': [p[key] for p in props_v5g],
        'v6 generated': [p[key] for p in props_v6g],
    }
    colors = ['steelblue', 'saddlebrown', 'royalblue', 'firebrick']
    for i, (name, data) in enumerate(vals.items()):
        ax_.hist(data, bins=15, alpha=0.55, color=colors[i],
                 label=name, edgecolor='white')
    ax_.set_xlabel(label)
    ax_.set_ylabel('Count')
    ax_.set_title(label, fontweight='bold')
    ax_.legend(fontsize=8)
    ax_.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('umap_property_shift.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: umap_property_shift.png")

df_umap = pd.DataFrame({
    'umap_x':   list(embedding[:,0]),
    'umap_y':   list(embedding[:,1]),
    'group':    labels,
    'sequence': (modern_sample + ancient_sample + v5_generated + v6_generated),
})
df_umap.to_csv('umap_coordinates.csv', index=False)
print("Saved: umap_coordinates.csv")

print("\n" + "="*65)
print("UMAP VISUALIZATION COMPLETE")
print("="*65)
print("Files saved:")
print("  umap_distribution_shift.png")
print("  umap_property_shift.png")
print("  umap_coordinates.csv")
print("="*65)