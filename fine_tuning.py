!pip install biopython -q

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Bio import SeqIO
import pickle, random

FASTA_FILES = [
    'DBAASPS_peptides.fasta',
    'DRAMP_antibacterial.fasta',
    'DRAMP_antimicrobial.fasta',
    'APD_natural.fasta',
]
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
CHARS   = ['B'] + sorted(STANDARD_AA)
VOCAB   = len(CHARS)
c2i     = {c: i for i, c in enumerate(CHARS)}
SEQ_LEN = 52

raw = []
for f in FASTA_FILES:
    try:
        seqs = [str(r.seq).upper() for r in SeqIO.parse(f, "fasta")]
        valid = [s for s in seqs if 5 <= len(s) <= 50
                                 and all(c in STANDARD_AA for c in s)]
        raw.extend(valid)
        print(f"  {f}: {len(valid):,} sequences")
    except:
        print(f"  {f} not found")

sequences = list(set(raw))
random.shuffle(sequences)
print(f"{len(sequences):,} unique sequences")

def encode_full(seq):
    full = 'B' + seq
    padded = full[:SEQ_LEN+1].ljust(SEQ_LEN+1, 'B')
    return [c2i[c] for c in padded]

encoded_full = [encode_full(s) for s in sequences]
X_int = np.array([s[:-1] for s in encoded_full], dtype=np.int32)
y_int = np.array([s[1:]  for s in encoded_full], dtype=np.int32)
X = keras.utils.to_categorical(X_int, num_classes=VOCAB).astype(np.float32)
y = keras.utils.to_categorical(y_int, num_classes=VOCAB).astype(np.float32)

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
print(f"Data ready: {len(X_train):,} train / {len(X_val):,} val")

model = keras.models.load_model('best_amp_lstm_v5.keras')
print(f"Model loaded")
print(f"Current LR: {model.optimizer.learning_rate}")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_amp_lstm_v5_finetuned.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
]

model.optimizer.learning_rate = 0.0001
print(f"LR set to 0.0001")

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

best_val = max(history2.history['val_accuracy'])
print(f"\nBest val accuracy after fine-tuning: {best_val*100:.2f}%")
print(f"Saved as: best_amp_lstm_v5_finetuned.keras")