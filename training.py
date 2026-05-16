!pip install biopython -q

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from Bio import SeqIO
import matplotlib.pyplot as plt
import pickle, time, random

print("="*65)
print("AMP LSTM v5 - Muller 2018 Architecture")
print("="*65)
print(f"TensorFlow : {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU        : {gpus[0].name if gpus else 'NO GPU'}")
print("="*65)

FASTA_FILES = [
    'DBAASPS_peptides.fasta',
    'DRAMP_antibacterial.fasta',
    'DRAMP_antimicrobial.fasta',
    'APD_natural.fasta',
]

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
MIN_LEN, MAX_LEN = 5, 50

print("\nLoading FASTA files...")
raw = []
for f in FASTA_FILES:
    try:
        seqs = [str(r.seq).upper() for r in SeqIO.parse(f, "fasta")]
        valid = [s for s in seqs if MIN_LEN <= len(s) <= MAX_LEN
                                 and all(c in STANDARD_AA for c in s)]
        print(f"  {f}: {len(valid):,} sequences")
        raw.extend(valid)
    except FileNotFoundError:
        print(f"  {f} not found")

sequences = list(set(raw))
random.shuffle(sequences)
print(f"\n{len(sequences):,} unique sequences loaded")

CHARS  = ['B'] + sorted(STANDARD_AA)
VOCAB  = len(CHARS)
c2i    = {c: i for i, c in enumerate(CHARS)}
i2c    = {i: c for c, i in c2i.items()}

print(f"\nVocab: {VOCAB} chars")

SEQ_LEN = MAX_LEN + 1

print(f"\nEncoding {len(sequences):,} sequences (pad to {SEQ_LEN})...")

def encode_sequence(seq):
    full = 'B' + seq
    padded = full[:SEQ_LEN].ljust(SEQ_LEN, 'B')
    return [c2i[c] for c in padded]

encoded_seqs = [encode_sequence(s) for s in sequences]

def encode_full(seq):
    full = 'B' + seq
    padded = full[:SEQ_LEN+1].ljust(SEQ_LEN+1, 'B')
    return [c2i[c] for c in padded]

encoded_full = [encode_full(s) for s in sequences]
X_int = np.array([s[:-1] for s in encoded_full], dtype=np.int32)
y_int = np.array([s[1:]  for s in encoded_full], dtype=np.int32)

print(f"   X shape: {X_int.shape}")
print(f"   y shape: {y_int.shape}")

print("   One-hot encoding...")
X = keras.utils.to_categorical(X_int, num_classes=VOCAB).astype(np.float32)
y = keras.utils.to_categorical(y_int, num_classes=VOCAB).astype(np.float32)

print(f"   X: {X.shape}, y: {y.shape}")
print(f"Encoded. Total: {len(X):,} examples")

split   = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"\nSplit: {len(X_train):,} train / {len(X_val):,} val")

N_UNITS     = 256
DROPOUT     = 0.1
LR          = 0.001
weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=42)

model = keras.Sequential(name='AMP_LSTM_v5')

model.add(layers.LSTM(
    N_UNITS,
    input_shape=(SEQ_LEN, VOCAB),
    return_sequences=True,
    dropout=DROPOUT * 1,
    recurrent_dropout=0.0,
    kernel_initializer=weight_init,
    recurrent_initializer=weight_init,
    name='lstm_1'
))

model.add(layers.LSTM(
    N_UNITS,
    return_sequences=True,
    dropout=DROPOUT * 2,
    recurrent_dropout=0.0,
    kernel_initializer=weight_init,
    recurrent_initializer=weight_init,
    name='lstm_2'
))

model.add(layers.TimeDistributed(
    layers.Dense(VOCAB, activation='softmax',
                 kernel_initializer=weight_init),
    name='output'
))

optimizer = keras.optimizers.Adam(
    learning_rate=LR,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

model.summary()
print(f"\n{model.count_params():,} parameters")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_amp_lstm_v5.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
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

EPOCHS     = 100
BATCH_SIZE = 64

print("\n" + "="*65)
print("TRAINING")
print("="*65)
print(f"   Epochs     : {EPOCHS}")
print(f"   Batch size : {BATCH_SIZE}")
print(f"   Train      : {len(X_train):,}")
print(f"   Val        : {len(X_val):,}")
print("="*65 + "\n")

train_start = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

total_time = time.time() - train_start
tm, ts = divmod(int(total_time), 60)

best_epoch   = int(np.argmax(history.history['val_accuracy']))
best_val_acc = max(history.history['val_accuracy'])
best_trn_acc = history.history['accuracy'][best_epoch]

print(f"\n{'='*65}")
print(f"RESULTS")
print(f"{'='*65}")
print(f"  Time            : {tm}m {ts}s")
print(f"  Best epoch      : {best_epoch + 1}")
print(f"  Train accuracy  : {best_trn_acc*100:.2f}%")
print(f"  Val accuracy    : {best_val_acc*100:.2f}%")
print("="*65)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Train', lw=2, color='steelblue')
ax1.plot(history.history['val_accuracy'], label='Val',   lw=2, color='darkorange')
for thr, lbl, col in [(0.80,'80%','green'), (0.85,'85%','purple'), (0.90,'90%','red')]:
    ax1.axhline(thr, ls='--', lw=1.2, color=col, label=f'Target {lbl}')
ax1.set_title('Accuracy (train vs val)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(alpha=0.3)
ax1.set_ylim([0, 1])

ax2.plot(history.history['loss'],     label='Train', lw=2, color='steelblue')
ax2.plot(history.history['val_loss'], label='Val',   lw=2, color='darkorange')
ax2.set_title('Loss (train vs val)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle(f'AMP LSTM v5 - Best Val Acc: {best_val_acc*100:.1f}%', fontsize=14)
plt.tight_layout()
plt.savefig('training_history_v5.png', dpi=150, bbox_inches='tight')
plt.show()

vocab_data = {
    'CHARS'  : CHARS,
    'VOCAB'  : VOCAB,
    'c2i'    : c2i,
    'i2c'    : i2c,
    'SEQ_LEN': SEQ_LEN,
}
with open('amp_vocab_v5.pkl', 'wb') as f:
    pickle.dump(vocab_data, f)
with open('training_history_v5.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("Saved: best_amp_lstm_v5.keras | amp_vocab_v5.pkl")

def generate_peptide(model, vocab, seed='', min_len=10, max_len=30, temperature=1.0):
    c2i     = vocab['c2i']
    i2c     = vocab['i2c']
    SEQ_LEN = vocab['SEQ_LEN']
    VOCAB   = vocab['VOCAB']

    context = [c2i['B']] * SEQ_LEN
    for aa in seed:
        if aa in c2i:
            context.append(c2i[aa])
            context = context[-SEQ_LEN:]

    peptide = seed
    for _ in range(max_len - len(seed)):
        x    = np.array(context[-SEQ_LEN:])
        x_oh = keras.utils.to_categorical(x, num_classes=VOCAB)[np.newaxis]

        preds = model.predict(x_oh, verbose=0)[0, -1].astype(float)

        preds = np.log(preds + 1e-10) / temperature
        preds = np.exp(preds - preds.max())
        preds /= preds.sum()

        preds[c2i['B']] = 0.0
        preds /= preds.sum()

        next_id  = np.random.choice(VOCAB, p=preds)
        next_chr = i2c[next_id]

        if next_chr == 'B' and len(peptide) >= min_len:
            break
        elif next_chr != 'B':
            peptide += next_chr
            context.append(next_id)

    return peptide

print("\nSample generated peptides:")
with open('amp_vocab_v5.pkl', 'rb') as f:
    vocab = pickle.load(f)

test_seeds = [('', 'random'), ('GIG', 'GIG'), ('KK', 'KK'), ('RR', 'RR'), ('FL', 'FL')]
for seed, label in test_seeds:
    pep = generate_peptide(model, vocab, seed=seed, temperature=0.8)
    print(f"   Seed '{label}' -> {pep}  (len={len(pep)})")

print("\n" + "="*65)
print("Done")
print("="*65)