
import os
import re
import string
import warnings

import numpy as np

# Suppress TensorFlow INFO/WARNING logs for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Dataset
DATASET_URL = "https://www.gutenberg.org/files/100/100-0.txt"
DATASET_PATH = "shakespeare.txt"

# Preprocessing
MAX_TEXT_LENGTH = 200_000          # Limit text length for faster training (set None for full)
SEQ_LENGTH = 100                   # Number of characters per input sequence
STEP_SIZE = 3                      # Sliding window step (reduces dataset size, speeds training)

# Training
BATCH_SIZE = 128
EPOCHS = 50
VALIDATION_SPLIT = 0.1             # 10% for validation
LEARNING_RATE = 0.001

# Text generation
GENERATION_LENGTH = 500            # Number of characters to generate

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# SECTION 1: Dataset Loading & Preprocessing
# =============================================================================

def download_dataset(url: str, filepath: str) -> str:
    """
    Download the text dataset from the given URL and save it locally.
    If the file already exists, load from disk (caching).

    Args:
        url:      URL to download the text from.
        filepath: Local path to save/load the file.

    Returns:
        Raw text content as a string.
    """
    if os.path.exists(filepath):
        print(f"[INFO] Dataset already cached at '{filepath}'. Loading from disk.")
    else:
        print(f"[INFO] Downloading dataset from {url} ...")
        import requests
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"[INFO] Dataset saved to '{filepath}' ({len(response.text):,} characters).")

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def preprocess_text(raw_text: str, max_length: int = None) -> str:
    """
    Clean and normalize the raw text:
      - Convert to lowercase
      - Remove punctuation (keep spaces and newlines for readability)
      - Collapse multiple whitespace into single spaces
      - Optionally truncate to max_length characters

    Args:
        raw_text:   The raw text string to preprocess.
        max_length: If set, truncate the text to this many characters.

    Returns:
        Cleaned text string.
    """
    text = raw_text.lower()

    # Remove punctuation but keep spaces and newlines
    # We translate punctuation characters to empty strings
    punct_to_remove = string.punctuation.replace("'", "")  # keep apostrophes for readability
    text = text.translate(str.maketrans("", "", punct_to_remove))

    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate if requested
    if max_length and len(text) > max_length:
        text = text[:max_length]
        print(f"[INFO] Text truncated to {max_length:,} characters for faster training.")

    return text


def build_char_mappings(text: str) -> tuple:
    """
    Create character-to-index and index-to-character mappings.

    Args:
        text: The cleaned text string.

    Returns:
        Tuple of (sorted_chars, char_to_idx, idx_to_char, vocab_size).
    """
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    print(f"[INFO] Vocabulary size: {vocab_size} unique characters")
    print(f"[INFO] Characters: {''.join(chars)}")

    return chars, char_to_idx, idx_to_char, vocab_size


def create_sequences(text: str, seq_length: int, step_size: int,
                     char_to_idx: dict, vocab_size: int) -> tuple:
    """
    Create input-output pairs using a sliding window approach.

    For each window of `seq_length` characters, the input is the window and
    the output (label) is the next character after the window.

    Args:
        text:        Cleaned text string.
        seq_length:  Number of characters per input sequence.
        step_size:   Step between consecutive windows.
        char_to_idx: Character-to-index mapping.
        vocab_size:  Total number of unique characters.

    Returns:
        Tuple of (X, y) numpy arrays.
        X shape: (num_sequences, seq_length)
        y shape: (num_sequences, vocab_size) — one-hot encoded
    """
    sequences = []
    next_chars = []

    for i in range(0, len(text) - seq_length, step_size):
        seq = text[i : i + seq_length]
        target = text[i + seq_length]
        sequences.append([char_to_idx[ch] for ch in seq])
        next_chars.append(char_to_idx[target])

    X = np.array(sequences, dtype=np.int32)
    y = keras.utils.to_categorical(next_chars, num_classes=vocab_size)

    print(f"[INFO] Created {len(sequences):,} sequences (seq_length={seq_length}, step={step_size})")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    return X, y


# =============================================================================
# SECTION 2: Model Architecture
# =============================================================================

def build_model(vocab_size: int, seq_length: int,
                embedding_dim: int = 64,
                lstm_units: int = 256,
                num_lstm_layers: int = 1,
                dropout_rate: float = 0.2,
                learning_rate: float = 0.001,
                model_name: str = "base_lstm") -> keras.Model:
    """
    Build an LSTM-based text generation model.

    Architecture:
        Input → Embedding → LSTM(s) → Dropout → Dense(softmax)

    Args:
        vocab_size:       Number of unique characters (output classes).
        seq_length:       Length of input sequences.
        embedding_dim:    Dimensionality of the character embedding.
        lstm_units:       Number of units in each LSTM layer.
        num_lstm_layers:  Number of stacked LSTM layers.
        dropout_rate:     Dropout rate after the last LSTM layer.
        learning_rate:    Learning rate for the Adam optimizer.
        model_name:       Name for the model.

    Returns:
        Compiled Keras model.
    """
    # Use Functional API for explicit input shape (ensures model is fully built in Keras 3.x)
    inputs = layers.Input(shape=(seq_length,), dtype="int32", name="input")

    # Embedding layer: maps character indices to dense vectors
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding"
    )(inputs)

    # LSTM layer(s)
    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1)  # Only last LSTM returns single output
        x = layers.LSTM(
            lstm_units,
            return_sequences=return_sequences,
            name=f"lstm_{i+1}"
        )(x)

    # Dropout to prevent overfitting
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Dense output layer with softmax activation for character prediction
    outputs = layers.Dense(vocab_size, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile with categorical crossentropy loss and Adam optimizer
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

    return model


# =============================================================================
# SECTION 3: Model Training
# =============================================================================

def train_model(model: keras.Model, X: np.ndarray, y: np.ndarray,
                batch_size: int = 128, epochs: int = 50,
                validation_split: float = 0.1,
                checkpoint_path: str = "best_model.keras") -> keras.callbacks.History:
    """
    Train the model with early stopping and model checkpointing.

    Args:
        model:            Compiled Keras model.
        X:                Input sequences (num_samples, seq_length).
        y:                One-hot encoded targets (num_samples, vocab_size).
        batch_size:       Training batch size.
        epochs:           Maximum number of training epochs.
        validation_split: Fraction of data to use for validation.
        checkpoint_path:  Path to save the best model weights.

    Returns:
        Training history object.
    """
    # Early stopping: stop training if validation loss doesn't improve for 3 epochs
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Model checkpoint: save the best model based on validation loss
    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    print("\n" + "=" * 70)
    print(f"TRAINING: {model.name}")
    print("=" * 70)
    model.summary()

    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

    return history


# =============================================================================
# SECTION 4: Text Generation
# =============================================================================

def sample_with_temperature(predictions: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample a character index from the model's prediction distribution,
    scaled by temperature.

    Temperature controls the randomness of the sampling:
      - temperature < 1.0 → more conservative (picks high-probability chars)
      - temperature = 1.0 → standard sampling from the distribution
      - temperature > 1.0 → more creative/random

    Args:
        predictions: Probability distribution over characters.
        temperature: Sampling temperature.

    Returns:
        Sampled character index.
    """
    predictions = np.asarray(predictions).astype("float64")

    # Apply temperature scaling in log-space
    log_preds = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(log_preds)
    probabilities = exp_preds / np.sum(exp_preds)

    # Sample from the distribution
    return np.random.choice(len(probabilities), p=probabilities)


def generate_text(model: keras.Model, seed_text: str,
                  char_to_idx: dict, idx_to_char: dict,
                  seq_length: int, length: int = 500,
                  temperature: float = 1.0) -> str:
    """
    Generate new text by iteratively predicting the next character.

    Process:
        1. Encode the seed text into character indices
        2. Feed the sequence to the model to get next-character probabilities
        3. Sample the next character using temperature-based sampling
        4. Append the new character and slide the window forward
        5. Repeat for `length` characters

    Args:
        model:       Trained Keras model.
        seed_text:   Initial text to seed the generation (must be >= seq_length chars).
        char_to_idx: Character-to-index mapping.
        idx_to_char: Index-to-character mapping.
        seq_length:  Sequence length the model was trained on.
        length:      Number of characters to generate.
        temperature: Sampling temperature (controls creativity).

    Returns:
        Generated text string (seed + generated characters).
    """
    # Ensure seed is long enough; pad or truncate to seq_length
    if len(seed_text) < seq_length:
        seed_text = seed_text.ljust(seq_length)
    elif len(seed_text) > seq_length:
        seed_text = seed_text[:seq_length]

    generated = list(seed_text)
    current_seq = [char_to_idx.get(ch, 0) for ch in seed_text]

    for _ in range(length):
        # Prepare input: shape (1, seq_length)
        x_input = np.array([current_seq], dtype=np.int32)

        # Predict next character probabilities
        predictions = model.predict(x_input, verbose=0)[0]

        # Sample next character
        next_idx = sample_with_temperature(predictions, temperature)
        next_char = idx_to_char[next_idx]

        # Append to generated text and slide the window
        generated.append(next_char)
        current_seq = current_seq[1:] + [next_idx]

    return "".join(generated)


def run_generation_demo(model: keras.Model, text: str,
                        char_to_idx: dict, idx_to_char: dict,
                        seq_length: int, model_name: str = "Model"):
    """
    Run text generation with multiple seed inputs and temperatures.

    Generates and prints sample outputs for evaluation.

    Args:
        model:       Trained model.
        text:        Full cleaned text (used to extract seed sequences).
        char_to_idx: Character-to-index mapping.
        idx_to_char: Index-to-character mapping.
        seq_length:  Sequence length.
        model_name:  Name for display.
    """
    # Define seed inputs from different parts of the text
    seeds = [
        text[:seq_length],                              # Beginning of text
        text[len(text)//4 : len(text)//4 + seq_length], # Quarter mark
        text[len(text)//2 : len(text)//2 + seq_length], # Middle of text
    ]

    temperatures = [0.5, 1.0, 1.2]

    print("\n" + "=" * 70)
    print(f"TEXT GENERATION — {model_name}")
    print("=" * 70)

    for i, seed in enumerate(seeds, 1):
        for temp in temperatures:
            print(f"\n--- Seed {i} | Temperature: {temp} ---")
            print(f"Seed: \"{seed[:60]}...\"")
            generated = generate_text(
                model, seed, char_to_idx, idx_to_char,
                seq_length, length=GENERATION_LENGTH, temperature=temp
            )
            # Print only the generated part (after the seed)
            print(f"Generated:\n{generated[seq_length:]}\n")


# =============================================================================
# SECTION 5: Bonus — Architecture Experiments
# =============================================================================

def run_experiments(text: str, X: np.ndarray, y: np.ndarray,
                    char_to_idx: dict, idx_to_char: dict,
                    vocab_size: int):
    """
    Compare different LSTM architectures and report findings.

    Experiments:
        1. Base model:   1 LSTM layer, 256 units, seq_length=100
        2. Deeper model: 2 LSTM layers, 256 units each
        3. Wider model:  1 LSTM layer, 512 units

    Each model is trained with the same data and hyperparameters,
    then compared on validation loss and generated text quality.

    Args:
        text:        Cleaned text.
        X:           Input sequences.
        y:           Target labels (one-hot).
        char_to_idx: Character-to-index mapping.
        idx_to_char: Index-to-character mapping.
        vocab_size:  Vocabulary size.
    """
    experiments = [
        {
            "name": "Deeper LSTM (2 layers × 256 units)",
            "params": {
                "num_lstm_layers": 2,
                "lstm_units": 256,
                "model_name": "deeper_lstm"
            },
            "checkpoint": "best_model_deeper.keras"
        },
        {
            "name": "Wider LSTM (1 layer × 512 units)",
            "params": {
                "num_lstm_layers": 1,
                "lstm_units": 512,
                "model_name": "wider_lstm"
            },
            "checkpoint": "best_model_wider.keras"
        },
    ]

    results = []

    for exp in experiments:
        print(f"\n{'#' * 70}")
        print(f"# EXPERIMENT: {exp['name']}")
        print(f"{'#' * 70}")

        model = build_model(
            vocab_size=vocab_size,
            seq_length=SEQ_LENGTH,
            learning_rate=LEARNING_RATE,
            **exp["params"]
        )

        history = train_model(
            model, X, y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            checkpoint_path=exp["checkpoint"]
        )

        # Record results
        best_val_loss = min(history.history["val_loss"])
        best_val_acc = max(history.history["val_accuracy"])
        num_epochs = len(history.history["loss"])

        results.append({
            "name": exp["name"],
            "val_loss": best_val_loss,
            "val_accuracy": best_val_acc,
            "epochs_trained": num_epochs,
        })

        # Generate sample text
        run_generation_demo(model, text, char_to_idx, idx_to_char, SEQ_LENGTH, exp["name"])

    # Print comparison report
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON REPORT")
    print("=" * 70)
    print(f"{'Model':<45} {'Val Loss':>10} {'Val Acc':>10} {'Epochs':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<45} {r['val_loss']:>10.4f} {r['val_accuracy']:>10.4f} {r['epochs_trained']:>8}")
    print("-" * 75)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function orchestrating the complete pipeline:
        1. Load and preprocess the dataset
        2. Build the base LSTM model
        3. Train the model
        4. Generate text samples
        5. Run bonus architecture experiments
    """
    print("=" * 70)
    print("  LSTM TEXT GENERATION — Shakespeare's Works")
    print("=" * 70)

    # ---- Step 1: Load & Preprocess Dataset ----
    print("\n[STEP 1] Loading and preprocessing dataset...")
    raw_text = download_dataset(DATASET_URL, DATASET_PATH)
    text = preprocess_text(raw_text, max_length=MAX_TEXT_LENGTH)
    print(f"[INFO] Cleaned text length: {len(text):,} characters")
    print(f"[INFO] Sample text: \"{text[:200]}...\"")

    # Build character mappings
    chars, char_to_idx, idx_to_char, vocab_size = build_char_mappings(text)

    # Create input-output sequence pairs
    X, y = create_sequences(text, SEQ_LENGTH, STEP_SIZE, char_to_idx, vocab_size)

    # ---- Step 2: Build Base Model ----
    print("\n[STEP 2] Building base LSTM model...")
    base_model = build_model(
        vocab_size=vocab_size,
        seq_length=SEQ_LENGTH,
        embedding_dim=64,
        lstm_units=256,
        num_lstm_layers=1,
        dropout_rate=0.2,
        learning_rate=LEARNING_RATE,
        model_name="base_lstm"
    )

    # ---- Step 3: Train the Model ----
    print("\n[STEP 3] Training the model...")
    base_history = train_model(
        base_model, X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        checkpoint_path="best_model_base.keras"
    )

    # Print training summary
    best_val_loss = min(base_history.history["val_loss"])
    best_val_acc = max(base_history.history["val_accuracy"])
    print(f"\n[RESULT] Base Model — Best val_loss: {best_val_loss:.4f}, Best val_accuracy: {best_val_acc:.4f}")

    # ---- Step 4: Generate Text ----
    print("\n[STEP 4] Generating text...")
    run_generation_demo(base_model, text, char_to_idx, idx_to_char, SEQ_LENGTH, "Base LSTM (1×256)")

    # ---- Step 5: Bonus — Architecture Experiments ----
    print("\n[STEP 5] Running architecture experiments (Bonus)...")
    run_experiments(text, X, y, char_to_idx, idx_to_char, vocab_size)

    print("\n" + "=" * 70)
    print("  ALL DONE! Check the generated text samples above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
