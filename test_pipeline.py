"""
Quick end-to-end test of lstm_text_generation.py pipeline.
Uses tiny data (5000 chars) and minimal epochs (2) to verify
all code paths work without errors in ~1-2 minutes.
"""

import os
import sys
import re
import string
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ---- Import functions from main script ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm_text_generation import (
    download_dataset,
    preprocess_text,
    build_char_mappings,
    create_sequences,
    build_model,
    train_model,
    sample_with_temperature,
    generate_text,
)

DATASET_URL = "https://www.gutenberg.org/files/100/100-0.txt"
DATASET_PATH = "shakespeare.txt"

def test_pipeline():
    print("=" * 60)
    print("  QUICK PIPELINE TEST (tiny data, 2 epochs)")
    print("=" * 60)

    errors = []

    # ---- Test 1: Download / Load dataset ----
    print("\n[TEST 1] Dataset loading...")
    try:
        raw_text = download_dataset(DATASET_URL, DATASET_PATH)
        assert len(raw_text) > 1000, f"Dataset too small: {len(raw_text)}"
        print(f"  PASS — loaded {len(raw_text):,} chars")
    except Exception as e:
        errors.append(f"Dataset loading: {e}")
        print(f"  FAIL — {e}")
        return errors  # Can't continue without data

    # ---- Test 2: Preprocessing ----
    print("\n[TEST 2] Text preprocessing...")
    try:
        text = preprocess_text(raw_text, max_length=5000)
        assert len(text) > 0, "Preprocessed text is empty"
        assert text == text.lower(), "Text not lowercase"
        # Check no standard punctuation (except apostrophes)
        bad_chars = [c for c in text if c in string.punctuation.replace("'", "") and c != "'"]
        assert len(bad_chars) == 0, f"Punctuation found: {set(bad_chars)}"
        print(f"  PASS — {len(text):,} chars, lowercase, no punctuation")
    except Exception as e:
        errors.append(f"Preprocessing: {e}")
        print(f"  FAIL — {e}")
        return errors

    # ---- Test 3: Character mappings ----
    print("\n[TEST 3] Character mappings...")
    try:
        chars, char_to_idx, idx_to_char, vocab_size = build_char_mappings(text)
        assert vocab_size > 10, f"Vocab too small: {vocab_size}"
        assert len(char_to_idx) == vocab_size
        assert len(idx_to_char) == vocab_size
        # Roundtrip test
        for ch in chars:
            assert idx_to_char[char_to_idx[ch]] == ch, f"Roundtrip failed for '{ch}'"
        print(f"  PASS — {vocab_size} unique chars, roundtrip OK")
    except Exception as e:
        errors.append(f"Char mappings: {e}")
        print(f"  FAIL — {e}")
        return errors

    # ---- Test 4: Sequence creation ----
    print("\n[TEST 4] Sequence creation...")
    seq_length = 40  # Shorter for test
    step_size = 10   # Larger step for test
    try:
        X, y = create_sequences(text, seq_length, step_size, char_to_idx, vocab_size)
        assert X.shape[1] == seq_length, f"X seq dim wrong: {X.shape[1]} != {seq_length}"
        assert y.shape[1] == vocab_size, f"y vocab dim wrong: {y.shape[1]} != {vocab_size}"
        assert X.shape[0] == y.shape[0], "X and y sample count mismatch"
        assert X.shape[0] > 10, f"Too few sequences: {X.shape[0]}"
        print(f"  PASS — X: {X.shape}, y: {y.shape}")
    except Exception as e:
        errors.append(f"Sequence creation: {e}")
        print(f"  FAIL — {e}")
        return errors

    # ---- Test 5: Model building ----
    print("\n[TEST 5] Model building (base)...")
    try:
        model = build_model(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embedding_dim=32,
            lstm_units=64,
            num_lstm_layers=1,
            dropout_rate=0.2,
            learning_rate=0.001,
            model_name="test_base"
        )
        model.summary()
        assert model.output_shape[-1] == vocab_size
        print(f"  PASS — output shape: {model.output_shape}")
    except Exception as e:
        errors.append(f"Model building (base): {e}")
        print(f"  FAIL — {e}")
        return errors

    # ---- Test 5b: Deeper model ----
    print("\n[TEST 5b] Model building (2-layer LSTM)...")
    try:
        deep_model = build_model(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embedding_dim=32,
            lstm_units=64,
            num_lstm_layers=2,
            dropout_rate=0.2,
            learning_rate=0.001,
            model_name="test_deeper"
        )
        deep_model.summary()
        print(f"  PASS — output shape: {deep_model.output_shape}")
    except Exception as e:
        errors.append(f"Model building (deeper): {e}")
        print(f"  FAIL — {e}")

    # ---- Test 6: Training ----
    print("\n[TEST 6] Model training (2 epochs)...")
    try:
        history = train_model(
            model, X, y,
            batch_size=32,
            epochs=2,
            validation_split=0.1,
            checkpoint_path="test_best_model.keras"
        )
        assert "loss" in history.history, "No loss in history"
        assert "val_loss" in history.history, "No val_loss in history"
        assert len(history.history["loss"]) > 0, "No training happened"
        final_loss = history.history["loss"][-1]
        print(f"  PASS — trained {len(history.history['loss'])} epochs, final loss: {final_loss:.4f}")
    except Exception as e:
        errors.append(f"Training: {e}")
        print(f"  FAIL — {e}")
        return errors

    # ---- Test 7: Temperature sampling ----
    print("\n[TEST 7] Temperature sampling...")
    try:
        dummy_probs = np.array([0.1, 0.3, 0.05, 0.5, 0.05])
        for temp in [0.5, 1.0, 1.5]:
            idx = sample_with_temperature(dummy_probs, temp)
            assert 0 <= idx < len(dummy_probs), f"Index out of range: {idx}"
        print(f"  PASS — sampling works for temps [0.5, 1.0, 1.5]")
    except Exception as e:
        errors.append(f"Temperature sampling: {e}")
        print(f"  FAIL — {e}")

    # ---- Test 8: Text generation ----
    print("\n[TEST 8] Text generation...")
    try:
        seed = text[:seq_length]
        for temp in [0.5, 1.0, 1.2]:
            generated = generate_text(
                model, seed, char_to_idx, idx_to_char,
                seq_length, length=100, temperature=temp
            )
            assert len(generated) == seq_length + 100, \
                f"Wrong length: {len(generated)} (expected {seq_length + 100})"
            assert isinstance(generated, str), "Generated is not a string"
        print(f"  PASS — generated text at 3 temperatures")
        print(f"  Sample (temp=1.0): \"{generated[seq_length:seq_length+80]}...\"")
    except Exception as e:
        errors.append(f"Text generation: {e}")
        print(f"  FAIL — {e}")

    # ---- Cleanup test checkpoint ----
    for f in ["test_best_model.keras"]:
        if os.path.exists(f):
            os.remove(f)

    # ---- Summary ----
    print("\n" + "=" * 60)
    if errors:
        print(f"  FAILED — {len(errors)} error(s):")
        for err in errors:
            print(f"    ✗ {err}")
    else:
        print("  ALL 8 TESTS PASSED ✓")
    print("=" * 60)

    return errors


if __name__ == "__main__":
    errors = test_pipeline()
    sys.exit(1 if errors else 0)
