# DA6401_MA23M021_Assignment-3: Sequence-to-Sequence Transliteration using RNNs
---

##  Problem Statement

The task is to build a character-level sequence-to-sequence transliteration model that converts Latin-script Hindi words (e.g., "ghar") into native Devanagari script (e.g., "घर"). This involves training an encoder-decoder architecture using RNNs on the **Dakshina dataset**.

---

## Dataset Used

**Dakshina Dataset v1.0**  
- Source: [Google Research Datasets](https://github.com/google-research-datasets/dakshina)
- We use:
  - `hi.translit.sampled.train.tsv` for training
  - `hi.translit.sampled.test.tsv` for evaluation

Each line in the dataset contains:  
- Column 1: The Hindi word in Devanagari (target)  
- Column 2: The Latin-script transliteration (input)  
- Column 3: Frequency count (ignored in this task)

---

## Model Overview

This project implements a sequence-to-sequence (Seq2Seq) character-level transliteration model using PyTorch. It is designed to transliterate words from Latin-script Hindi to Devanagari using RNN-based architectures. The model supports configurable RNN cells (RNN, GRU, LSTM), bidirectional encoders, dropout, and attention mechanisms.

The model is a character-level encoder-decoder built using PyTorch. Key features:
- Flexible RNN cell types: **RNN**, **GRU**, or **LSTM**
- Optionally **bidirectional** encoder
- **Teacher forcing** for improved learning
- Trained using **CrossEntropyLoss** with padding mask
- Evaluated using **word-level accuracy**
- Optionally visualizes **attention weights** (if attention is included)

---

## Functionality::

### `build_vocab_and_prepare_batch(train_set, device)`

* Creates character-level vocabularies and index mappings for both scripts.Returns vocab dictionaries and a batch creation function.
* Converts a batch of string pairs into padded tensors for training.Returns source and target tensors with appropriate token indices.

### `TransliterationModel(...)`

* Defines the Seq2Seq model architecture with encoder and decoder.
* Supports LSTM/GRU/RNN cells, dropout, and bidirectional encoders.

### `compute_word_level_accuracy(preds, targets, vocab)`

* Measures word-level transliteration accuracy (exact match of sequences).
* Ignores `<pad>` and `<eos>` tokens during comparison.

### `training_test(best_config)`

* Trains the transliteration model using the given configuration dictionary.
* Evaluates on the test set and saves predictions to a CSV file.

---

##  Output

* After training, the model prints a 3×3 grid of sample test predictions.
* Full predictions are uploaded in github for both Vanilla and Attention.

---


##  Best Hyperparameter Configuration

```python
best_config = {
    "embed_dim": 128,
    "hidden_dim": 256,
    "enc_layers": 3,
    "dec_layers": 3,
    "cell_type": "LSTM",
    "dropout": 0.2,
    "batch_size": 64,
    "bidirectional": False,
    "learning_rate": 0.001,
    "epochs": 10,
    "teacher_forcing_ratio": 0.5
}
````



