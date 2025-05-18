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

This is a character-level sequence-to-sequence model that learns to map Latin-script Hindi words to their corresponding Devanagari representation. The architecture consists of:
- **Encoder**: An RNN-based model that processes the input sequence.
- **Decoder**: An RNN-based model that generates the output sequence.
- **Attention Mechanism** (optional): Helps the model focus on specific parts of the input sequence while generating each token of the output.

 Key features:
- Flexible RNN cell types: **RNN**, **GRU**, or **LSTM**
- Optionally **bidirectional** encoder
- **Teacher forcing** for improved learning
- Trained using **CrossEntropyLoss** with padding mask
- Evaluated using **word-level accuracy**
- Optionally visualizes **attention weights** (if attention is included)

---

## Functionality::

### `Build_vocab_and_prepare_batch(train_set, device)`

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
* The training process uses **teacher forcing** and **backpropagation through time (BPTT)** to train the model. Weights and biases are logged during training and testing, allowing for analysis of performance.
---


### Attention Mechanism Integration

The `attention-based Seq2Seq` model enhances the translation process by focusing on relevant parts of the input sequence during the decoding phase. The attention mechanism calculates attention weights, which indicate the importance of each input token for generating the corresponding output token. This allows the model to handle longer sequences more effectively and improve the accuracy of transliterations.

The attention weights are extracted during inference and visualized using heatmaps, giving insights into which parts of the input the model attended to while generating the output sequence. This visualization helps in understanding the model's decision-making process and interpreting its behavior.


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
---

##  Evaluation

The model is evaluated using **word-level accuracy**, where the predicted word is compared to the actual target word. Additionally, attention weights are visualized if the attention mechanism is included.

### Attention Heatmaps

Attention is implemented, heatmaps of the attention weights are generated to visualize how the model attends to different parts of the input during prediction.

---


### Wandb Report Link:
https://wandb.ai/ma23m021-iit-madras/MA23M021_A3_seq2seq/reports/MA23M021-Assignment-3--VmlldzoxMjgyMzA4Mw?accessToken=d6xibilzbyh24b2feph3uttysjcq55k9y77bg3gyns2iyf54qhg92cv0f3vff8lh


## How to Run

I have given ```MA23M021_A3_Seq2Seq.py``` and ```train_Seq2Seq.py``` files. And I have given ```MA23M021_A3_Attention.py``` and ```train_Attention.py``` files. These files have to be in same directory after downloading as I am importing the functions from this file in respective train files. You just have to change the address of data in the required location in respective train files of Vanilla and Attention. And use this command to run the code finally:
```
python train_Seq2Seq.py 
```
and

```
python train_Attention.py 
```
for both the files.
