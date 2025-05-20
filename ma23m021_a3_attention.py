import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import wandb
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

#logging into wandb............................
wandb.login(key='b5d1fbca9d5170f54415e9c5a70ef09cee7a0aec')

# Encoder class....................................
class InputEncoder(nn.Module):
    #initializing.....................................................
    def __init__(self, vocab_size, embedding_size, hidden_size, layers, rnn_type='LSTM', dropout_rate=0.2, is_bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_class(embedding_size, hidden_size, layers, dropout=dropout_rate, batch_first=True, bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = layers
    # forward pass...............................
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)  # output: [B, T, H*num_directions]
        return output, hidden  # Return all outputs for attention

# Attention class................................................................
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: [B, H], encoder_outputs: [B, T, H]
        timestep = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, timestep, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B, T, H]
        energy = energy @ self.v  # [B, T]
        return F.softmax(energy, dim=1)

# Decoder class....................................................................................
class OutputDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layers, rnn_type='LSTM', dropout_rate=0.2, is_bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_class(embedding_size + hidden_size, hidden_size, layers, dropout=dropout_rate, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = layers

    def forward(self, token, hidden, encoder_outputs):
        token = token.unsqueeze(1)
        embedded = self.embedding(token)  # [B, 1, E]

        if self.rnn_type == 'LSTM':
            h = hidden[0][-1]  # last layer hidden
        else:
            h = hidden[-1]  # last layer hidden

        attn_weights = self.attention(h, encoder_outputs)  # [B, T]
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B, 1, H]

        rnn_input = torch.cat((embedded, attn_applied), dim=2)  # [B, 1, E+H]
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.output_layer(output.squeeze(1))
        #if return_attention:
          #  return output, hidden, attn_weights
        return output, hidden

# model for encoding-decdoing sequences................................................................
class TransliterationModelattention(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_size, hidden_size, enc_layers, dec_layers,
                 rnn_type='LSTM', dropout_rate=0.2, is_bidirectional=False):
        super().__init__()
        self.encoder = InputEncoder(input_vocab_size, embedding_size, hidden_size, enc_layers, rnn_type, dropout_rate, is_bidirectional)
        self.decoder = OutputDecoder(output_vocab_size, embedding_size, hidden_size * (2 if is_bidirectional else 1),
                                     dec_layers, rnn_type, dropout_rate, is_bidirectional=False)  # decoder not bidirectional
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.is_bidirectional = is_bidirectional

    def forward(self, source, target, teacher_forcing_prob=0.5):
        batch_size, target_len = target.size()
        output_vocab_size = self.decoder.output_layer.out_features
        predictions = torch.zeros(batch_size, target_len, output_vocab_size, device=source.device)

        encoder_outputs, encoder_hidden = self.encoder(source)

        def merge_bidirectional(state):
            return torch.cat([state[i::2] for i in range(2)], dim=2)

        def match_layers(state, required_layers):
            actual_layers = state.size(0)
            if actual_layers == required_layers:
                return state
            elif actual_layers > required_layers:
                return state[:required_layers]
            else:
                pad = torch.zeros(required_layers - actual_layers, *state.shape[1:], device=state.device)
                return torch.cat([state, pad], dim=0)

        if self.rnn_type == 'LSTM':
            h, c = encoder_hidden
            if self.encoder.is_bidirectional:
                h, c = merge_bidirectional(h), merge_bidirectional(c)
            h = match_layers(h, self.decoder.rnn.num_layers)
            c = match_layers(c, self.decoder.rnn.num_layers)
            decoder_hidden = (h, c)
        else:
            h = encoder_hidden
            if self.encoder.is_bidirectional:
                h = merge_bidirectional(h)
            h = match_layers(h, self.decoder.rnn.num_layers)
            decoder_hidden = h

        decoder_input = target[:, 0]
        for t in range(1, target_len):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            predictions[:, t] = output
            top1 = output.argmax(1)
            decoder_input = target[:, t] if random.random() < teacher_forcing_prob else top1

        return predictions

# Vocabulary and creating batch.......................................................
def build_vocab_and_prepare_batch(seqs, device):
    special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}

    # Extract unique characters from Latin and Devanagari sequences.........................
    unique_chars_latin = sorted(set(ch for seq in seqs for ch in seq[0]))
    unique_chars_dev = sorted(set(ch for seq in seqs for ch in seq[1]))

    # Build vocabularies and reverse mappings....................................................
    src_vocab = {ch: idx+len(special_tokens) for idx, ch in enumerate(unique_chars_latin)}
    src_vocab.update(special_tokens)
    tgt_vocab = {ch: idx+len(special_tokens) for idx, ch in enumerate(unique_chars_dev)}
    tgt_vocab.update(special_tokens)
    idx2src = {idx: ch for ch, idx in src_vocab.items()}
    idx2tgt = {idx: ch for ch, idx in tgt_vocab.items()}

    def encode_text(seq, vocab):
        return [vocab.get(ch, vocab['<unk>']) for ch in seq]

    # creating batches........................................................................
    def create_batch(pairs):
        src = [torch.tensor(encode_text(x, src_vocab) + [src_vocab['<eos>']]) for x, _ in pairs]
        tgt = [torch.tensor([tgt_vocab['<sos>']] + encode_text(y, tgt_vocab) + [tgt_vocab['<eos>']]) for _, y in pairs]
        src = pad_sequence(src, batch_first=True, padding_value=src_vocab['<pad>'])
        tgt = pad_sequence(tgt, batch_first=True, padding_value=tgt_vocab['<pad>'])
        return src.to(device), tgt.to(device)

    return src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch, unique_chars_latin, unique_chars_dev

def read_pairs(file_path):
    with open(file_path, encoding='utf-8') as f:
        return [(line.split('\t')[1], line.split('\t')[0]) for line in f.read().strip().split('\n') if '\t' in line]

# function for computing word-level accuracy................................................
def compute_word_level_accuracy(preds, targets, vocab):
    sos, eos, pad = vocab['<sos>'], vocab['<eos>'], vocab['<pad>']
    preds = preds.tolist()
    targets = targets.tolist()
    correct = 0
    for p, t in zip(preds, targets):
        p = [x for x in p if x != pad and x != eos]
        t = [x for x in t if x != pad and x != eos]
        if p == t:
            correct += 1
    return correct / len(preds) * 100

# Training Function.....................................................................
def run_training():
    wandb.init(config={
        "embedding_size": 128,
        "hidden_size": 256,
        "enc_layers": 2,
        "dec_layers": 2,
        "rnn_type": "LSTM",
        "dropout_rate": 0.2,
        "epochs": 10,
        "batch_size": 64,
        "is_bidirectional": False,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "teacher_forcing_prob": 0.5
    })
    cfg = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # takig the path.......................................................
    train_path = "/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    dev_path = "/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
    train_set = read_pairs(train_path)
    dev_set = read_pairs(dev_path)

    # src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch = build_vocab_and_prepare_batch(train_set, device)
    src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch,unique_chars_latin, unique_chars_dev = build_vocab_and_prepare_batch(train_set, device)

    # model creation .....................................................................................
    model = TransliterationModelattention(len(src_vocab), len(tgt_vocab), cfg.embedding_size, cfg.hidden_size,
                                 cfg.enc_layers, cfg.dec_layers, cfg.rnn_type, cfg.dropout_rate, cfg.is_bidirectional).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    epochs = cfg.epochs if isinstance(cfg.epochs, int) else cfg.epochs[0]
    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0
        random.shuffle(train_set)

        for i in range(0, len(train_set), cfg.batch_size):
            batch = train_set[i:i+cfg.batch_size]
            src, tgt = create_batch(batch)

            optimizer.zero_grad()
            outputs = model(src, tgt, cfg.teacher_forcing_prob)
            loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))

            preds = outputs.argmax(-1)
            acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += acc

        avg_train_loss = total_loss / (len(train_set) // cfg.batch_size)
        avg_train_acc = total_acc / (len(train_set) // cfg.batch_size)

        # evaluation module................................................................
        model.eval()
        dev_loss, dev_acc = 0, 0
        printed = 0
        with torch.no_grad():
            for i in range(0, len(dev_set), cfg.batch_size):
                batch = dev_set[i:i+cfg.batch_size]
                src, tgt = create_batch(batch)
                outputs = model(src, tgt, 0)
                loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))

                preds = outputs.argmax(-1)
                acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

                dev_loss += loss.item()
                dev_acc += acc
        # validation loss and accuracy..............................................
        avg_dev_loss = dev_loss / (len(dev_set) // cfg.batch_size)
        avg_dev_acc = dev_acc / (len(dev_set) // cfg.batch_size)

        wandb.log({
            "Train Loss": avg_train_loss,
            "Train Accuracy": avg_train_acc,
            "Validation Loss": avg_dev_loss,
            "Validation Accuracy": avg_dev_acc,
            "Epoch": epoch + 1
        })

        print(f"Epoch {epoch + 1}/{cfg.epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}% | Val Loss: {avg_dev_loss:.4f}, Val Acc: {avg_dev_acc:.2f}%")

    wandb.finish()
    return model

# sweep configuration...........................................
sweep_config = {
    'method': 'random',
    'metric': {'name': 'Validation Accuracy', 'goal': 'maximize'},
    'parameters': {
        'embed_dim': {
            'values': [32, 64, 128, 256]
        },
        'hidden_dim': {
            'values': [64, 128, 256]
        },
        'enc_layers': {
            'values': [1,2,3]
        },
        'dec_layers': {
            'values': [1,2,3]
        },
        'cell_type': {
            'values': ['GRU', 'LSTM','RNN']
        },
        'dropout': {
            'values': [0.2, 0.3]
        },
        'batch_size': {
            'values': [32, 64]
        },
        'epochs': {
            'values': [5,10]

        },
        'bidirectional': {
            'values': [False, True]
        },
        'learning_rate': {
            'values': [0.001, 0.002, 0.001]
        },
        'optimizer': {
            'values': ['adam', 'nadam']
        },
        'teacher_forcing_ratio': {
            'values': [0.2, 0.5, 0.7]
        },
        'beam_width': {
            'values': [1, 3, 5]
        }
    }
}

#import wandb
#wandb.login(key='b5d1fbca9d5170f54415e9c5a70ef09cee7a0aec')
sweep_id = wandb.sweep(sweep_config, project="MA23M021_A3_Attention")
wandb.agent(sweep_id, function=run_training, count = 50)

import torch
import torch.nn as nn
import random
import wandb
import torch.nn.functional as F
import torch.optim as optim

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


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
    "beam_width":3,
    "teacher_forcing_ratio": 0.5
}

def training_test(best_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_path = "/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    test_path = "/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
    train_set = read_pairs(train_path)
    test_set = read_pairs(test_path)

    # src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch = build_vocab_and_prepare_batch(train_set, device)
    src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch,unique_chars_latin, unique_chars_dev = build_vocab_and_prepare_batch(train_set, device)
    model = TransliterationModelattention(len(src_vocab), len(tgt_vocab), best_config["embed_dim"],best_config["hidden_dim"],
                                 best_config["enc_layers"], best_config["dec_layers"], best_config["cell_type"], best_config["dropout"], best_config["bidirectional"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    for epoch in range(best_config["epochs"]):
        model.train()
        total_loss, total_acc = 0, 0
        random.shuffle(train_set)

        for i in range(0, len(train_set), best_config["batch_size"]):
            batch = train_set[i:i+best_config["batch_size"]]
            src, tgt = create_batch(batch)


            optimizer.zero_grad()
            outputs = model(src, tgt, best_config["teacher_forcing_ratio"])
            loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))

            preds = outputs.argmax(-1)
            acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += acc

        avg_train_loss = total_loss / (len(train_set) // best_config["batch_size"])
        avg_train_acc = total_acc / (len(train_set) // best_config["batch_size"])

        model.eval()
        test_loss, test_acc = 0, 0
        printed = 0
        with torch.no_grad():
            for i in range(0, len(test_set), best_config["batch_size"]):
                batch = test_set[i:i+ best_config["batch_size"]]
                src, tgt = create_batch(batch)
                outputs = model(src, tgt, 0)
                loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))


                preds = outputs.argmax(-1)
                acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

                test_loss += loss.item()
                test_acc += acc

                if printed < 5:
                    for j in range(min(3, src.size(0))):
                        input_seq = ''.join([idx2src.get(idx.item(), '<unk>') for idx in src[j] if idx.item() not in [src_vocab['<pad>'], src_vocab['<eos>']]])
                        target_seq = ''.join([idx2tgt.get(idx.item(), '<unk>') for idx in tgt[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]])
                        pred_seq = ''.join([idx2tgt.get(idx.item(), '<unk>') for idx in preds[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]])
                        print(f"\n Input:{input_seq} | Target:{target_seq} | Predicted:{pred_seq}")
                        print("-" * 40)
                        printed += 1

        avg_test_loss = test_loss / (len(test_set) // best_config["batch_size"])
        avg_test_acc = test_acc / (len(test_set) // best_config["batch_size"])


    print(f" test Loss: {avg_test_loss:.4f}, test Acc: {avg_test_acc:.2f}%")


training_test(best_config)

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
    "beam_width":3,
    "teacher_forcing_ratio": 0.5
}


import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import random
import matplotlib.pyplot as plt
import pandas as pd

def training_test(best_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = "/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    test_path = "/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
    train_set = read_pairs(train_path)
    test_set = read_pairs(test_path)

    src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch, unique_chars_latin, unique_chars_dev = build_vocab_and_prepare_batch(train_set, device)

    model = TransliterationModelattention(
        len(src_vocab), len(tgt_vocab),
        best_config["embed_dim"], best_config["hidden_dim"],
        best_config["enc_layers"], best_config["dec_layers"],
        best_config["cell_type"], best_config["dropout"],
        best_config["bidirectional"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    os.makedirs("predictions_attention", exist_ok=True)

    for epoch in range(best_config["epochs"]):
        model.train()
        total_loss, total_acc = 0, 0
        random.shuffle(train_set)

        for i in range(0, len(train_set), best_config["batch_size"]):
            batch = train_set[i:i + best_config["batch_size"]]
            src, tgt = create_batch(batch)

            optimizer.zero_grad()
            outputs = model(src, tgt, best_config["teacher_forcing_ratio"])
            loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))

            preds = outputs.argmax(-1)
            acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

        avg_train_loss = total_loss / (len(train_set) // best_config["batch_size"])
        avg_train_acc = total_acc / (len(train_set) // best_config["batch_size"])

        model.eval()
        test_loss, test_acc = 0, 0
        predictions = []
        printed = 0


        with torch.no_grad():
            for i in range(0, len(test_set), best_config["batch_size"]):
                batch = test_set[i:i + best_config["batch_size"]]
                src, tgt = create_batch(batch)
                outputs = model(src, tgt, 0)

                loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
                preds = outputs.argmax(-1)
                acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

                test_loss += loss.item()
                test_acc += acc

                for j in range(src.size(0)):
                    input_seq = ''.join([idx2src.get(idx.item(), '') for idx in src[j] if idx.item() not in [src_vocab['<pad>'], src_vocab['<eos>']]])
                    target_seq = ''.join([idx2tgt.get(idx.item(), '') for idx in tgt[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]])
                    pred_seq = ''.join([idx2tgt.get(idx.item(), '') for idx in preds[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]])
                    predictions.append({'Input': input_seq, 'Target': target_seq, 'Predicted': pred_seq})




        avg_test_loss = test_loss / (len(test_set) // best_config["batch_size"])
        avg_test_acc = test_acc / (len(test_set) // best_config["batch_size"])

    # Save all predictions to CSV................................
    os.makedirs("predictions_vanilla", exist_ok=True)
    with open("predictions_vanilla/test_predictions.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Input', 'Target', 'Predicted'])
        writer.writeheader()
        writer.writerows(predictions)

   # Print 3x3 sample grid....................
    sample_df = pd.DataFrame(predictions[:9])
    print("\nSample Prediction Grid:\n")
    for i, row in sample_df.iterrows():
        print(f"{i+1}. Input: {row['Input']} | Target: {row['Target']} | Predicted: {row['Predicted']}")


training_test(best_config)

"""Heatmap................................................"""

import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import wandb
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

wandb.login(key='b5d1fbca9d5170f54415e9c5a70ef09cee7a0aec')
class InputEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layers, rnn_type='LSTM', dropout_rate=0.2, is_bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_class(embedding_size, hidden_size, layers, dropout=dropout_rate, batch_first=True, bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = layers

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)  # output: [B, T, H*num_directions]
        return output, hidden  # Return all outputs for attention


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [B, H], encoder_outputs: [B, T, H]
        B, T, H = encoder_outputs.shape
        hidden = hidden.unsqueeze(1).repeat(1, T, 1)  # [B, T, H]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B, T, H]
        energy = energy @ self.v  # [B, T]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        return F.softmax(energy, dim=1)  # [B, T]


class OutputDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layers, rnn_type='LSTM', dropout_rate=0.2, is_bidirectional=False, return_attention=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_class(embedding_size + hidden_size, hidden_size, layers, dropout=dropout_rate, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size)  # Using the updated attention mechanism
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.return_attention = return_attention

    def forward(self, token, hidden, encoder_outputs, mask=None):
        token = token.unsqueeze(1)  # [B, 1]
        embedded = self.embedding(token)  # [B, 1, E]

        # Getting the last hidden state from the encoder
        if self.rnn_type == 'LSTM':
            h = hidden[0][-1]  # last layer hidden state
        else:
            h = hidden[-1]  # last layer hidden state

        # Applying attention mechanism with optional mask
        attn_weights = self.attention(h, encoder_outputs, mask)  # [B, T]
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B, 1, H]

        # Concatenating the attention-applied vector with the embedded token
        rnn_input = torch.cat((embedded, attn_applied), dim=2)  # [B, 1, E+H]

        # Passing the concatenated input through the RNN
        output, hidden = self.rnn(rnn_input, hidden)

        # Passing the output through the output layer
        output = self.output_layer(output.squeeze(1))  # [B, vocab_size]

        # Returning the output, hidden state, and attention weights (if needed)
        if self.return_attention:
            return output, hidden, attn_weights
        return output, hidden


class TransliterationModelattention(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_size, hidden_size, enc_layers, dec_layers,
                 rnn_type='LSTM', dropout_rate=0.2, is_bidirectional=False):
        super().__init__()
        self.encoder = InputEncoder(input_vocab_size, embedding_size, hidden_size, enc_layers, rnn_type, dropout_rate, is_bidirectional)
        self.decoder = OutputDecoder(output_vocab_size, embedding_size, hidden_size * (2 if is_bidirectional else 1),
                                     dec_layers, rnn_type, dropout_rate, is_bidirectional=False, return_attention= True)  # decoder not bidirectional
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.is_bidirectional = is_bidirectional


    def forward(self, source, target, teacher_forcing_prob=0.5,mask = None):
        batch_size, target_len = target.size()
        output_vocab_size = self.decoder.output_layer.out_features
        predictions = torch.zeros(batch_size, target_len, output_vocab_size, device=source.device)
        all_attentions = torch.zeros(batch_size, target_len, source.size(1), device=source.device)

        encoder_outputs, encoder_hidden = self.encoder(source)

        def merge_bidirectional(state):
            return torch.cat([state[i::2] for i in range(2)], dim=2)

        def match_layers(state, required_layers):
            actual_layers = state.size(0)
            if actual_layers == required_layers:
                return state
            elif actual_layers > required_layers:
                return state[:required_layers]
            else:
                pad = torch.zeros(required_layers - actual_layers, *state.shape[1:], device=state.device)
                return torch.cat([state, pad], dim=0)

        if self.rnn_type == 'LSTM':
            h, c = encoder_hidden
            if self.encoder.is_bidirectional:
                h, c = merge_bidirectional(h), merge_bidirectional(c)
            h = match_layers(h, self.decoder.rnn.num_layers)
            c = match_layers(c, self.decoder.rnn.num_layers)
            decoder_hidden = (h, c)
        else:
            h = encoder_hidden
            if self.encoder.is_bidirectional:
                h = merge_bidirectional(h)
            h = match_layers(h, self.decoder.rnn.num_layers)
            decoder_hidden = h

        decoder_input = target[:, 0]
        for t in range(1, target_len):
            output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs,mask)
            predictions[:, t] = output
            all_attentions[:, t] = attn_weights  # store attention weights for step t
            top1 = output.argmax(1)
            decoder_input = target[:, t] if random.random() < teacher_forcing_prob else top1

        return predictions, all_attentions

def build_vocab_and_prepare_batch(seqs, device):
    special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    unique_chars_latin = sorted(set(ch for seq in seqs for ch in seq[0]))
    unique_chars_dev = sorted(set(ch for seq in seqs for ch in seq[1]))

    src_vocab = {ch: idx+len(special_tokens) for idx, ch in enumerate(unique_chars_latin)}
    src_vocab.update(special_tokens)
    tgt_vocab = {ch: idx+len(special_tokens) for idx, ch in enumerate(unique_chars_dev)}
    tgt_vocab.update(special_tokens)
    idx2src = {idx: ch for ch, idx in src_vocab.items()}
    idx2tgt = {idx: ch for ch, idx in tgt_vocab.items()}

    def encode_text(seq, vocab):
        return [vocab.get(ch, vocab['<unk>']) for ch in seq]

    def create_batch(pairs):
        src = [torch.tensor(encode_text(x, src_vocab) + [src_vocab['<eos>']]) for x, _ in pairs]
        tgt = [torch.tensor([tgt_vocab['<sos>']] + encode_text(y, tgt_vocab) + [tgt_vocab['<eos>']]) for _, y in pairs]

        src = pad_sequence(src, batch_first=True, padding_value=src_vocab['<pad>'])
        tgt = pad_sequence(tgt, batch_first=True, padding_value=tgt_vocab['<pad>'])


def plot_attention_grid(attentions, src_tokens, tgt_tokens, idx2src, idx2tgt, src_vocab, tgt_vocab):
    font_path = "/kaggle/input/noto-sans/static/NotoSansDevanagari-Regular.ttf"
    devanagari_font = fm.FontProperties(fname=font_path)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i in range(9):
        ax = axes[i]
        attn = attentions[i].cpu().numpy()

        # Mask out invalid values (e.g., -inf)
        mask = attn == -float('inf')
        attn[mask] = 0  # Replace invalid values with zeros

        input_chars = [idx2src[idx.item()] for idx in src_tokens[i]
                       if idx.item() not in [src_vocab['<pad>'], src_vocab['<eos>'], src_vocab['<sos>']]]
        target_chars = [idx2tgt[idx.item()] for idx in tgt_tokens[i]
                        if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>'], tgt_vocab['<sos>']]]

        # sns.heatmap(
        #     attn[1:len(target_chars)+1, :len(input_chars)],
        #     xticklabels=input_chars,
        #     yticklabels=target_chars,

        #     cmap="viridis",
        #     ax=ax,
        #     cbar=False
        # )

        ax.set_xlabel("Source (Latin)", fontproperties=devanagari_font)
        ax.set_ylabel("Target (Devanagari)", fontproperties=devanagari_font)
        ax.set_title(f"Sample {i+1}", fontproperties=devanagari_font)

        for label in ax.get_xticklabels():
            label.set_fontproperties(devanagari_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(devanagari_font)

        ax.tick_params(axis='x', labelrotation=90)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    # plt.savefig("plots/attention_heatmap_grid.png", bbox_inches="tight")
    # plt.show()

import os
import csv
import random
import torch
import torch.optim as optim
import torch.nn as nn

# Define the best_config
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

# Create the folder if it doesn't exist
os.makedirs("predictions_attention", exist_ok=True)

def training_test(best_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = "/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    test_path = "/kaggle/input/dakshina-dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
    train_set = read_pairs(train_path)
    test_set = read_pairs(test_path)

    # Prepare vocabulary and batch creation
    src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch, unique_chars_latin, unique_chars_dev = build_vocab_and_prepare_batch(train_set, device)

    # Initialize the model
    model = TransliterationModelattention(len(src_vocab), len(tgt_vocab), best_config["embed_dim"], best_config["hidden_dim"],
                                         best_config["enc_layers"], best_config["dec_layers"], best_config["cell_type"], best_config["dropout"], best_config["bidirectional"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    # Open CSV file to save predictions
    with open("predictions_attention/predictions.csv", "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ['Input', 'Target', 'Predicted']  # CSV headers
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for epoch in range(best_config["epochs"]):
            model.train()
            total_loss, total_acc = 0, 0
            random.shuffle(train_set)


            for i in range(0, len(train_set), best_config["batch_size"]):
                batch = train_set[i:i + best_config["batch_size"]]
                src, tgt = create_batch(batch)
                mask = (src != 0).float()
                optimizer.zero_grad()
                outputs,_ = model(src, tgt, best_config["teacher_forcing_ratio"],mask)
                loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))

                preds = outputs.argmax(-1)
                acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_acc += acc

            avg_train_loss = total_loss / (len(train_set) // best_config["batch_size"])
            avg_train_acc = total_acc / (len(train_set) // best_config["batch_size"])

            model.eval()
            test_loss, test_acc = 0, 0
            printed = 0
            with torch.no_grad():
                for i in range(0, len(test_set), best_config["batch_size"]):
                    batch = test_set[i:i + best_config["batch_size"]]
                    src, tgt = create_batch(batch)
                    outputs,attn_weights = model(src, tgt, 0)

                    # if printed == 0:
                    #     # Save first 9 samples for heatmap
                    #     plot_attention_grid(attn_weights[:9], src[:9], tgt[:9], idx2src, idx2tgt)
                    #     printed += 9

                    loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))

                    preds = outputs.argmax(-1)
                    acc = compute_word_level_accuracy(preds[:, 1:], tgt[:, 1:], tgt_vocab)

                    test_loss += loss.item()
                    test_acc += acc

            # # Save predictions to

            avg_test_loss = test_loss / (len(test_set) // best_config["batch_size"])
            avg_test_acc = test_acc / (len(test_set) // best_config["batch_size"])
            # Save predictions to CSV file
            for j in range(min(4, src.size(0))):  # Print the first 3 samples
                input_seq = ''.join([idx2src.get(idx.item(), '<unk>') for idx in src[j] if idx.item() not in [src_vocab['<pad>'], src_vocab['<eos>'], src_vocab['<sos>']]])
                target_seq = ''.join([idx2tgt.get(idx.item(), '<unk>') for idx in tgt[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>'],src_vocab['<sos>']]])
                pred_seq = ''.join([idx2tgt.get(idx.item(), '<unk>') for idx in preds[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>'],src_vocab['<sos>']]])

                # Write each row to the CSV file
                writer.writerow({'Input': input_seq, 'Target': target_seq, 'Predicted': pred_seq})
                print({'Input': input_seq, 'Target': target_seq, 'Predicted': pred_seq})
        plot_attention_grid(attn_weights[:9], src[:9], tgt[:9], idx2src, idx2tgt,src_vocab,tgt_vocab)


    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%")

# Call the training_test function
training_test(best_config)



