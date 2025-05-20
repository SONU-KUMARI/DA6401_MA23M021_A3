import torch
import torch.nn as nn
import random
import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoder module............................................................
class InputEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layers, rnn_type='LSTM', dropout_rate=0.2, is_bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.is_bidirectional = is_bidirectional
        self.rnn_type = rnn_type
        self.num_directions = 2 if is_bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = layers

        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_class(
            embedding_size,
            hidden_size // self.num_directions,
            layers,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=is_bidirectional
        )
# forward pass........................................
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return hidden

# Decoder module...................................................
class OutputDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layers, rnn_type='LSTM', dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = layers
        # model defining.....................................
        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_class(
            embedding_size,
            hidden_size,
            layers,
            dropout=dropout_rate,
            batch_first=True
        )
        # output layer................................
        self.output_layer = nn.Linear(hidden_size, vocab_size)

# forward pass.........................
    def forward(self, token, hidden):
        token = token.unsqueeze(1)
        embedded = self.embedding(token)
        output, hidden = self.rnn(embedded, hidden)
        output = self.output_layer(output.squeeze(1))
        return output, hidden

# Main module..................................................
class TransliterationModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_size, hidden_size, enc_layers, dec_layers,
                 rnn_type='LSTM', dropout_rate=0.2, is_bidirectional=False):
        super().__init__()
        self.encoder = InputEncoder(input_vocab_size, embedding_size, hidden_size, enc_layers, rnn_type, dropout_rate, is_bidirectional)
        self.decoder = OutputDecoder(output_vocab_size, embedding_size, hidden_size, dec_layers, rnn_type, dropout_rate)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.is_bidirectional = is_bidirectional

    # forward pass.................................................................
    def forward(self, source, target, teacher_forcing_prob=0.5):
        batch_size, target_len = target.size()
        output_vocab_size = self.decoder.output_layer.out_features
        predictions = torch.zeros(batch_size, target_len, output_vocab_size, device=source.device)

        encoder_hidden = self.encoder(source)

# bidirectional.....................................................
        def merge_bidirectional(state):
            return torch.cat([state[::2], state[1::2]], dim=2)

        def match_layers(state, required_layers):
            actual_layers = state.size(0)
            if actual_layers == required_layers:
                return state
            elif actual_layers < required_layers:
                pad = torch.zeros(required_layers - actual_layers, *state.shape[1:], device=state.device)
                return torch.cat([state, pad], dim=0)
            else:
                return state[:required_layers]

        if self.rnn_type == 'LSTM':
            h, c = encoder_hidden
            if self.is_bidirectional:
                h = merge_bidirectional(h)
                c = merge_bidirectional(c)
            h = match_layers(h, self.dec_layers)
            c = match_layers(c, self.dec_layers)
            decoder_hidden = (h, c)
        else:
            h = encoder_hidden
            if self.is_bidirectional:
                h = merge_bidirectional(h)
            h = match_layers(h, self.dec_layers)
            decoder_hidden = h

        decoder_input = target[:, 0]
        for t in range(1, target_len):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            predictions[:, t] = output
            top1 = output.argmax(1)
            decoder_input = target[:, t] if random.random() < teacher_forcing_prob else top1

        return predictions

# Data processing and vocabulary.............................
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
        return src.to(device), tgt.to(device)

    return src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch, unique_chars_latin, unique_chars_dev

# input.................................
def read_pairs(file_path):
    with open(file_path, encoding='utf-8') as f:
        return [(line.split('\t')[1], line.split('\t')[0]) for line in f.read().strip().split('\n') if '\t' in line]

# word level accuracy..................................
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

# Training funtion..................................................
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

    train_path = "/kaggle/input/dakshina-data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    dev_path = "/kaggle/input/dakshina-data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
    train_set = read_pairs(train_path)
    dev_set = read_pairs(dev_path)

    # src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch = build_vocab_and_prepare_batch(train_set, device)
    src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch,unique_chars_latin, unique_chars_dev = build_vocab_and_prepare_batch(train_set, device)
    model = TransliterationModel(len(src_vocab), len(tgt_vocab), cfg.embedding_size, cfg.hidden_size,
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

# sweep configuration..............................................
sweep_config = {
    'method': 'bayes',
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
            'values': ['GRU','LSTM','RNN']
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

sweep_id = wandb.sweep(sweep_config, project="MA23M021_A3_seq2seq")
wandb.agent(sweep_id, function=run_training, count = 50)

# best configuration............................
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

# printing the test accuracy after training the best model...............................
def training_test(best_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_path = "/kaggle/input/dakshina-data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    test_path = "/kaggle/input/dakshina-data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
    train_set = read_pairs(train_path)
    test_set = read_pairs(test_path)

    # src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch = build_vocab_and_prepare_batch(train_set, device)
    src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch,unique_chars_latin, unique_chars_dev = build_vocab_and_prepare_batch(train_set, device)
    model = TransliterationModel(len(src_vocab), len(tgt_vocab), best_config["embed_dim"],best_config["hidden_dim"],
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

# evaluating for test data........................................................
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

# calling the function..........................
training_test(best_config)

# plotting the grid and saving to folder.................................

import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def training_test(best_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = "/kaggle/input/dakshina-data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    test_path = "/kaggle/input/dakshina-data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
    train_set = read_pairs(train_path)
    test_set = read_pairs(test_path)

    src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch, unique_chars_latin, unique_chars_dev = build_vocab_and_prepare_batch(train_set, device)

    model = TransliterationModel(len(src_vocab), len(tgt_vocab), best_config["embed_dim"], best_config["hidden_dim"],
                                 best_config["enc_layers"], best_config["dec_layers"], best_config["cell_type"],
                                 best_config["dropout"], best_config["bidirectional"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    # training for number of epochs...........................
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

# evaluating test data.......................................
    model.eval()
    test_loss, test_acc = 0, 0
    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(test_set), best_config["batch_size"]):
            batch = test_set[i:i + best_config["batch_size"]]
            src, tgt = create_batch(batch)
            outputs = model(src, tgt, 0)
            preds = outputs.argmax(-1)

            for j in range(src.size(0)):
                input_seq = ''.join([idx2src.get(idx.item(), '') for idx in src[j] if idx.item() not in [src_vocab['<pad>'], src_vocab['<eos>']]])
                target_seq = ''.join([idx2tgt.get(idx.item(), '') for idx in tgt[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]])
                pred_seq = ''.join([idx2tgt.get(idx.item(), '') for idx in preds[j][1:] if idx.item() not in [tgt_vocab['<pad>'], tgt_vocab['<eos>']]])
                all_predictions.append({'Input': input_seq, 'Target': target_seq, 'Predicted': pred_seq})

    # Save all predictions to CSV................................
    os.makedirs("predictions_vanilla", exist_ok=True)
    with open("predictions_vanilla/test_predictions.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Input', 'Target', 'Predicted'])
        writer.writeheader()
        writer.writerows(all_predictions)



    # Print 3x3 sample grid....................
    sample_df = pd.DataFrame(all_predictions[:9])
    print("\nSample Prediction Grid:\n")
    for i, row in sample_df.iterrows():
        print(f"{i+1}. Input: {row['Input']} | Target: {row['Target']} | Predicted: {row['Predicted']}")



# Call the function
training_test(best_config)





