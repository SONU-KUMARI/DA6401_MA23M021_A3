import argparse
import torch
import wandb
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from ma23m021_a3_attention import build_vocab_and_prepare_batch, TransliterationModelattention, run_training,read_pairs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transliteration Model")
    parser.add_argument('--embedding_size', type=int, default=128, help="Size of word embeddings")
    parser.add_argument('--hidden_size', type=int, default=256, help="Size of hidden layers")
    parser.add_argument('--enc_layers', type=int, default=2, help="Number of layers in the encoder")
    parser.add_argument('--dec_layers', type=int, default=2, help="Number of layers in the decoder")
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help="Type of RNN")
    parser.add_argument('--dropout_rate', type=float, default=0.2, help="Dropout rate")
    parser.add_argument('--is_bidirectional', type=bool, default=False, help="Whether the encoder RNN is bidirectional")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument('--train_path', type=str, required=True, help="Path to the training dataset")
    parser.add_argument('--dev_path', type=str, required=True, help="Path to the validation dataset")
    parser.add_argument('--test_path', type=str, required=True, help="Path to the test dataset")
    parser.add_argument('--teacher_forcing_prob', type=float, default=0.5, help="Teacher forcing probability")
    return parser.parse_args()

# Load datasets and vocabularies
args = parse_args()
train_data = read_pairs(args.train_path)
dev_data = read_pairs(args.dev_path)
test_data = read_pairs(args.test_path)


src_vocab, idx2src, tgt_vocab, idx2tgt, create_batch, unique_chars_latin, unique_chars_dev = build_vocab_and_prepare_batch(train_data, device)

# Initialize the model
model = TransliterationModelattention(
    input_vocab_size=len(src_vocab),
    output_vocab_size=len(tgt_vocab),
    embedding_size=args.embedding_size,
    hidden_size=args.hidden_size,
    enc_layers=args.enc_layers,
    dec_layers=args.dec_layers,
    rnn_type=args.rnn_type,
    dropout_rate=args.dropout_rate,
    is_bidirectional=args.is_bidirectional
).to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=src_vocab['<pad>'])

# Run training
run_training(args, model, optimizer, criterion, train_data, dev_data, src_vocab, tgt_vocab, create_batch, device)
