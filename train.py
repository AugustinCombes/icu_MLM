import torch 
import torch.nn as nn
import torch.nn.functional as F

import json
import argparse

from os.path import join as pjoin
import numpy as np

from dataloader import get_dataloader
from model import BERT_MLM

import ast

#JSON_DIR = "data/datasets_full"

parser = argparse.ArgumentParser(description='Training python script.')
parser.add_argument('input_json_dir', type=str, help='Directory where the train/valid/test json are stored.')

parser.add_argument('--embedding_dim', type=int, default=128, help='The dimension in which the codes are embedded.')
parser.add_argument('--model_dim', type=int, default=256, help='The dimension of the BERT model.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of encoder layers in the BERT model.')
parser.add_argument('--dropout', type=float, default=0.2, help='The dropout of the BERT model.')
parser.add_argument('--heads', type=int, default=128, help='The dimension in which the codes are embedded.')

parser.add_argument('--batch_size', type=int, default=32, help='The size of the batches when training.')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='The learning rate when training.')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')

parser.add_argument('--debug', action='store_true', default=False, help='Whether the training script is launched in debug mode.')

args = parser.parse_args()


JSON_DIR = args.input_json_dir

d_embedding = args.embedding_dim
d_model = args.model_dim
n_layers = args.n_layers
dropout = args.dropout
nhead = args.heads

batchSize = args.batch_size
learningRate = args.learning_rate
epochs = args.epochs

debug = args.debug

device = 'cuda'

train_dl = get_dataloader(batch_size=batchSize, drop_last=True, mode="train", device=device, mlm_index="all", json_path="data/datasets_full", tokenizer_path="data/datasets_full/tokenizer.json")
valid_dl = get_dataloader(batch_size=batchSize, drop_last=True, mode="valid", device=device, mlm_index="all", json_path="data/datasets_full", tokenizer_path="data/datasets_full/tokenizer.json")

model = BERT_MLM(d_embedding, d_model, tokenizer_path="data/datasets_full/tokenizer.json", dropout=dropout, nhead=nhead, device=device)
#model.load_state_dict(torch.load("saved_models/2106_1/2106_1.pt"))

criterion = nn.NLLLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(1, epochs+1):
    print("Epoch", epoch)
    
    model.train(); torch.cuda.empty_cache()
    epoch_loss, epoch_acc = [], []

    for batch in train_dl:
        optimizer.zero_grad()
        
        tokens = batch["tokens"]
        targets = batch["target_tokens"]
        
        predicted_tokens = model(tokens)

        was_mlm_token = tokens == 2
        mlm_predicted_tokens = predicted_tokens[was_mlm_token]

        loss = criterion(mlm_predicted_tokens, targets)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.detach().item())
        epoch_acc.append((targets == mlm_predicted_tokens.argmax(dim=1)).float().mean().item())

    epoch_loss = np.array(epoch_loss).mean()
    epoch_acc = np.array(epoch_acc).mean()

    print(f"Epoch {epoch}:", "train: loss {:.2f}; accuracy {:.2f}".format(epoch_loss, epoch_acc))

    model.eval(); torch.cuda.empty_cache()
    epoch_loss, epoch_acc = [], []

    for batch in valid_dl:
        tokens = batch["tokens"]
        targets = batch["target_tokens"]
        
        predicted_tokens = model(tokens)

        was_mlm_token = tokens == 2
        mlm_predicted_tokens = predicted_tokens[was_mlm_token]

        loss = criterion(mlm_predicted_tokens, targets)
        
        epoch_loss.append(loss.detach().item())
        epoch_acc.append((targets == mlm_predicted_tokens.argmax(dim=1)).float().mean().item())

    epoch_loss = np.array(epoch_loss).mean()
    epoch_acc = np.array(epoch_acc).mean()

    print(f"Epoch {epoch}:", "valid: loss {:.2f}; accuracy {:.2f}".format(epoch_loss, epoch_acc))