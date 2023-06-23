import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import optuna
from optuna.trial import TrialState

import numpy as np

from dataloader import get_dataloader
from model import BERT_MLM

device = 'cuda'
batch_size = 32
epochs = 100

def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    nhead = trial.suggest_int("n_layers", 4, 16, 4)
    d_embedding = trial.suggest_int("d_embedding", 128, 1024, 128)
    d_model = trial.suggest_int("d_model", 128, 1024, 128)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)

    return BERT_MLM(d_embedding, d_model, tokenizer_path="data/datasets_full/tokenizer.json", n_layers=n_layers, dropout=dropout, nhead=nhead, device=device)


def objective(trial):
    model = define_model(trial)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    criterion = nn.NLLLoss().to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    train_dl = get_dataloader(batch_size=batch_size, drop_last=True, mode="train", device=device, mlm_index="all", json_path="data/datasets_full", tokenizer_path="data/datasets_full/tokenizer.json")
    valid_dl = get_dataloader(batch_size=batch_size, drop_last=True, mode="valid", device=device, mlm_index="all", json_path="data/datasets_full", tokenizer_path="data/datasets_full/tokenizer.json")

    for epoch in range(1, epochs+1):
        print("Epoch", epoch)

        model.train(); torch.cuda.empty_cache()
        epoch_loss, epoch_acc = [], []

        for idx, batch in enumerate(train_dl):
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

            # if idx>10:
            #     break

        epoch_loss = np.array(epoch_loss).mean()
        epoch_acc = np.array(epoch_acc).mean()

        print(f"Epoch {epoch}:", "train: loss {:.2f}; accuracy {:.2f}".format(epoch_loss, epoch_acc))

        model.eval(); torch.cuda.empty_cache()
        epoch_loss, epoch_acc = [], []

        with torch.no_grad():
            for idx, batch in enumerate(valid_dl):
                tokens = batch["tokens"]
                targets = batch["target_tokens"]
                
                predicted_tokens = model(tokens)

                was_mlm_token = tokens == 2
                mlm_predicted_tokens = predicted_tokens[was_mlm_token]

                loss = criterion(mlm_predicted_tokens, targets)
                
                epoch_loss.append(loss.detach().item())
                epoch_acc.append((targets == mlm_predicted_tokens.argmax(dim=1)).float().mean().item())

                # if idx>10:
                #     break

            epoch_loss = np.array(epoch_loss).mean()
            epoch_acc = np.array(epoch_acc).mean()

            print(f"Epoch {epoch}:", "valid: loss {:.2f}; accuracy {:.2f}".format(epoch_loss, epoch_acc))

        trial.report(epoch_acc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return epoch_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))