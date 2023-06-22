import torch
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import ast
import os

def collate_fn(batch):
    tokens = [item['tokens'] for item in batch]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)

    target_tokens = torch.cat([item['target_tokens'] for item in batch])
    
    return {
        "tokens": tokens, 
        "target_tokens": target_tokens
        }

max_len = 150

class EHRDataset(Dataset):
    def __init__(self, mode="train", device="cuda", tokenizer=None, mlm_index=None, json_path=None):
        
        assert mode in ["train", "valid", "test"]
        assert json_path is not None
        assert tokenizer is not None
        assert mlm_index in ["all", "labels"]

        with open(os.path.join("data/datasets_full", f"{mode}.json"), "r") as f:
            data = f.readlines()
        data = list(map(ast.literal_eval, data))
        dic = {row[1]: row[2:] for row in data}

        self.adm_index = list(dic.keys())
        self.tokens = [torch.tensor(list(map(tokenizer.encode, record)), device=device)[:max_len] for record in dic.values()]
        
        self.device = device
        self.mlm_index = mlm_index
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.adm_index)
    
    def prob_mask_like(self, inputs, mask_ratio=0.15):
        return torch.zeros_like(inputs).float().uniform_(0, 1) < mask_ratio

    def __getitem__(self, index):
        tokens = self.tokens[index]

        bert_mask = torch.zeros_like(tokens)
        #zero if we let the token unchanged
        #one if we mask it
        #two if we replace it with a random token

        if self.mlm_index == "labels":
            bert_mask[:3] = 1

        elif self.mlm_index == "all":
            total_mask = self.prob_mask_like(bert_mask).to(self.device)        
            if total_mask[total_mask].numel() > 0:
                sample = Categorical(torch.tensor([0.1, 0.8, 0.1], device=self.device)).sample(total_mask[total_mask].shape)
                bert_mask[total_mask] = sample

        tokens_with_mask = torch.where(bert_mask==1, self.tokenizer.encoder['[MSK]'] * torch.ones_like(tokens), tokens)
        tokens_with_mask = torch.where(bert_mask==2, torch.randint_like(tokens_with_mask, low=4, high=len(self.tokenizer.encoder)), tokens_with_mask)
        
        targets = tokens[bert_mask==1]

        sample = {
            "tokens": tokens_with_mask,
            "target_tokens": targets
        }
        return sample

def get_dataloader(batch_size, shuffle=True, drop_last=True, **kwargs):
    ds = EHRDataset(**kwargs)
    return DataLoader(dataset=ds, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, collate_fn=collate_fn)