import torch
from torch.utils.data import IterableDataset
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import json
import os
import ast
import random

max_len = 150

class JsonEHRDataset(IterableDataset):
    def __init__(self, json_path, tokenizer_path, mode="train", device="cuda", mlm_index=None):

        assert mode in ["train", "valid", "test"]
        assert mlm_index in ["all", "labels"]

        self.json_path = json_path
        self.device = device
        self.mlm_index = mlm_index
        self.tokenizer = json.load(open(tokenizer_path))
        self.mode = mode

    def prob_mask_like(self, inputs, mask_ratio=0.15):
        return torch.zeros_like(inputs).float().uniform_(0, 1) < mask_ratio

    def __iter__(self):
        with open(os.path.join(self.json_path, f"{self.mode}.json")) as f:
            for sample_line in f:
                line = ast.literal_eval(sample_line)

                tokens = line[1:] #would be 2: with original data
                tokens = torch.tensor(list(map(lambda x: self.tokenizer.get(x, 1), tokens)), device=self.device)[:max_len]

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

                tokens_with_mask = torch.where(bert_mask==1, self.tokenizer['[MSK]'] * torch.ones_like(tokens), tokens)
                tokens_with_mask = torch.where(bert_mask==2, torch.randint_like(tokens_with_mask, low=4, high=len(self.tokenizer)), tokens_with_mask)
                
                targets = tokens[bert_mask==1]

                sample = {
                    "tokens": tokens_with_mask,
                    "target_tokens": targets
                }
                yield sample

class ShuffleDataset(IterableDataset):
  def __init__(self, dataset, buffer_size):
    super().__init__()
    self.dataset = dataset
    self.buffer_size = buffer_size

  def __iter__(self):
    shufbuf = []
    try:
        dataset_iter = iter(self.dataset)
        for i in range(self.buffer_size):
            shufbuf.append(next(dataset_iter))
    except:
        self.buffer_size = len(shufbuf)

    try:
        while True:
            try:
                item = next(dataset_iter)
                evict_idx = random.randint(0, self.buffer_size - 1)
                yield shufbuf[evict_idx]
                shufbuf[evict_idx] = item
            except StopIteration:
                break
        while len(shufbuf) > 0:
            yield shufbuf.pop()
    except GeneratorExit:
        pass

def get_dataloader(batch_size, drop_last=True, buffer_size=42, **kwargs):
    ds = JsonEHRDataset(**kwargs)
    ds = ShuffleDataset(ds, buffer_size)
    return DataLoader(dataset=ds, batch_size=batch_size, drop_last=drop_last, collate_fn=collate_fn)

def collate_fn(batch):
    tokens = [item['tokens'] for item in batch]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)

    target_tokens = torch.cat([item['target_tokens'] for item in batch])
    
    return {
        "tokens": tokens, 
        "target_tokens": target_tokens
        }