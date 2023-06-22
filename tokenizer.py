import numpy as np
import hashlib
import os

class Tokenizer():
    def __init__(self,
                 data_dir: str,
                 special_tokens: dict,
                 min_freq: int=5):
        """ Initialize word-level tokenizer.
            
        Parameters
        ----------
            special_tokens (list of str)
                tokens used in the model for unknown words, padding, masking,
                starting/closing sentences, etc.
            min_freq (int)
                minimum frequency at which a word should occur in the corpus to have
                its own token_id (else: token_id of '[UNK]')
                
        """
        self.data_dir = data_dir
        self.encoder = dict(special_tokens)
        self.special_tokens = dict(special_tokens)
        self.min_freq = min_freq
        self.unique_id = self.create_unique_id()
        self.path = os.path.join(self.data_dir, 'tokenizer', self.unique_id)
        
    def create_unique_id(self):
        unique_str = str(vars(self))
        return hashlib.sha256(unique_str.encode()).hexdigest()

    def fit(self, words: list):        
        # Compute and sort vocabulary
        word_vocab, word_counts = np.unique(words, return_counts=True)
        if self.min_freq > 0:  # remove rare words
            word_vocab = word_vocab[word_counts >= self.min_freq]
            word_counts = word_counts[word_counts >= self.min_freq]
        inds = word_counts.argsort()[::-1]
        word_vocab = word_vocab[inds]
        
        # Generate word level encoder
        self.encoder.update({i: (idx + len(self.special_tokens))
                             for idx, i in enumerate(word_vocab)})
        
        # Store word count for every word (useful for skipgram dataset)
        self.word_counts = {self.encoder[word]: count for word, count in \
                            zip(word_vocab, sorted(word_counts)[::-1])}
        self.word_counts.update({k: 1 for k in self.special_tokens.values()})
        
        # Build decoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # Store and print tokenizer vocabulary information
        self.vocab_sizes = {'total': len(self.encoder),
                            'special': len(self.special_tokens),
                            'word': len(word_vocab)}
        
    def encode(self, word: str):
        try:
            return self.encoder[word]
        except:
            return self.encoder['[UNK]']
    
    def decode(self, token_id: int):
        return self.decoder[token_id]
    
    def get_vocab(self):
        return self.encoder.keys()