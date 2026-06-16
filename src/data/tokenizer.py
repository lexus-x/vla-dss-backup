"""
Simple word-level tokenizer for LIBERO task instructions.
Maps task names like "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
to integer token sequences.
"""

import re
from collections import defaultdict


class SimpleTokenizer:
    """Word-level tokenizer with fixed vocabulary built from LIBERO task names."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, max_seq_len: int = 64):
        self.max_seq_len = max_seq_len
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self._next_idx = 2

    def _tokenize_text(self, text: str) -> list[str]:
        """Split text into words."""
        # Handle both underscore-separated task names and natural language
        text = text.replace("_", " ").lower().strip()
        return re.findall(r'[a-z]+', text)

    def fit(self, texts: list[str]):
        """Build vocabulary from list of task descriptions."""
        for text in texts:
            for word in self._tokenize_text(text):
                if word not in self.word2idx:
                    self.word2idx[word] = self._next_idx
                    self.idx2word[self._next_idx] = word
                    self._next_idx += 1

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        """
        Encode text to token IDs with attention mask.

        Returns:
            token_ids: list of ints, padded to max_seq_len
            attention_mask: list of 1s (real tokens) and 0s (padding)
        """
        words = self._tokenize_text(text)
        ids = []
        for w in words[:self.max_seq_len]:
            ids.append(self.word2idx.get(w, self.word2idx[self.UNK_TOKEN]))

        # Pad
        attn_mask = [1] * len(ids) + [0] * (self.max_seq_len - len(ids))
        ids = ids + [0] * (self.max_seq_len - len(ids))

        return ids[:self.max_seq_len], attn_mask[:self.max_seq_len]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)


class HFTokenizerAdapter:
    """Wraps a HuggingFace tokenizer behind SimpleTokenizer's .encode interface,
    so it drops into the dataset unchanged. Used with PretrainedLanguageEncoder.
    """

    def __init__(self, model_name: str, max_seq_len: int = 32):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_len = max_seq_len
        self.model_name = model_name

    def encode(self, text: str):
        enc = self.tok(
            text.replace("_", " "), padding="max_length", truncation=True,
            max_length=self.max_seq_len, return_tensors=None,
        )
        return enc["input_ids"], enc["attention_mask"]

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size
