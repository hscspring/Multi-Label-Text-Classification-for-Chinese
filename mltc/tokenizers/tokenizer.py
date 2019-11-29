import os
import collections

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


class Tokenizer:

    def __init__(self, vocab_path: str):
        if not os.path.isfile(vocab_path):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_path))
        self.vocab = load_vocab(vocab_path)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def tokenize(self, text: str):
        return text.split(" ")


    def convert_tokens_to_ids(self, token_list: list):
        ids = []
        for token in token_list:
            ids.append(self._convert_token_to_id(token))
        return ids


