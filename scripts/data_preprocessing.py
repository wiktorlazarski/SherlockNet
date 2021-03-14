import collections
import json
import os
import re


class BytePairEncoding:
    def __init__(self, tokens, vocab):
        self.tokens = tokens
        self.vocab = vocab

    @classmethod
    def create(cls, corpus, num_merges):
        tokens, vocab_encoded = cls._create_tokens(corpus, num_merges)
        tokens = sorted(tokens.keys(), key=lambda k: len(k), reverse=True)

        return cls(tokens, vocab_encoded)

    @classmethod
    def load(cls, token_json, vocab_json):
        with open(token_json, 'r') as token_json:
            tokens = json.load(token_json)

        with open(vocab_json, 'r') as vocab_json:
            vocab = json.load(vocab_json)

        return cls(tokens, vocab)

    @classmethod
    def _create_tokens(self, corpus, num_merges):
        vocab = self._build_vocab(corpus)

        for i in range(num_merges):
            pairs = self._get_stats(vocab)

            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)

        tokens = collections.defaultdict(int)
        vocab_encoded = {}
        for word, freq in vocab.items():
            word_tokens = word.split()

            for token in word_tokens:
                tokens[token] += freq

            word = ''.join(word_tokens).replace('</w>', '')
            vocab_encoded[word] = word_tokens

        return tokens, vocab_encoded

    @classmethod
    def _build_vocab(self, corpus):
        tokens = [" ".join(word) + " </w>" for word in corpus.split()]

        vocab = collections.Counter(tokens)

        return vocab

    @classmethod
    def _get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, frequency in vocab.items():
            symbols = word.split()

            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += frequency

        return pairs

    @classmethod
    def _merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]

        return v_out

    def encoder(self):
        pass

    def decode(self):
        pass


def merge_corpora(data_dir):
    corpora = []
    for book in os.listdir(data_dir):
        book_path = os.path.join(data_dir, book)

        with open(book_path, 'r') as file:
            corpora.append(file.read())

    return '\n'.join(corpora)


if __name__ == '__main__':
    corpus = merge_corpora("./data/processed/sherlock")

    encoder = BytePairEncoding.create(corpus, 1000)

    with open('./assets/tokens_1k.json', 'w') as token_json:
        json.dump(encoder.tokens, token_json)

    with open('./assets/vocab_1k.json', 'w') as vocab_json:
        json.dump(encoder.vocab, vocab_json)
