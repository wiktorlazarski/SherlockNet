import os
import re
import collections

class BytePairEncoding:
    def __init__(self, corpus, num_merges):
        self.tokens, self.vocab_encoded = self._create_tokens(corpus, num_merges)

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

    def encoder(self):
        pass

    def decode(self):
        pass

    def _build_vocab(self, corpus):
        tokens = [" ".join(word) + " </w>" for word in corpus.split()]

        vocab = collections.Counter(tokens)

        return vocab

    def _get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, frequency in vocab.items():
            symbols = word.split()

            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += frequency

        return pairs

    def _merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]

        return v_out


def merge_corpora(data_dir):
    corpora = []
    for book in os.listdir(data_dir):
        book_path = os.path.join(data_dir, book)

        with open(book_path, 'r') as file:
            corpora.append(file.read())

    return '\n'.join(corpora)


if __name__ == '__main__':
    corpus = merge_corpora("./data/processed/sherlock")

    encoder = BytePairEncoding(corpus, 500)
    print(encoder.tokens)
    print(encoder.vocab_encoded)
    # print(len(encoder.tokens))