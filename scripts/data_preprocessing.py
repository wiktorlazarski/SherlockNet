import re
from collections import Counter, defaultdict

class BytePairEncoding:
    def __init__(self, corpus, num_merges):
        self.tokens = create_tokens()

    def _create_tokens(corpus, num_merges):
        vocab = self._build_vocab(corpus)

        for i in range(num_merges):
            pairs = self._get_stats(vocab)

            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)

    def encoder(self):
        pass

    def decode(self):
        pass

    def _build_vocab(self, corpus):
        tokens = [" ".join(word) + " </w>" for word in corpus.split()]

        vocab = Counter(tokens)

        return vocab

    def _get_stats(self, vocab):
        pairs = defaultdict(int)
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
