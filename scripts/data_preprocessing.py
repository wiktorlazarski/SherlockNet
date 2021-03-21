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

        def token_len(token):
            return len(token[:-4]) + 1 if token.endswith("</w>") else len(token)

        tokens = sorted(tokens.keys(), key=token_len, reverse=True)

        return cls(tokens, vocab_encoded)

    @classmethod
    def load(cls, token_json, vocab_json):
        with open(token_json, "r") as token_json:
            tokens = json.load(token_json)

        with open(vocab_json, "r") as vocab_json:
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

            word = "".join(word_tokens).replace("</w>", "")
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
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        for word in v_in:
            w_out = p.sub("".join(pair), word)
            v_out[w_out] = v_in[word]

        return v_out

    def encode(self, word, unknown_token="</u>"):
        if word in self.vocab.keys():
            return self.vocab[word]

        word += "</w>"
        return self._encode(word, self.tokens, unknown_token)

    def _encode(self, word, sorted_tokens, unknown_token):
        if word == "":
            return []

        word_tokens = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_reg = re.escape(token.replace(".", "[.]"))

            matched_positions = [
                (m.start(0), m.end(0)) for m in re.finditer(token_reg, word)
            ]
            if len(matched_positions) == 0:
                continue

            substring_end_positions = [
                matched_position[0] for matched_position in matched_positions
            ]

            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = word[
                    substring_start_position:substring_end_position
                ]
                word_tokens += self._encode(
                    substring, sorted_tokens[i + 1 :], unknown_token
                )
                word_tokens += [token]
                substring_start_position = substring_end_position + len(token)

            remaining_substring = word[substring_start_position:]
            word_tokens += self._encode(
                remaining_substring, sorted_tokens[i + 1 :], unknown_token
            )
            break

        return word_tokens if sorted_tokens[i + 1 :] != [] else [unknown_token]

    def decode(self, encoded_text):
        text = "".join(encoded_text)
        text = text.replace("</w>", "")

        return text


def merge_corpora(data_dir):
    corpora = []
    for book in os.listdir(data_dir):
        book_path = os.path.join(data_dir, book)

        with open(book_path, "r") as file:
            corpora.append(file.read())

    return "\n".join(corpora)


def create_jsons(corpus, num_iters, token_json, vocab_json):
    encoder = BytePairEncoding.create(corpus, num_iters)

    with open(token_json, "w") as json_file:
        json.dump(encoder.tokens, json_file)

    with open(vocab_json, "w") as json_file:
        json.dump(encoder.vocab, json_file)


def create_encoded_corpus_file(corpus, out_path, token_json, vocab_json):
    corpus = corpus.split()

    encoder = BytePairEncoding.load(token_json, vocab_json)

    corpus = [encoder.encode(word) for word in corpus]
    corpus = [" ".join(encoded_word) for encoded_word in corpus]
    corpus = " ".join(corpus)

    with open(out_path, "w") as out_file:
        out_file.write(corpus)


if __name__ == "__main__":
    corpus = merge_corpora("./data/processed/sherlock")

    create_encoded_corpus_file(
        corpus,
        "./data/encoded_corpus.txt",
        "./assets/tokens_1k.json",
        "./assets/vocab_1k.json",
    )
