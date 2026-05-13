"""BPE Tokenizer — Built from scratch."""

class BPETokenizer:
    def __init__(self, text, num_merges=200):
        self.num_merges = num_merges
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}
        self._build_vocab(text)

    def _build_vocab(self, text):
        chars = sorted(list(set(text)))
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        self.inverse_vocab = {ch: i for i, ch in enumerate(chars)}
        tokens = [self.inverse_vocab[ch] for ch in text]
        next_id = len(chars)

        for _ in range(self.num_merges):
            pair_counts = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break
            new_str = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[next_id] = new_str
            self.inverse_vocab[new_str] = next_id
            self.merges[best_pair] = next_id
            tokens = self._merge(tokens, best_pair, next_id)
            next_id += 1

        self.vocab_size = len(self.vocab)

    def _merge(self, tokens, pair, new_id):
        merged = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and
                tokens[i] == pair[0] and tokens[i+1] == pair[1]):
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def encode(self, text):
        tokens = [self.inverse_vocab.get(ch, 0) for ch in text]
        for pair, new_id in self.merges.items():
            tokens = self._merge(tokens, pair, new_id)
        return tokens

    def decode(self, ids):
        return ''.join([self.vocab.get(i, '?') for i in ids])

    def get_stats(self, text):
        encoded = self.encode(text)
        return {
            "vocab_size": self.vocab_size,
            "num_merges": self.num_merges,
            "original_length": len(text),
            "encoded_length": len(encoded),
            "compression_ratio": round(len(text) / len(encoded), 2),
        }
