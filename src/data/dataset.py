import numpy as np

class TextDataset:
    def __init__(self, tokens, context_length):
        self.tokens = np.array(tokens)
        self.context_length = context_length

    def get_batch(self, batch_size):
        X = []
        Y = []

        max_idx = len(self.tokens) - self.context_length

        for _ in range(batch_size):
            i = np.random.randint(0, max_idx)

            x = self.tokens[i : i + self.context_length]
            y = self.tokens[i + 1 : i + self.context_length + 1]

            X.append(x)
            Y.append(y)

        return np.array(X), np.array(Y)

