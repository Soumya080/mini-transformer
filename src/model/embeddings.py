import numpy as np

class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = np.random.randn(vocab_size, d_model)
    
    def __call__(self, token_ids):
        return self.weight[token_ids]


def sinusoidal_position_encoding(T, d_model):
    PE = np.zeros((T, d_model))
    for p in range(T):
        for d in range(d_model):
            if d % 2 == 0:
                PE[p, d] = np.sin(p / 10000 ** (d / d_model))
            else:
                PE[p, d] = np.cos(p / 10000 ** (d / d_model))
    return PE


## Sanity Check 
if __name__ == "__main__":
    T = 32
    vocab_size = 100
    d_model = 128
    tokens = np.random.randint(0, vocab_size, size=(T,))
    embed = TokenEmbedding(vocab_size, d_model)

    X = embed(tokens)
    print(X.shape)  # MUST be (32, 128)

    PE = sinusoidal_position_encoding(T=32, d_model=128)
    print(PE.shape)  # MUST be (32, 128)
                   
    ## FINAL 
    X = embed(tokens)
    PE = sinusoidal_position_encoding(T=32, d_model=128)
    X = X + PE

    print(X.shape)

    print("mean:", X.mean())
    print("std:", X.std())
    print("min:", X.min())
    print("max:", X.max())

