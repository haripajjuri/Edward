class modelConfig:
    def __init__(
        self, 
        vocab_size, 
        emb_dim = 256, #number of features
        n_layers = 8, #number of layers in model
        n_heads = 8, #number of heads for attention model ( emb_dim // 4 )
        max_token_len = 512 #moximum token length that model can accept
    ):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_token_len = max_token_len