import torch
import torch.nn as nn
import torch.nn.functional as F
from config import modelConfig

class Transformer(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(normalized_shape=emb_dim)
        self.layernorm2 = nn.LayerNorm(normalized_shape=emb_dim)
        self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
        self.ffd = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim*4, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_norm1 = self.layernorm1(x)
        mha_out,_ = self.mha( query=x_norm1, key=x_norm1, value=x_norm1, key_padding_mask = key_padding_mask, attn_mask = attn_mask)
        x = x + mha_out

        x_norm2 = self.layernorm2(x)
        ffd_out = self.ffd(x_norm2)
        x = x + ffd_out
        return x


class Edward(nn.Module):
    def __init__(self, config: modelConfig):
        super().__init__()
        self.config = config
        self.embedding_layer = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.emb_dim)
        self.positioning_layer = nn.Embedding(num_embeddings=config.max_token_len, embedding_dim=config.emb_dim) 
        #max_token_len is max num of tokens model accepts. we should manage the len of tokens beforehand to manage Index out ot range error.

        self.blocks = nn.ModuleList([Transformer(config.emb_dim, config.n_heads) for  _ in range(config.n_layers)])
        
        self.output_layer = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, inputs, attention_mask=None, lables = None ):
        if inputs.shape[1] > self.config.max_token_len:
            raise ValueError(f"Input token length({inputs.shape[1]}) exceeded the maximum model token length({self.config.max_token_len})")
        
        positions = torch.arange(0, inputs.shape[1], device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
        x = self.embedding_layer(inputs) + self.positioning_layer(positions)

        seq_len = inputs.shape[1]

        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=inputs.device), diagonal=1).bool()
        
        for transformer in self.blocks:
            x = transformer(x, attn_mask = causal_mask, key_padding_mask = (attention_mask==0) if attention_mask is not None else None ) 

        logits = self.output_layer(x)

        if lables is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), lables.view(-1))
            return logits, loss
        
        return logits