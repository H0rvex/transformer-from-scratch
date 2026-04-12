import torch
import torch.nn as nn
import math
from config import config

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
       
        # linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, sql_len, d_model = x.shape

        # Step 1: Project
        Q = self.W_q(x) # (batch, sql_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Reshape into heads
        Q = Q.view(batch_size, sql_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, sql_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, sql_len, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Apply scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attention = weights @ V # (batch, num_heads, seq_len, d_k)
       
        # Step 4: Concat heads back → (batch, sql_len, d_model)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, sql_len, d_model)
        
        # Step 5: Final linear projection
        return self.W_o(attention)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        # layer normaliaztion
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # feed-forward layers
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        # dropouts
        self.drop1 = nn.Dropout(config["dropout"])
        self.drop2 = nn.Dropout(config["dropout"])
    
    def forward(self, x, mask=None):
        x = self.ln1(x + self.drop1(self.attention(x, mask)))
        x = self.ln2(x + self.drop2(self.ff2(torch.relu(self.ff1(x)))))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float() # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, 
                 num_heads, d_ff, num_layers, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=5000)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.clas = nn.Linear(d_model, num_classes)
        self.drop = nn.Dropout(config["dropout"])

    def forward(self, x):
        mask = (x != 0).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, sql_len)
        x_input = x
        x = self.drop(self.pos(self.emb(x)))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.drop(x)
        mask_for_avg = (x_input != 0).unsqueeze(-1).float() # (batch, sql_len, 1)
        x = (x * mask_for_avg).sum(dim=1) / mask_for_avg.sum(dim=1).clamp(min=1)
        return self.clas(x)