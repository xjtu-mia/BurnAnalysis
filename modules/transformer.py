import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, input_size, embed_dim = 128, num_heads = 8, hidden_size = 64, dropout_prob=0.1):
        super(TransformerBlock, self).__init__()
        self.embedding = nn.Linear(input_size, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_prob)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.layer_norm2 = nn.LayerNorm(input_size)
        self.dropout2 = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        B = x.shape[0]
        num_once = 40
        out = []
        for i in range(0, B, num_once):
            out.append(self._forward_layers(x[i:i+num_once if i+num_once < B else B]))
        return torch.cat(out, dim=0)

    def _forward_layers(self, x):
        B, C, H, W = x.shape
        x_ = x.view(B, C, -1)
        x = self.embedding(x_)
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        # Residual Connection and Layer Normalization
        x = self.layer_norm1(x + attn_output)
        
        # Feed-Forward Neural Network
        ff_output = self.fc(x)
        ff_output = self.dropout2(ff_output)
        # Residual Connection and Layer Normalization
        x = self.layer_norm2(x_ + ff_output)

        x = x.view(B, C, H, W)
        return x
