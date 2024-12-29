
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_query = nn.Linear(d_model, d_model)
        self.w_keys = nn.Linear(d_model, d_model)
        self.w_values = nn.Linear(d_model, d_model)
        self.w_output = nn.Linear(d_model, d_model)

        def scaled_dot_product(self, q, k, v, mask=None):
            attention_scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(d_model))
            if mask != None:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            attention_probabilities = torch.softmax(attention_scores, dim=-1)
            output = torch.matmul(attention_probabilities, v)
            return output
        
        def split_heads(self, x):
            batch_size, sequence_length, d_k = x.size()
            return x.view(batch_size, sequence_length, self.num_heads, self.d_k).self.transpose(1, 2)

        def combine_heads(self, x):
            batch_size, sequence_length, d_k = x.size()
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

        def forward(self, q, k, v, mask=None):
            q = self.split_heads(self.w_query(q))
            k = self.split_heads(self.w_keys(k))
            v = self.split_heads(self.w_values(v))

            attention_output = self.scaled_dot_product_attention(q, k, v, mask)
            output = self.w_output(self.combine_heads(attention_output))
            return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))



