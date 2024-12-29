
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
            