"""
* @ means matrix multiplication
Transformers
d_model, W_Q, W_K, and W_V
heads

The input can be equally split into several heads based on features
There will also be a neural network that maps the input x to the size of d_model

From now on the x refers to the input in each head

Say the input x is an NxM matrix, that is, N frames of M features for example
W_Q, W_K, and W_V will be MxM matrices trained with a neural network

Q = x @ w_Q # with size (N, M)
K = x @ w_K # with size (N, M)
V = x @ w_V # with size (N, M)

# K.T inverts the matrix
scores = Q @ K.T / sqrt(M) # with size (N, N), M is here to scale down the dot products
scores = softmax(scores)
output = scores @ V

Now we get (N, M) output, we can either take the first one and see which sign it is and use cross entropy loss or:

CTC

First of all we want to define a "dictionary", which is like
0 - CTC blank
1 - Hello
2 - Bye
for example

The dictionary will have length V (including the CTC blank)

We then want to have another matrix W_V with size (M, V)
we do output @ W_V, which will produce a matrix of (N, V)

now this (N, V) matrix has the probability of each word in each frame, we will take the highest one
e.g.:
            Highest probability
frame 0:    Hello
frame 1:    Hello
frame 2:    Hello
...
frame 9:    Hello
frame 10:   CTC blank
...
frame 15:   CTC blank
frame 16:   Bye 
...
frame 20:   Bye 
frame 20:   CTC blank 

we get to use pytorch's CTC decoder to decode this
Then we send it to CTC loss and pytorch will do the magic during training
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    This is not an NN model it adds timestamp to each frame
    """

    def __init__(self, d_model, max_len=5000, dropout:float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # initialize a giant array of 0's of (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
    
        # initialize it with [0..max_len] and unsqueeze into [[0], [1], [2], ...]
        # the reason we unsqueeze here is that it can then be calculated the dot product
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # this was in a google paper I believe
        # this turns the array into a decreasing array like [1,00, ... 0.0001]
        div_term = torch.exp(
            # initializes [0, 2, 4, 6 .. d_model - 2]
            torch.arange(0, d_model, 2)
                .float()
                # this is a magic number that is used
                * (-math.log(10000.0) / d_model))
        
        # a::b syntax is starting from a and step b
        pe[:, 0::2] = torch.sin(position * div_term) # dot product here
        pe[:, 1::2] = torch.cos(position * div_term)

        # unsqueeze(0) makes it [1, max_len, d_model]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Sequence, d_model]
        # pytorch automatically copies it
        x = x + self.pe[:, :x.size(1), :] # Add the timestamps
        return self.dropout(x)

class CtcRecognitionModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, input_size: int, num_words: int) -> None:
        super().__init__()

        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.input_projection = nn.Linear(in_features=input_size, out_features=d_model)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.ctc_mapper = nn.Linear(in_features=d_model, out_features=num_words + 1)

    def forward(self, x):
        projected = self.input_projection(x)
        w_output = self.transformer_encoder(projected)
        return self.ctc_mapper(w_output)
