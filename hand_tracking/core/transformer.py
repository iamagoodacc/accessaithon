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

from typing import List, TypedDict
import torch
from torch import Tensor
import torch.nn as nn
import math

from torch.utils.data import DataLoader

VOCABULARY = {
    0: "blank",
    1: "Hello",
    2: "I",
    3: "You",
    4: "Want",
    5: "Apple",
}

class FrameData(TypedDict):
    nframes: int
    lhand_features: Tensor  # shape: (nframes, lhand_features_count)
    rhand_features: Tensor  # shape: (nframes, rhand_features_count)
    pose_features: Tensor   # shape: (nframes, pose_features_count)
    labels: Tensor          # shape: (num_signs,) — sequence of word indices e.g. [1, 3, 2]
                            # indices must start from 1, since 0 is reserved for CTC blank

class SignLangDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths: list[str]):
        """
        @param file_paths: list of paths to .pt files
        """
        super().__init__()
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index) -> FrameData:
        # Load the dictionary from the disk
        # map_location='cpu' ensures we don't accidentally flood the GPU memory
        data: FrameData = torch.load(self.file_paths[index], map_location='cpu', weights_only=True)
        return data

def collate_fn(batch: List[FrameData]):
    lhand_list = [data["lhand_features"] for data in batch]
    rhand_list = [data["rhand_features"] for data in batch]
    pose_list = [data["pose_features"] for data in batch]
    
    # shape of (batch_size)
    lengths = torch.tensor([d['nframes'] for d in batch], dtype=torch.long)

    lhand_padded = torch.nn.utils.rnn.pad_sequence(lhand_list, batch_first=True)
    rhand_padded = torch.nn.utils.rnn.pad_sequence(rhand_list, batch_first=True)
    pose_padded  = torch.nn.utils.rnn.pad_sequence(pose_list, batch_first=True)

    max_len = lhand_padded.size(1)

    # first get (max_len), and then expand that into (batch_size, max_len)
    # then length.unsqueeze(1) to turn it into (batch_size, 1)
    # >= will use broadcasting to turn (batch_sizee, 1) to (batch_size, max_len) and return a bool tensor of (batch_size, max_len) where the bool is for the comparison between each element
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    labels = torch.cat([data["labels"] for data in batch], dim=0)
    target_lengths = torch.tensor([data["labels"].size(0) for data in batch], dtype=torch.long)
    
    # this is what we will get for each batch
    return {
        'lhand': lhand_padded,   # (batch, max_frames, lhand_features)
        'rhand': rhand_padded,   # (batch, max_frames, rhand_features)
        'pose': pose_padded,    # (batch, max_frames, pose_features)
        'mask': mask,           # (batch, max_frames) — True means ignore
        'lengths': lengths,        # (batch,) — real frame count per sample
        'labels': labels,  # (total_labels,) — all labels concatenated
        'target_lengths': target_lengths  # (batch,) — how many labels each sample has
    }

def get_dataloader(dataset: SignLangDataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    return loader

class PositionalEncoder(nn.Module):
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
    """This is used to train on entire sentences"""
    def __init__(self, nhead: int, lhand_size: int, rhand_size: int, pose_size: int, num_words: int, num_layers: int, d_model: int):
        """
        nhead needs to be a factor of 128 
        """

        super().__init__()

        self.d_model = d_model
        
        # this makes sure that the features are separately mapped so they don't get mixed up
        self.lhand_extractor = nn.Linear(in_features=lhand_size, out_features=d_model // 4)
        self.rhand_extractor = nn.Linear(in_features=rhand_size, out_features=d_model // 4)
        self.pose_extractor = nn.Linear(in_features=pose_size, out_features=d_model // 2)

        self.positional_encoder = PositionalEncoder(d_model=self.d_model)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.ctc_mapper = nn.Linear(in_features=self.d_model, out_features=num_words + 1)

    def forward(self, lhand_x, rhand_x, pose_x, padding_mask=None):
        """Padding mask has shape [batch_size, frame_count], basically if this frame should be considered, mainly for training videos of different lengths"""

        lhand = self.lhand_extractor(lhand_x)
        rhand = self.rhand_extractor(rhand_x)
        pose = self.pose_extractor(pose_x)

        # eventually the goal is to have a 512 sized matrix
        # we will have tensors with shape (batch, frames, features), so if we set dimension to -1, we stack on features
        x = torch.cat([lhand, rhand, pose], dim=-1)
        x = self.positional_encoder(x)

        w_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return self.ctc_mapper(w_output)

def train_ctc(model: CtcRecognitionModel, lr: float, num_epochs: int, dataset: SignLangDataset):
    # 0 as CTC blank and make infinities 0
    criterion = nn.CTCLoss(zero_infinity=True, blank=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = get_dataloader(dataset)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            logits = model(batch['lhand'], batch['rhand'], batch['pose'], batch['mask'])
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            log_probs = log_probs.permute(1, 0, 2)

            optimizer.zero_grad()

            loss = criterion(log_probs, batch['labels'], batch['lengths'], batch['target_lengths'])
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

class BertRecognitionModel(nn.Module):
    def __init__(self, nhead: int, lhand_size: int, rhand_size: int, pose_size: int, num_words: int, num_layers: int):
        super().__init__()

        # This one just adds a frame as a weighted data
        # nn.Parameter() means that loss.backward() will affect it
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))

        self.d_model = 512
        
        # this makes sure that the features are separately mapped so they don't get mixed up
        # If we just straight up converts them into a 512 tensor that might be consufing
        self.lhand_extractor = nn.Linear(in_features=lhand_size, out_features=128)
        self.rhand_extractor = nn.Linear(in_features=rhand_size, out_features=128)
        self.pose_extractor = nn.Linear(in_features=pose_size, out_features=256)

        self.positional_encoder = PositionalEncoder(d_model=self.d_model)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.mapper = nn.Linear(in_features=self.d_model, out_features=num_words + 1)


    def forward(self, lhand_x, rhand_x, pose_x, padding_mask=None):
        lhand = self.lhand_extractor(lhand_x)
        rhand = self.rhand_extractor(rhand_x)
        pose = self.pose_extractor(pose_x)
        
        x = torch.cat([lhand, rhand, pose], dim=-1)
 
        # this gets the batch size because the structure of a tensor here is (batch, frame_count, len)
        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # adds the cls tokens to the frame_count one
        x = torch.cat([cls_tokens, x], dim=1)

        # the cls tokens need a pose encoder asw
        x = self.positional_encoder(x)

        padding_mask_with_cls = None
        if padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            padding_mask_with_cls = torch.cat([cls_mask, padding_mask], dim=1)

        w_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask_with_cls)

        return self.mapper(w_output[:, 0, :])
