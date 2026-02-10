import torch
import torch.nn as nn

"""
A long-short term memory model implementation with pytorch
A LSTM model has cell state and hidden state
Cell state is the consistent "memory" of the model
Hidden state is the memory about the current one

Each pass c0 is updated with 3 gates:
forget - forget this data (0)
add - add this data (1)
learn - learn based on the hidden state
"""

class RecognitionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float):
        """
        :param input_size:
        :param hidden_size: how many neurons in each layer
        :param num_layers:
        :param output_size:
        :param dropout: what percentage of neurons to drop out each layer
        """

        super(RecognitionModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        # batch_first means that the output structure will not be [sequence, batch, features]
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # out is a 3d tensor has structure of [batch, sequence, features]
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # : means selecting all
        out = self.fc(out[:, -1, :])

        return out

def train(model: RecognitionModel, num_epochs: int, learning_rate: float, x_train: torch.FloatTensor, y_train: torch.FloatTensor):
    """
    :param model:
    :param num_epochs:
    :param learning_rate:
    :param x_train: should have the structure [batch, seq_len, input_size]
    :param y_train: should have the structure [batch, output_size]
    """

    # mean squared error
    criterion = nn.MSELoss()
    # updates the weights every time
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # enter training mode
        model.train()

        # passes this into the model
        # outputs contains the predicted result + computational graph, which is hidden
        outputs = model(x_train)
        # calculate the error
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()

        # here, the hidden computational graph is used to see how to update the weights
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")