import numpy as np
import torch

from core.model import RecognitionModel, train

from main import SIGNS

x_data = []
y_data = []

for label_idx, sign in enumerate(SIGNS):
    data = np.load(f"data/{sign}.npy")   # shape: (20, 30, num_features)
    x_data.append(data)
    y_data.extend([label_idx] * len(data))   # 0 for hello, 1 for next sign, etc.

x_data = np.concatenate(x_data, axis=0)     # shape: (total_samples, 30, num_features)
y_data = np.array(y_data)                    # shape: (total_samples,)

x_train = torch.tensor(x_data, dtype=torch.float32)
y_train = torch.tensor(y_data, dtype=torch.long)    # long for classification

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

input_size  = x_train.shape[2]   # num_features per frame
output_size = len(SIGNS)         # one output per sign

model = RecognitionModel(
    input_size=input_size,
    hidden_size=128,
    num_layers=2,
    output_size=output_size,
    dropout=0.2
)

train(
    model=model,
    num_epochs=100,
    learning_rate=0.001,
    x_train=x_train,
    y_train=y_train
)

torch.save(model.state_dict(), "recognition_model.pth")
print("Model saved")
