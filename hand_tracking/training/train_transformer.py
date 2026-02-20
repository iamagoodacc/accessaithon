"""
Trains the CTC model on recorded .pt files and saves the trained model.
"""
import os
import glob
import torch
import torch.nn as nn
from core.transformer import CtcRecognitionModel, SignLangDataset, get_dataloader

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "training/data"
MODEL_PATH = "model.pt"
LR         = 3e-4
NUM_EPOCHS = 300
NHEAD      = 4 
NUM_LAYERS = 1
D_MODEL    = 128
NUM_WORDS  = 5   # Hello, I, You, Want, Apple (blank is added automatically at 0)

# must match collect_handle feature sizes
LHAND_SIZE = 63
RHAND_SIZE = 63
POSE_SIZE  = 51


def train():
    file_paths = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    if not file_paths:
        raise RuntimeError(f"No .pt files found in {DATA_DIR}/")
    print(f"Found {len(file_paths)} samples.")

    dataset    = SignLangDataset(file_paths)
    dataloader = get_dataloader(dataset)

    model = CtcRecognitionModel(
        nhead=NHEAD,
        lhand_size=LHAND_SIZE,
        rhand_size=RHAND_SIZE,
        pose_size=POSE_SIZE,
        num_words=NUM_WORDS,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL
    )

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            logits    = model(batch['lhand'], batch['rhand'], batch['pose'], batch['mask'])
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.permute(1, 0, 2)   # (frames, batch, num_words+1)

            loss = criterion(
                log_probs,
                batch['labels'],
                batch['lengths'],
                batch['target_lengths']
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")

    # save model weights + config so we can reload it later
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'nhead':      NHEAD,
            'lhand_size': LHAND_SIZE,
            'rhand_size': RHAND_SIZE,
            'pose_size':  POSE_SIZE,
            'num_words':  NUM_WORDS,
            'num_layers': NUM_LAYERS,
            "d_model": D_MODEL
        }
    }, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
