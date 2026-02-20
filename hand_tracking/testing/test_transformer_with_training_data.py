"""
Tests the trained model against the training data to verify the pipeline is correct.
If the model can't overfit its own training data, something is wrong with the code.
"""
import os
import glob
import torch
from core.transformer import CtcRecognitionModel, SignLangDataset, collate_fn, VOCABULARY
from torch.utils.data import DataLoader

MODEL_PATH = "model.pt"
DATA_DIR   = "training/data"

def decode(log_probs: torch.Tensor) -> list[str]:
    """
    Greedy CTC decode — collapse repeats and remove blanks.
    log_probs: (frames, num_words+1)
    """
    indices = log_probs.argmax(dim=-1).tolist()
    decoded = []
    prev = None
    for idx in indices:
        if idx != prev:
            if idx != 0:  # 0 is blank
                decoded.append(VOCABULARY.get(idx, f"?{idx}"))
        prev = idx
    return decoded


def test():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"No model found at {MODEL_PATH} — run train.py first")

    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    config     = checkpoint['config']

    model = CtcRecognitionModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    file_paths = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    if not file_paths:
        raise RuntimeError(f"No .pt files found in {DATA_DIR}/")

    dataset    = SignLangDataset(file_paths)
    # batch_size=1 so we can inspect each sample individually
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    correct = 0
    total   = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logits    = model(batch['lhand'], batch['rhand'], batch['pose'], batch['mask'])
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # decode — squeeze batch dimension since batch_size=1
            predicted = decode(log_probs.squeeze(0))

            # decode ground truth from concatenated labels
            target_len = batch['target_lengths'][0].item()
            ground_truth = [VOCABULARY.get(idx.item(), f"?{idx}") for idx in batch['labels'][:target_len]]

            match = predicted == ground_truth
            if match:
                correct += 1
            total += 1

            status = "✓" if match else "✗"
            print(f"[{status}] Sample {i+1:>3} | GT: {ground_truth} | Pred: {predicted}")

    print(f"\nAccuracy on training data: {correct}/{total} ({100*correct/total:.1f}%)")
    print("Note: if accuracy is very low the model may need more epochs or more data.")


if __name__ == "__main__":
    test()
