# src/eval.py
"""
Evaluate the fine-tuned model on the test set.
Run:
    python src/eval.py --data_dir data/goemo_small --ckpt checkpoints/best
"""

import argparse, json, numpy as np, torch
from datasets import load_from_disk, DatasetDict
from transformers import AutoModelForSequenceClassification
from utils import compute_metrics

NUM_LABELS = 28


# ------------------------------------------------------------------ #
def ensure_multihot(ds_test) -> np.ndarray:
    """
    Return labels as float32 multi-hot ndarray (N, 28),
    converting from list<int> if necessary.
    """
    if "labels" in ds_test.column_names:
        return np.stack(ds_test["labels"])

    # else column is still called 'emotion' and stores list<int>
    def to_multihot(example):
        vec = np.zeros(NUM_LABELS, dtype=np.float32)
        for idx in example["emotion"]:
            vec[idx] = 1.0
        example["labels"] = vec
        return example

    ds_test = ds_test.map(to_multihot, batched=False)
    return np.stack(ds_test["labels"])


# ------------------------------------------------------------------ #
def main(data_dir: str, ckpt: str):
    ds_test = load_from_disk(data_dir)["test"]

    labels = ensure_multihot(ds_test)

    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(len(ds_test)):
            batch = {
                "input_ids":      torch.tensor(ds_test["input_ids"][i]).unsqueeze(0),
                "attention_mask": torch.tensor(ds_test["attention_mask"][i]).unsqueeze(0),
            }
            logits = model(**batch).logits
            preds.append((logits.sigmoid() > 0.5).cpu().numpy())

    preds = np.vstack(preds).astype(int)
    metrics = compute_metrics((preds, labels))
    print(json.dumps(metrics, indent=2))


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/goemo_small")
    ap.add_argument("--ckpt",     default="checkpoints/best")
    main(**vars(ap.parse_args()))
