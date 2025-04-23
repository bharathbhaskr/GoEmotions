# src/train.py
"""
Fine-tune roberta-base on GoEmotions (multi-label).
Run:
    python src/train.py --data_dir data/goemotions_tok
"""

import argparse, os
import numpy as np
from datasets import load_from_disk, DatasetDict, Sequence, Value, Features
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    IntervalStrategy,
)
from utils import seed_everything, compute_metrics

NUM_LABELS = 28


# ---------- helpers --------------------------------------------------------- #
def ensure_multihot_float(ds: DatasetDict) -> DatasetDict:
    """
    1. Rename `emotion` → `labels` if needed.
    2. Convert list<int> → 28-d float32 multi-hot vector.
    """
    if "emotion" in ds["train"].column_names and "labels" not in ds["train"].column_names:
        ds = ds.rename_column("emotion", "labels")

    # Only convert once.
    sample = ds["train"][0]["labels"]
    needs_multihot = isinstance(sample, list) and isinstance(sample[0], int)

    def to_multihot(example):
        vec = np.zeros(NUM_LABELS, dtype=np.float32)
        for idx in example["labels"]:
            vec[idx] = 1.0
        example["labels"] = vec
        return example

    if needs_multihot:
        ds = ds.map(to_multihot, batched=False)

    # Explicitly declare column type so set_format -> torch.float32 tensor
    new_features = ds["train"].features.copy()
    new_features["labels"] = Sequence(feature=Value("float32"), length=NUM_LABELS)
    ds = ds.cast(new_features)

    return ds


# ---------- main ------------------------------------------------------------ #
def main(data_dir: str, ckpt_dir: str = "checkpoints"):
    seed_everything(42)

    # 1. Load dataset and fix labels dtype
    ds = load_from_disk(data_dir)
    ds = ensure_multihot_float(ds)

    # 2. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    # 3. TrainingArguments (Transformers ≥ 4.51)
    args = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        eval_delay=0,
    )
    args.set_evaluate(strategy=IntervalStrategy.EPOCH)
    args.set_save(strategy=IntervalStrategy.EPOCH)
    args.set_logging(strategy=IntervalStrategy.STEPS, steps=50)
    args.evaluation_delay = 0  
    args.eval_delay = 0  

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    # 5. Train & save
    trainer.train()
    save_path = os.path.join(ckpt_dir, "best")
    trainer.save_model(save_path)

    # NEW: save tokenizer too
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    tokenizer.save_pretrained(save_path)
    print("✅  Training complete; best model saved to", os.path.join(ckpt_dir, "best"))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/goemotions_tok")
    p.add_argument("--ckpt_dir", default="checkpoints")
    main(**vars(p.parse_args()))
