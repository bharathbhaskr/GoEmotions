# src/data_prep.py
"""
Download, subsample, tokenize and save GoEmotions.
Example:
    # create 2 000 train / 250 val / 250 test rows
    python src/data_prep.py \
           --out_dir data/goemo_small \
           --train_rows 2000 --val_rows 250 --test_rows 250
"""

import os, random, argparse
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

SEED = 42
random.seed(SEED)


# --------------------------------------------------------------------------- #
def main(out_dir: str = "data/goemotions_tok",
         max_len: int = 128,
         train_rows: int | None = None,
         val_rows: int | None = None,
         test_rows: int | None = None):
    # 1. download (cached after first use)
    ds = load_dataset("go_emotions")

    # 2. drop blank rows
    ds = ds.filter(lambda ex: ex["text"].strip() != "")

    # 3. optional subsampling ------------------------------------------------ #
    def slice_split(split: str, n_rows: int | None):
        if n_rows is None or n_rows >= len(ds[split]):
            return ds[split]
        return ds[split].shuffle(seed=SEED).select(range(n_rows))

    ds = DatasetDict({
        "train":      slice_split("train",      train_rows),
        "validation": slice_split("validation", val_rows),
        "test":       slice_split("test",       test_rows),
    })
    # ----------------------------------------------------------------------- #

    # 4. tokenise
    tok = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tok(batch["text"],
                   truncation=True,
                   padding="max_length",
                   max_length=max_len)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds_tok = ds_tok.rename_column("labels", "emotion")

    # 5. save to disk
    os.makedirs(out_dir, exist_ok=True)
    ds_tok.save_to_disk(out_dir)

    print(f"✅ Saved {len(ds_tok['train'])}/{len(ds_tok['validation'])}/"
          f"{len(ds_tok['test'])} rows → {out_dir}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/goemotions_tok")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--train_rows", type=int)
    p.add_argument("--val_rows",   type=int)
    p.add_argument("--test_rows",  type=int)
    main(**vars(p.parse_args()))
