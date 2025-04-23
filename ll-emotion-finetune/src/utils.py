# src/utils.py
import random, numpy as np, torch
from sklearn.metrics import f1_score, accuracy_score

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0).astype(int)   # multi-label threshold @ 0
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    exact = (preds == labels).all(axis=1).mean()
    return {"macro_f1": macro_f1, "exact_match": exact}
