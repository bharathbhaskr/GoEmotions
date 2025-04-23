# src/hpo.py
"""
Hyper-parameter optimisation with Optuna.
Run: python src/hpo.py --data_dir data/goemotions_tok
"""

import argparse
from datasets import load_from_disk
from transformers import (AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import optuna
from utils import seed_everything, compute_metrics

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=28,
        problem_type="multi_label_classification")

def objective(trial, ds):
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    epochs = trial.suggest_int("epochs", 2, 6)
    bs = trial.suggest_categorical("bs", [16, 32])

    args = TrainingArguments(
        output_dir="optuna_logs",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="no",
        disable_tqdm=True,
        logging_steps=9999
    )
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics)
    res = trainer.train()
    return trainer.evaluate()["eval_macro_f1"]

def main(data_dir):
    seed_everything(42)
    ds = load_from_disk(data_dir)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, ds), n_trials=20)
    print("Best params:", study.best_params)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/goemotions_tok")
    main(**vars(ap.parse_args()))
