# 🎭 Mini-GoEmotions Fine-Tune

Small-scale multi-label emotion classifier built by fine-tuning **roberta-base** on a 2 000-example subset of the GoEmotions dataset.

| Split | Rows |
|-------|------|
| Train | 2 000 |
| Val   | 250  |
| Test  | 250  |

## 📊 Final Test Metrics

| Metric            | Baseline (zero-shot) | Fine-tuned |
|-------------------|----------------------|------------|
| Macro-F1          | 0.000                | **0.056**  |
| Exact-Match       | 0.000                | **0.208**  |

## 🚀 Quick Start

```bash
conda create -n goemo python=3.10 -y
conda activate goemo
pip install -r requirements.txt

# Prepare data
python src/data_prep.py --out_dir data/goemo_small --train_rows 2000 --val_rows 250 --test_rows 250

# Fine-tune
python src/train.py --data_dir data/goemo_small

# Evaluate
python src/eval.py --data_dir data/goemo_small --ckpt checkpoints/best
