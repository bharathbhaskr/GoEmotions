# src/inference.py
"""
Start an API that returns emotions for any sentence.
Run:  uvicorn src.inference:app --reload
Then test: curl -X POST "http://127.0.0.1:8000/emotion" -d '{"text": "I love this!"}'
"""

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CKPT = "checkpoints/best"      # path from train.py

tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(CKPT)
model.eval()

id2label = model.config.id2label
app = FastAPI()

class InputText(BaseModel):
    text: str

@torch.no_grad()
@app.post("/emotion")
def emotion(inp: InputText):
    tok = tokenizer(inp.text, return_tensors="pt", truncation=True,
                    padding=True, max_length=128)
    scores = torch.sigmoid(model(**tok).logits)[0].cpu().numpy()
    labels = [id2label[i] for i, s in enumerate(scores) if s > 0.5]
    return {"labels": labels, "scores": scores.tolist()}
