from datasets import load_from_disk
ds = load_from_disk("data/goemotions_tok")
print(ds["train"][0])
