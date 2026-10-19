import pandas as pd
from sentence_transformers import SentenceTransformer

train_df = pd.read_parquet("data/banking77/train.parquet")
test_df = pd.read_parquet("data/banking77/test.parquet")
model = SentenceTransformer("models/minilm-l6", device="cpu")

corpus = train_df.iloc[:3000]  # Teilmenge, damit es schnell geht
queries = test_df.sample(200)
corpus_emb = model.encode(corpus["text"].tolist(), normalize_embeddings=True)
query_emb = model.encode(queries["text"].tolist(), normalize_embeddings=True)

best = (query_emb @ corpus_emb.T).argmax(axis=1)
hit_rate = (corpus["label"].to_numpy()[best] == queries["label"].to_numpy()).mean()
print(f"Trefferquote Top-1: {hit_rate:.2f}")




index_emb, seconds = timed_encode(minilm, train["text"].tolist(), BATCH_SIZE)
rate = len(index_emb)/seconds
hours_10m = 10_000_000 / rate / 3600 
print(f"Durchsatz: {rate}, 10 Mio. docs: {hours_10m}")
