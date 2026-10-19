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



DATA_HASH = dataset_hash(train, test)[:16]


evalset = build_evalset(train,test, n_queries=1000, seed=SEED)
evalset.fingerprint

assert DATA_HASH == "3176e3ac9aef7185" and evalset.fingerprint == "e21d87d6477c7f22"

print(f"Daten-Hash {DATA_HASH}, Eval-Set {evalset.fingerprint} (Seed {evalset.seed}): "
      f"{len(evalset.queries)} Queries gegen {len(evalset.corpus)} Dokumente")
query_emb, query_seconds = timed_encode(minilm, evalset.queries, BATCH_SIZE)
metrics = retrieval_metrics(query_emb, evalset.query_labels, index_emb, evalset.corpus_labels, k=10)
metrics["texts_per_second"] = len(evalset.queries) / query_seconds
metrics["size_mb"] = size_mb(minilm)
pd.Series(metrics).round(3)
