"""Stufe 3: Das gespeicherte Modell auf dem Testset bewerten.

Eingabe:  model/, params.yaml (train.max_length)
Ausgabe:  reports/eval_metrics.json (Accuracy, Macro-F1, schwächste Anliegen, Durchsatz, Größe)
Die Stufe lädt das Modell von der Festplatte: Bewertet wird genau das, was später ausgeliefert wird.
"""
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import pipeline_lib as lib
from common.data import load_banking77


def main() -> None:
    max_length = lib.load_params()["train"]["max_length"]
    _, test, intents = load_banking77()
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSequenceClassification.from_pretrained("model")
    pred, seconds = lib.predict_labels(model, tokenizer, test["text"].tolist(), max_length)
    per_intent = f1_score(test["label"], pred, average=None, labels=list(range(len(intents))))
    weakest = sorted(zip(intents, per_intent), key=lambda item: item[1])[:5]
    metrics = {
        "test_accuracy": float(accuracy_score(test["label"], pred)),
        "test_macro_f1": float(f1_score(test["label"], pred, average="macro")),
        "weakest_intents": {name: round(float(score), 3) for name, score in weakest},
        "texts_per_second": round(len(test) / seconds),
        "model_size_mb": round(lib.weights_mb(model), 1),
    }
    lib.write_json("reports/eval_metrics.json", metrics)
    print(f"evaluate: test_accuracy={metrics['test_accuracy']:.4f}, test_macro_f1={metrics['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
