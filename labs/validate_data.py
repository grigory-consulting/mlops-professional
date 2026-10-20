import sys

import pipeline_lib as lib
from common.data import dataset_hash, load_banking77


def main() -> None:
    params = lib.load_params()["data"]
    train, test, intents = load_banking77()
    counts = train["label"].value_counts()
    data_hash = dataset_hash(train, test)[:16]
    checks = {
        "hash_matches": data_hash == params["expected_hash"],
        "n_intents": len(intents) == params["n_intents"] and train["label"].nunique() == params["n_intents"],
        "labels_in_range": bool(train["label"].between(0, len(intents) - 1).all() and test["label"].between(0, len(intents) - 1).all()),
        "no_empty_texts": bool((train["text"].str.strip().str.len() > 0).all() and (test["text"].str.strip().str.len() > 0).all()),
        "min_examples_per_intent": int(counts.min()) >= params["min_examples_per_intent"],
    }
    lib.write_json("reports/data_report.json", {
        "data_hash": data_hash,
        "n_train": len(train),
        "n_test": len(test),
        "n_intents": len(intents),
        "examples_per_intent_min": int(counts.min()),
        "examples_per_intent_max": int(counts.max()),
        "checks": checks,
        "passed": all(checks.values()),
    })
    print(f"validate_data: hash={data_hash}, checks={checks}")
    if not all(checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()