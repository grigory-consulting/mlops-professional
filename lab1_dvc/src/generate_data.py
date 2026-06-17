"""DVC Stage 1: synthetischen Datensatz erzeugen."""
from pathlib import Path

import numpy as np
import yaml
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = yaml.safe_load((ROOT / "params.yaml").read_text())
    X, y = make_classification(
        n_samples=p["data"]["n_samples"],
        n_features=p["data"]["n_features"],
        n_informative=p["data"]["n_informative"],
        n_redundant=p["data"]["n_redundant"],
        weights=p["data"]["weights"],
        random_state=p["seed"],
    )
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / "data" / "raw.npz", X=X, y=y)
    print(f"raw.npz geschrieben: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
