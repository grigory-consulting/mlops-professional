"""DVC Stage 2: stratified Train/Test-Split."""
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = yaml.safe_load((ROOT / "params.yaml").read_text())
    raw = np.load(ROOT / "data" / "raw.npz")
    X, y = raw["X"], raw["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=p["data"]["test_size"],
        stratify=y,
        random_state=p["seed"],
    )
    np.save(ROOT / "data" / "X_train.npy", X_train)
    np.save(ROOT / "data" / "X_test.npy", X_test)
    np.save(ROOT / "data" / "y_train.npy", y_train)
    np.save(ROOT / "data" / "y_test.npy", y_test)
    print(f"Split: train={X_train.shape}, test={X_test.shape}")


if __name__ == "__main__":
    main()
