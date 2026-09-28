from pathlib import Path

import numpy
import pandas
import scipy
import sklearn


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "paper": ROOT / "paper" / "Feature-based_transfer_learning_for_network_security.pdf",
    "train": ROOT / "data" / "raw" / "KDDTrain+.txt",
    "test": ROOT / "data" / "raw" / "KDDTest+.txt",
}


def main() -> None:
    print("Python dependencies:")
    print(f"  numpy: {numpy.__version__}")
    print(f"  pandas: {pandas.__version__}")
    print(f"  scipy: {scipy.__version__}")
    print(f"  scikit-learn: {sklearn.__version__}")

    print("\nRequired files:")
    missing = []
    for name, path in EXPECTED.items():
        status = "OK" if path.is_file() else "MISSING"
        print(f"  {name}: {status} - {path}")
        if status == "MISSING":
            missing.append(name)

    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    print("\nEnvironment and project inputs are ready.")


if __name__ == "__main__":
    main()
