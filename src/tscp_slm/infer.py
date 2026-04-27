from __future__ import annotations

import argparse
import json

from .model import TSCPMiniModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the TSCP mini model.")
    parser.add_argument("--model", required=True, help="Path to model artifact JSON.")
    parser.add_argument("--text", required=True, help="Text to classify.")
    args = parser.parse_args()

    model = TSCPMiniModel.load(args.model)
    prediction = model.predict(args.text)
    print(json.dumps(prediction.to_dict(), indent=2))


if __name__ == "__main__":
    main()
