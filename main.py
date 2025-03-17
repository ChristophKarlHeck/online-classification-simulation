import argparse
import numpy as np
import pandas as pd
import torch
import importlib.util
import os
import inspect

def load_classifier(path_to_trained_model: str):
    """Loads a trained PyTorch model dynamically from a `.pt` file."""

    model_file = os.path.join(path_to_trained_model, "model.pt")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    try:
        # Try loading as a TorchScript model first
        model = torch.jit.load(model_file)
        print("Loaded TorchScript model.")
    except RuntimeError:
        # If that fails, try loading as a standard PyTorch model
        print("Failed to load as TorchScript.")

    model.eval()
    return model





#def adjusted_min_max(input: np.array) -> np.array:



def main():
    parser = argparse.ArgumentParser(description="Preprocess CSV files.")
    parser.add_argument("--data_dir", required=True, help="Directory with raw files.")
    parser.add_argument("--classifier_dir", required=True, help="Directory with trained CNN.")
    parser.add_argument("--normalization", required=True, help="Normalization method.")
    parser.add_argument("--prefix", required=True, help="C1, basically choose the plant.")
    args = parser.parse_args()

    data_dir = args.data_dir
    classifier_dir = args.classifier_dir
    normalization_method = args.normalization.lower()
    prefix = args.prefix.upper()

    classifier = load_classifier(classifier_dir)



if __name__ == "__main__":
    main()