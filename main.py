import argparse
import ast 
import numpy as np
import pandas as pd
import torch
import os

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


def load_data(data_dir: str, prefix: str) -> pd.DataFrame:
    """Loads raw data"""
    file_name = f"input_not_normalized_{prefix}.csv"
    file_path = os.path.join(data_dir, file_name)

    # Check if the file exists before loading
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Convert stringified lists back into NumPy arrays
    df["input_not_normalized_ch0_arr"] = df["input_not_normalized_ch0"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
    df["input_not_normalized_ch1_arr"] = df["input_not_normalized_ch1"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))

    df.drop(columns=["input_not_normalized_ch0", "input_not_normalized_ch1"], inplace=True)

    df.rename(columns={
        "input_not_normalized_ch0_arr": "input_not_normalized_ch0",
        "input_not_normalized_ch1_arr": "input_not_normalized_ch1"
    }, inplace=True)

    #print(df.head())
    return df


def adjusted_min_max(arr: np.ndarray) -> np.ndarray:
    """Adjusted Min-Max Normalization"""
    min_val = -0.2
    max_val = 0.2
    arr = np.array(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def apply_normalization(arr: np.ndarray, normalization: str) -> np.ndarray:
    """Applies the selected normalization method."""
    if normalization == "adjusted_min_max":
        return adjusted_min_max(arr)
    else:
        raise ValueError(f"Unsupported normalization method: {normalization}")

def online_experiment(classifier, df_input_not_normalized: pd.DataFrame, normalization: str) -> pd.DataFrame:

    printf("Running Online Experiment")

    df = df_input_not_normalized.copy()
    df["classification_ch0"] = None
    df["classification_ch1"] = None
    df["input_normalized_ch0"] = None
    df["input_normalized_ch1"] = None

    for index, row in df.iterrows():

        if isinstance(row["input_not_normalized_ch0"], (list, np.ndarray)):
            normalized_ch0 = apply_normalization(np.array(row["input_not_normalized_ch0"]), normalization)
            input_tensor_ch0 = torch.tensor(normalized_ch0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction_ch0 = classifier(input_tensor_ch0)

            # Extract the second value from the prediction list ([prob_class0, prob_class1])
            df.at[index, "classification_ch0"] = prediction_ch0.flatten().tolist()[1]
            # Use .at[] to store the list as a single object in the cell
            df.at[index, "input_normalized_ch0"] = normalized_ch0.tolist()

        if isinstance(row["input_not_normalized_ch1"], (list, np.ndarray)):
            normalized_ch1 = apply_normalization(np.array(row["input_not_normalized_ch1"]), normalization)
            input_tensor_ch1 = torch.tensor(normalized_ch1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction_ch1 = classifier(input_tensor_ch1)

            df.at[index, "classification_ch1"] = prediction_ch1.flatten().tolist()[1]
            df.at[index, "input_normalized_ch1"] = normalized_ch1.tolist()

    return df




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
    df_input_not_normalized = load_data(data_dir, prefix)
    result = online_experiment(classifier, df_input_not_normalized, normalization_method)
    print(result.columns)
    



if __name__ == "__main__":
    main()