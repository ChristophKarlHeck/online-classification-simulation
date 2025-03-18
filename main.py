import argparse
import ast 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import torch
import os

from online_min_max import OnlineMinMax

def plot_data(df_classified: pd.DataFrame, threshold: float) -> None:
    
    plant_id=999

    #------------------Prepare Data for Plot---------------------------------------#
    window_size = 100 # 100 = 10min
    df_classified["ch0_smoothed"] = df_classified["classification_ch0"].rolling(window=window_size, min_periods=1).mean()
    df_classified["ch1_smoothed"] = df_classified["classification_ch1"].rolling(window=window_size, min_periods=1).mean()
    df_classified['datetime'] = pd.to_datetime(df_classified['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df_classified['datetime'] = df_classified['datetime'] + pd.Timedelta(hours=1)
    df_classified['LastVoltageCh0'] = df_classified['input_normalized_ch0'].apply(lambda x: x[-1])
    df_classified['LastVoltageCh1'] = df_classified['input_normalized_ch1'].apply(lambda x: x[-1])
    df_classified["LastVoltageCh0"] = df_classified["LastVoltageCh0"].rolling(window=window_size, min_periods=1).mean()
    df_classified["LastVoltageCh1"] = df_classified["LastVoltageCh1"].rolling(window=window_size, min_periods=1).mean()

    fig_width = 5.90666  # Width in inches
    aspect_ratio = 0.618  # Example aspect ratio (height/width)
    fig_height = fig_width * aspect_ratio

    fig, axs = plt.subplots(2, 1, figsize=(fig_width, 8), sharex=True)

    time_fmt = mdates.DateFormatter('%H:%M')

    for ax in axs:
        ax.grid(True, linestyle='dashed', linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(time_fmt)  # Format x-axis as hours
        ax.tick_params(axis='x', labelsize=10)  # Set font size to 10
        plt.setp(ax.get_xticklabels(), fontsize=10, rotation=0, ha='center')

    # Scatter plot for classification
    axs[0].plot(df_classified['datetime'], df_classified["ch0_smoothed"], label="CH0", color="blue")
    axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed"], label="CH1", color="green")

    axs[0].axhline(y=threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold: {threshold}")

    axs[0].fill_between(df_classified['datetime'], 0, 1.0, 
                    where=(df_classified["ch0_smoothed"] > threshold) & (df_classified["ch1_smoothed"] > threshold), 
                    color='gray', alpha=0.3, label="Stimulus prediction")


    axs[0].fill_between(
        df_classified['datetime'], 0, 1.0, 
        where=(df_classified["ground_truth"] == 1), 
        color='limegreen', alpha=0.3, label="Stimulus application"
    )


    # Ensure y-axis limits and set explicit tick marks
    axs[0].set_ylim(0, 1.05)
    axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1])  # Explicitly set y-ticks
    axs[0].set_ylabel("Heat Phase Probability",fontsize=10)
    axs[0].tick_params(axis='y', labelsize=10) 

    axs[0].set_title(f"Online Heat Phase Classification Using Ivy Data (ID {plant_id})", fontsize=10, pad=40)
    axs[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, framealpha=0.7)


    # Line plot for interpolated electric potential
    axs[1].plot(df_classified['datetime'], df_classified['LastVoltageCh0'], label="CH0", color="blue")
    axs[1].plot(df_classified['datetime'], df_classified['LastVoltageCh1'], label="CH1", color="green", linestyle="dashed")

    # Labels and Titles
    axs[1].tick_params(axis='y', labelsize=10)
    axs[1].set_ylabel("EDP [scaled]",fontsize=10)
    axs[1].set_title("Normalized CNN Input via Adjusted Min-Max Scaling",fontsize=10)
    axs[1].legend(fontsize=8, loc="lower right")

    # Improve spacing to prevent label cutoff
    fig.tight_layout()

    # Save figure in PGF format with proper bounding box
    #plt.savefig(f"minMaxOnlineClassificationAdjusted{plant_id}Shifted.pgf", format="pgf", bbox_inches="tight", pad_inches=0.05)
    #plot_path = os.path.join(save_dir, f"{prefix}_classified_plot.png")
    #plt.savefig(plot_path, dpi=300)
    plt.show()


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

def min_max(arr: np.ndarray, min_val: float, max_val: float, factor: float) -> np.ndarray:
    """Adjusted Min-Max Normalization"""

    arr = np.array(arr, dtype=np.float32)
    return ((arr - min_val) / (max_val - min_val))*factor


online_min_max_ch0 = OnlineMinMaxCh0(600)
online_min_max_ch1 = OnlineMinMaxCh1(600)

def apply_normalization(arr: np.ndarray, normalization: str, channel: bool) -> np.ndarray:
    """Applies the selected normalization method."""
    if normalization == "adjusted_min_max":
        return adjusted_min_max(arr)
    elif normalization == "min_max":
        if not channel:
            online_min_max_ch0.update(arr)
            return min_max(arr, online_min_max_ch0.get_min_value(), online_min_max_ch0.get_max_value(), 1000)
        else:
            online_min_max_ch1.update(arr)
            return min_max(arr, online_min_max_ch1.get_min_value(), online_min_max_ch1.get_max_value(), 1000)
    else:
        raise ValueError(f"Unsupported normalization method: {normalization}")

def online_experiment(classifier, df_input_not_normalized: pd.DataFrame, normalization: str) -> pd.DataFrame:

    print("Running Online Experiment")

    df = df_input_not_normalized.copy()
    df["classification_ch0"] = None
    df["classification_ch1"] = None
    df["input_normalized_ch0"] = None
    df["input_normalized_ch1"] = None

    for index, row in df.iterrows():

        if isinstance(row["input_not_normalized_ch0"], (list, np.ndarray)):
            normalized_ch0 = apply_normalization(np.array(row["input_not_normalized_ch0"]), normalization, False)
            input_tensor_ch0 = torch.tensor(normalized_ch0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction_ch0 = classifier(input_tensor_ch0)

            # Extract the second value from the prediction list ([prob_class0, prob_class1])
            df.at[index, "classification_ch0"] = prediction_ch0.flatten().tolist()[1]
            # Use .at[] to store the list as a single object in the cell
            df.at[index, "input_normalized_ch0"] = normalized_ch0.tolist()

        if isinstance(row["input_not_normalized_ch1"], (list, np.ndarray)):
            normalized_ch1 = apply_normalization(np.array(row["input_not_normalized_ch1"]), normalization, True)
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
    df_result = online_experiment(classifier, df_input_not_normalized, normalization_method)
    plot_data(df_result, 0.6)



if __name__ == "__main__":
    main()