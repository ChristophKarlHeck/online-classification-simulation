import argparse
import ast 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import torch
import os

from online_window import OnlineWindow
from typing import Optional

online_window_ch0 = OnlineWindow(600) #600
online_window_ch1 = OnlineWindow(600) #600
factor = 1000

def plot_data(df_classified: pd.DataFrame, threshold: float, normalization: str, num_classes: int, objective: str) -> None:
    
    plant_id=999

    #------------------Prepare Data for Plot---------------------------------------#
    window_size = 100 # 100 = 10min
    df_classified['datetime'] = pd.to_datetime(df_classified['datetime'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
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
    if num_classes == 2:
        axs[0].plot(df_classified['datetime'], df_classified["ch0_smoothed"], label="CH0", color="blue")
        axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed"], label="CH1", color="orange")

        axs[0].axhline(y=threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold: {threshold}")

        axs[0].fill_between(df_classified['datetime'], 0, 1.0, 
                        where=(df_classified["ch0_smoothed"] > threshold) & (df_classified["ch1_smoothed"] > threshold), 
                        color='gray', alpha=0.3, label="Stimulus prediction")


        if objective == "temp":
            axs[0].fill_between(
                df_classified['datetime'], 0, 1.0, 
                where=(df_classified["heat_ground_truth"] == 1), 
                color='limegreen', alpha=0.3, label="Stimulus application"
            )
        if objective == "ozone":
            axs[0].fill_between(
                df_classified['datetime'], 0, 1.0, 
                where=(df_classified["ozone_ground_truth"] == 1), 
                color='limegreen', alpha=0.3, label="Stimulus application"
            )

    if num_classes == 3:
        df_classified["smoothed_heat_mean"] = (df_classified["ch0_smoothed_heat"] + df_classified["ch1_smoothed_heat"])/2
        df_classified["smoothed_ozone_mean"] = (df_classified["ch0_smoothed_ozone"] + df_classified["ch1_smoothed_ozone"])/2
        df_classified["smoothed_idle_mean"] = (df_classified["ch0_smoothed_idle"] + df_classified["ch1_smoothed_idle"])/2

        # CH0: blues
        #axs[0].plot(df_classified['datetime'], df_classified["ch0_smoothed_idle"], label="Idle CH0", color="#add8e6")   # lightblue
        axs[0].plot(df_classified['datetime'], df_classified["smoothed_heat_mean"], label="Heat CH0", color="#FF0000")  # matplotlib default blue
        axs[0].plot(df_classified['datetime'], df_classified["smoothed_ozone_mean"], label="Ozone CH0", color="#007BFF") # darkblue
        axs[0].plot(df_classified['datetime'], df_classified["smoothed_idle_mean"], label="Ozone CH0", color="#012169")

        # CH1: oranges
        #axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed_idle"], label="Idle CH1", color="#ffdab9")   # peachpuff (light orange)
        # axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed_heat_mean"], label="Heat CH1", color="#8B0000")   # matplotlib default orange
        # axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed_ozone"], label="Ozone CH1", color="#00008B") # dark orange/brown


        axs[0].axhline(y=threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold: {threshold}")

        axs[0].fill_between(df_classified['datetime'], 0, 1.0, 
                        where=(df_classified["smoothed_heat_mean"] > threshold),# & (df_classified["ch1_smoothed_heat"] > threshold), 
                        color='#722F37', alpha=0.3, label="Stimulus prediction")

        if objective == "temp":
            axs[0].fill_between(
                df_classified['datetime'], 0, 1.0, 
                where=(df_classified["heat_ground_truth"] == 1), 
                color='#DC143C', alpha=0.3, label="Stimulus application"
            )
        if objective == "ozone":
            axs[0].fill_between(
                df_classified['datetime'], 0, 1.0, 
                where=(df_classified["ozone_ground_truth"] == 1), 
                color='#DC143C', alpha=0.3, label="Stimulus application"
            )


    # Ensure y-axis limits and set explicit tick marks
    axs[0].set_ylim(0, 1.05)
    axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1])  # Explicitly set y-ticks
    axs[0].set_ylabel("Heat Phase Probability",fontsize=10)
    axs[0].tick_params(axis='y', labelsize=10) 

    axs[0].set_title(f"Online Heat Phase Classification Using Ivy Data (ID {plant_id})", fontsize=10, pad=40)
    axs[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, framealpha=0.7)


    # Line plot for interpolated electric potential
    axs[1].plot(df_classified['datetime'], df_classified['LastVoltageCh0'], label="CH0", color="#90EE90")
    axs[1].plot(df_classified['datetime'], df_classified['LastVoltageCh1'], label="CH1", color="#013220")

    # Labels and Titles
    axs[1].tick_params(axis='y', labelsize=10)
    axs[1].set_ylabel("EDP [scaled]",fontsize=10)
    axs[1].set_title(f"Normalized CNN Input via {normalization}",fontsize=10)
    axs[1].legend(fontsize=8, loc="lower right")

    # Improve spacing to prevent label cutoff
    fig.tight_layout()

    # Save figure in PGF format with proper bounding box
    #plt.savefig(f"minMaxOnlineClassificationAdjusted{plant_id}Shifted.pgf", format="pgf", bbox_inches="tight", pad_inches=0.05)
    #plot_path = os.path.join(save_dir, f"{prefix}_classified_plot.png")
    #plt.savefig(plot_path, dpi=300)
    plt.show()

def smooth_classification(df_classified: pd.DataFrame, window_size: int, num_classes: int) -> pd.DataFrame:

    if num_classes == 2:
        df_classified["ch0_smoothed"] = df_classified["classification_ch0"].rolling(window=window_size, min_periods=1).mean()
        df_classified["ch1_smoothed"] = df_classified["classification_ch1"].rolling(window=window_size, min_periods=1).mean()
    if num_classes == 3:
        df_classified["ch0_smoothed_idle"] = df_classified["classification_ch0_idle"].rolling(window=window_size, min_periods=1).mean()
        df_classified["ch0_smoothed_heat"] = df_classified["classification_ch0_heat"].rolling(window=window_size, min_periods=1).mean()
        df_classified["ch0_smoothed_ozone"] = df_classified["classification_ch0_ozone"].rolling(window=window_size, min_periods=1).mean()
        df_classified["ch1_smoothed_idle"] = df_classified["classification_ch1_idle"].rolling(window=window_size, min_periods=1).mean()
        df_classified["ch1_smoothed_heat"] = df_classified["classification_ch1_heat"].rolling(window=window_size, min_periods=1).mean()
        df_classified["ch1_smoothed_ozone"] = df_classified["classification_ch1_ozone"].rolling(window=window_size, min_periods=1).mean()


    return df_classified

def metrics(df_classified: pd.DataFrame, threshold: float, num_classes: int, objective: str):

    true_positive_cases = 0
    false_positive_cases = 0
    true_negative_cases = 0
    false_negative_cases = 0

    if num_classes == 2:
        if objective == "temp":
            true_positive_cases =  (
                ((df_classified["heat_ground_truth"] == 1) & 
                (df_classified["ch0_smoothed"] > threshold) & 
                (df_classified["ch1_smoothed"] > threshold))
            )
        if objective == "ozone":
            true_positive_cases =  (
                ((df_classified["ozone_ground_truth"] == 1) & 
                (df_classified["ch0_smoothed"] > threshold) & 
                (df_classified["ch1_smoothed"] > threshold))
            )
        if objective == "temp":
            false_positive_cases =  (
                ((df_classified["heat_ground_truth"] == 0) & 
                (df_classified["ch0_smoothed"] > threshold) & 
                (df_classified["ch1_smoothed"] > threshold))
            )
        if objective == "ozone":
            false_positive_cases =  (
                ((df_classified["ozone_ground_truth"] == 0) & 
                (df_classified["ch0_smoothed"] > threshold) & 
                (df_classified["ch1_smoothed"] > threshold))
            )

        if objective == "temp":
            true_negative_cases =  (
                ((df_classified["heat_ground_truth"] == 0) & 
                ((df_classified["ch0_smoothed"] <= threshold) |
                (df_classified["ch1_smoothed"] <= threshold)))
            )
        if objective == "ozone":
            true_negative_cases =  (
                ((df_classified["ozone_ground_truth"] == 0) & 
                ((df_classified["ch0_smoothed"] <= threshold) |
                (df_classified["ch1_smoothed"] <= threshold)))
            )
        if objective == "temp":
            false_negative_cases =  (
                ((df_classified["heat_ground_truth"] == 1) & 
                ((df_classified["ch0_smoothed"] <= threshold) | 
                (df_classified["ch1_smoothed"] <= threshold)))
            )
        if objective == "ozone":
            false_negative_cases =  (
                ((df_classified["ozone_ground_truth"] == 1) & 
                ((df_classified["ch0_smoothed"] <= threshold) | 
                (df_classified["ch1_smoothed"] <= threshold)))
            )

    if num_classes == 3:
        if objective == "temp":
            true_positive_cases =  (
                ((df_classified["heat_ground_truth"] == 1) & 
                (df_classified["ch0_smoothed_heat"] > threshold) & 
                (df_classified["ch1_smoothed_heat"] > threshold))
            )
        if objective == "ozone":
            true_positive_cases =  (
                ((df_classified["ozone_ground_truth"] == 1) & 
                (df_classified["ch0_smoothed_heat"] > threshold) & 
                (df_classified["ch1_smoothed_heat"] > threshold))
            )

        if objective == "temp":
            false_positive_cases =  (
                ((df_classified["heat_ground_truth"] == 0) & 
                (df_classified["ch0_smoothed_heat"] > threshold) & 
                (df_classified["ch1_smoothed_heat"] > threshold))
            )
        if objective == "ozone":
            false_positive_cases =  (
                ((df_classified["ozone_ground_truth"] == 0) & 
                (df_classified["ch0_smoothed_heat"] > threshold) & 
                (df_classified["ch1_smoothed_heat"] > threshold))
            )

        if objective == "temp":
            true_negative_cases =  (
                ((df_classified["heat_ground_truth"] == 0) & 
                ((df_classified["ch0_smoothed_heat"] <= threshold) |
                (df_classified["ch1_smoothed_heat"] <= threshold)))
            )

        if objective == "ozone":
            true_negative_cases =  (
                ((df_classified["ozone_ground_truth"] == 0) & 
                ((df_classified["ch0_smoothed_heat"] <= threshold) |
                (df_classified["ch1_smoothed_heat"] <= threshold)))
            )

        if objective == "temp":
            false_negative_cases =  (
                ((df_classified["heat_ground_truth"] == 1) & 
                ((df_classified["ch0_smoothed_heat"] <= threshold) | 
                (df_classified["ch1_smoothed_heat"] <= threshold)))
            )
        if objective == "ozone":
            false_negative_cases =  (
                ((df_classified["ozone_ground_truth"] == 1) & 
                ((df_classified["ch0_smoothed_heat"] <= threshold) | 
                (df_classified["ch1_smoothed_heat"] <= threshold)))
            )
    
    true_positive = true_positive_cases.sum()
    false_positive = false_positive_cases.sum()
    true_negative = true_negative_cases.sum()
    false_negative = false_negative_cases.sum()

    return true_positive, false_positive, true_negative, false_negative

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


def load_temp_data(data_dir: str, prefix: str) -> pd.DataFrame:
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
        "input_not_normalized_ch1_arr": "input_not_normalized_ch1",
        "ground_truth": "heat_ground_truth"
    }, inplace=True)

    print(df.head())
    "Columns: datetime, heat_ground_truth, input_not_normalized_ch0, input_not_normalized_ch1"
    return df

def load_ozone_data(data_dir: str) -> pd.DataFrame:
    """Loads raw data"""
    file_name = f"simulation_data.csv"
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
        "input_not_normalized_ch1_arr": "input_not_normalized_ch1",
        "ground_truth": "ozone_ground_truth"
    }, inplace=True)

    print(df.head())
    "Columns: datetime, ozone_ground_truth, input_not_normalized_ch0, input_not_normalized_ch1"
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


def z_score(arr: np.ndarray, factor: float = 1.0, mean_val: Optional[float] = None, std_val: Optional[float] = None) -> np.ndarray:
    """
    Applies z-score normalization to the array and scales it by the given factor.
    """
    arr = np.array(arr, dtype=np.float32)
    
    if mean_val is None:
        mean_val = np.mean(arr)
    if std_val is None:
        std_val = np.std(arr)
    
    if std_val == 0:
        return np.zeros_like(arr)
    
    return ((arr - mean_val) / std_val) * factor


def apply_normalization(arr: np.ndarray, normalization: str, channel: bool) -> np.ndarray:
    """Applies the selected normalization method."""
    if normalization == "adjusted-min-max":
        return adjusted_min_max(arr)
    elif normalization == "min-max-sliding-window-60-min":
        if not channel:
            online_window_ch0.update(arr)
            return min_max(arr, online_window_ch0.get_min_value(), online_window_ch0.get_max_value(), factor)
        else:
            online_window_ch1.update(arr)
            return min_max(arr, online_window_ch1.get_min_value(), online_window_ch1.get_max_value(), factor)

    elif normalization == "z-score-sliding-window-60-min":
        if not channel:
            online_window_ch0.update(arr)
            return z_score(arr, factor, online_window_ch0.get_mean(), online_window_ch0.get_std())
        else:
            online_window_ch1.update(arr)
            return z_score(arr, factor, online_window_ch1.get_mean(), online_window_ch1.get_std())

    elif normalization == "min-max":
        return min_max(arr, -10, 10.0, factor)
    
    elif normalization == "z-score":
        return z_score(arr, factor)

    else:
        raise ValueError(f"Unsupported normalization method: {normalization}")


def online_experiment(classifier, df_input_not_normalized: pd.DataFrame, normalization: str, num_classes: int) -> pd.DataFrame:

    print("Running Online Experiment")

    df = df_input_not_normalized.copy()
    df["input_normalized_ch0"] = None
    df["input_normalized_ch1"] = None

    if num_classes == 2:
        df["classification_ch0"] = None
        df["classification_ch1"] = None

    if num_classes == 3:
        df["classification_ch0_idle"] = None
        df["classification_ch0_heat"] = None
        df["classification_ch0_ozone"] = None
        df["classification_ch1_idle"] = None
        df["classification_ch1_heat"] = None
        df["classification_ch1_ozone"] = None


    for index, row in df.iterrows():

        if isinstance(row["input_not_normalized_ch0"], (list, np.ndarray)):
            normalized_ch0 = apply_normalization(np.array(row["input_not_normalized_ch0"]), normalization, False)
            input_tensor_ch0 = torch.tensor(normalized_ch0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction_ch0 = classifier(input_tensor_ch0)

            df.at[index, "input_normalized_ch0"] = normalized_ch0.tolist()
            if num_classes == 2:
                # Extract the second value from the prediction list ([prob_class0, prob_class1])
                df.at[index, "classification_ch0"] = prediction_ch0.flatten().tolist()[1]
                # Use .at[] to store the list as a single object in the cell
            if num_classes == 3:
                df.at[index,"classification_ch0_idle"] = prediction_ch0.flatten().tolist()[0]
                df.at[index,"classification_ch0_heat"] = prediction_ch0.flatten().tolist()[1]
                df.at[index,"classification_ch0_ozone"] = prediction_ch0.flatten().tolist()[2]

        if isinstance(row["input_not_normalized_ch1"], (list, np.ndarray)):
            normalized_ch1 = apply_normalization(np.array(row["input_not_normalized_ch1"]), normalization, True)
            input_tensor_ch1 = torch.tensor(normalized_ch1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction_ch1 = classifier(input_tensor_ch1)

            df.at[index, "input_normalized_ch1"] = normalized_ch1.tolist()
            if num_classes == 2:
                df.at[index, "classification_ch1"] = prediction_ch1.flatten().tolist()[1]
            if num_classes == 3:
                df.at[index,"classification_ch1_idle"] = prediction_ch1.flatten().tolist()[0]
                df.at[index,"classification_ch1_heat"] = prediction_ch1.flatten().tolist()[1]
                df.at[index,"classification_ch1_ozone"] = prediction_ch1.flatten().tolist()[2]


    return df


def main(data_dir=None, classifier_dir=None, normalization=None, prefix=None, threshold=None, num_classes=None, objective=None):
    if data_dir is None or classifier_dir is None or normalization is None or prefix is None:
        parser = argparse.ArgumentParser(description="Test forcasting methods")
        parser.add_argument("--data_dir", required=True, type=str, help="Directory with raw files.")
        parser.add_argument("--classifier_dir", required=True, type=str, help="Directory with trained CNN.")
        parser.add_argument("--normalization", required=True, type=str, help="Normalization method.")
        parser.add_argument("--prefix", required=True, type=str, help="C1, basically choose the plant.")
        parser.add_argument("--threshold", required=False, type=float, default=0.8, help="Threshold for optimization")
        parser.add_argument("--num_classes", type=int, choices=[2, 3], default=2)
        parser.add_argument("--objective", type=str, choices=["temp", "ozone"], default=2)
        args = parser.parse_args()
        # Use parsed args for any parameters not passed to main()
        if data_dir is None: 
            data_dir = args.data_dir
        if classifier_dir is None: 
            classifier_dir = args.classifier_dir
        if normalization is None: 
            normalization = args.normalization.lower()
        if prefix is None: 
            prefix = args.prefix.upper()
        if threshold is None: 
            threshold = args.threshold
        if num_classes is None: 
            num_classes = args.num_classes
        if objective is None: 
            objective = args.objective

    classifier = load_classifier(classifier_dir)
    df_result = None
    if objective == "temp":
        df_input_not_normalized_temp = load_temp_data(data_dir, prefix)
        df_result = online_experiment(classifier, df_input_not_normalized_temp, normalization, num_classes)
    
    if objective == "ozone":
        df_input_not_normalized_ozone = load_ozone_data(data_dir)
        df_result = online_experiment(classifier, df_input_not_normalized_ozone, normalization, num_classes)

    df_result = smooth_classification(df_result, 100, num_classes)

    true_positive, false_positive, true_negative, false_negative = metrics(df_result, threshold, num_classes, objective)
    plot_data(df_result, threshold, normalization, num_classes, objective)

    return true_positive, false_positive, true_negative, false_negative



if __name__ == "__main__":
    main()