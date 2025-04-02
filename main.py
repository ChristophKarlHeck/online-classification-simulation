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

def plot_data(df_classified: pd.DataFrame, threshold: float, normalization: str,  objective: str, validation_method: str) -> None:
    
    plant_id=999
    #------------------Prepare Data for Plot---------------------------------------#
    window_size = 100 # 100 = 10min
    if objective == "temp":
        pd.options.display.max_columns = None
        df_classified['datetime'] = pd.to_datetime(df_classified['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df_classified['datetime'] = df_classified['datetime'] + pd.Timedelta(hours=1)
    if objective == "ozone":
        pd.options.display.max_columns = None
        df_classified['datetime'] = pd.to_datetime(df_classified['datetime'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
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

    axs[0].axhline(y=threshold, color="black", linestyle="--", linewidth=1, label=f"Threshold: {threshold}")


    if objective == "temp":

        axs[0].fill_between(
            df_classified['datetime'], 0, 1.0, 
            where=(df_classified["heat_ground_truth"] == 1), 
            color='#DC143C', alpha=0.3, label="Stimulus application"
        )

        if validation_method == "both":
            # Scatter plot for classification
            axs[0].plot(df_classified['datetime'], df_classified["ch0_smoothed"], label="CH0", color="#FF0000")
            axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed"], label="CH1", color="#8B0000")

            high_both = (df_classified["ch0_smoothed"] > threshold) & (df_classified["ch1_smoothed"] > threshold)
            axs[0].fill_between(df_classified['datetime'], 0, 1.0, 
                where=(high_both),
                color='#722F37', alpha=0.3, label="Stimulus prediction")

        if validation_method == "min":
            # Scatter plot for classification
            min_smoothed = df_classified[["ch0_smoothed", "ch1_smoothed"]].min(axis=1)
            axs[0].plot(df_classified['datetime'], min_smoothed, label="Min of CH0 & CH1", color="#C50000")

            above_threshold = min_smoothed > threshold
            axs[0].fill_between(df_classified['datetime'], 0, 1.0, 
                where=(above_threshold),
                color='#722F37', alpha=0.3, label="Stimulus prediction")

        if validation_method == "max":
            # Scatter plot for classification
            max_smoothed = df_classified[["ch0_smoothed", "ch1_smoothed"]].max(axis=1)
            axs[0].plot(df_classified['datetime'], max_smoothed, label="Min of CH0 & CH1", color="#C50000")

            above_threshold = max_smoothed > threshold
            axs[0].fill_between(df_classified['datetime'], 0, 1.0, 
                where=(above_threshold),
                color='#722F37', alpha=0.3, label="Stimulus prediction")

        if validation_method == "mean":
            # Scatter plot for classification
            mean_smoothed = df_classified[["ch0_smoothed", "ch1_smoothed"]].mean(axis=1)
            axs[0].plot(df_classified['datetime'], mean_smoothed, label="Min of CH0 & CH1", color="#C50000")

            above_threshold = mean_smoothed > threshold
            axs[0].fill_between(df_classified['datetime'], 0, 1.0, 
                where=(above_threshold),
                color='#722F37', alpha=0.3, label="Stimulus prediction")

    if objective == "ozone":
        axs[0].fill_between(df_classified['datetime'], 0, 1.0, 
                        where=(df_classified["smoothed_ozone_mean"] > threshold),# & (df_classified["ch1_smoothed_heat"] > threshold), 
                        color='#000080', alpha=0.3, label="Stimulus prediction")
        
        axs[0].fill_between(
            df_classified['datetime'], 0, 1.0, 
            where=(df_classified["ozone_ground_truth"] == 1), 
            color='#4169E1', alpha=0.3, label="Stimulus application"
        )


    # Ensure y-axis limits and set explicit tick marks
    axs[0].set_ylim(0, 1.05)
    axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1])  # Explicitly set y-ticks
    axs[0].set_ylabel("Phase Probability",fontsize=10)
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

def smooth_classification(df_classified: pd.DataFrame, window_size: int) -> pd.DataFrame:

    df_classified["ch0_smoothed"] = df_classified["classification_ch0"].rolling(window=window_size, min_periods=1).mean()
    df_classified["ch1_smoothed"] = df_classified["classification_ch1"].rolling(window=window_size, min_periods=1).mean()

    return df_classified


def both_channels_higher_threshhold(df_classified: pd.DataFrame, objective: str, threshold: float):
    # Select the correct ground truth column based on the objective
    if objective == "temp":
        gt_col = "heat_ground_truth"
    elif objective == "ozone":
        gt_col = "ozone_ground_truth"
    else:
        raise ValueError("Objective must be either 'temp' or 'ozone'.")
    
    # Create a mask for cases where both channels are above the threshold
    high_both = (df_classified["ch0_smoothed"] > threshold) & (df_classified["ch1_smoothed"] > threshold)
    # Complementary mask: at least one channel is below or equal to threshold
    low_any = ~high_both

    # Calculate counts
    true_positive_cases = ((df_classified[gt_col] == 1) & high_both).sum()
    false_positive_cases = ((df_classified[gt_col] == 0) & high_both).sum()
    true_negative_cases = ((df_classified[gt_col] == 0) & low_any).sum()
    false_negative_cases = ((df_classified[gt_col] == 1) & low_any).sum()

    return true_positive_cases, false_positive_cases, true_negative_cases, false_negative_cases


def max_prob_higher_threshold(df_classified: pd.DataFrame, objective: str, threshold: float):
    # Select the correct ground truth column based on the objective
    if objective == "temp":
        gt_col = "heat_ground_truth"
    elif objective == "ozone":
        gt_col = "ozone_ground_truth"
    else:
        raise ValueError("Objective must be either 'temp' or 'ozone'.")

    # Compute the maximum probability between the two channels for each row
    max_prob = df_classified[["ch0_smoothed", "ch1_smoothed"]].max(axis=1)

    # Create a mask for when the maximum probability exceeds the threshold
    above_threshold = max_prob > threshold
    below_threshold = ~above_threshold  # Inverse of the mask

    # Count cases based on the ground truth and whether the max prob is above the threshold
    true_positive_cases = ((df_classified[gt_col] == 1) & above_threshold).sum()
    false_positive_cases = ((df_classified[gt_col] == 0) & above_threshold).sum()
    true_negative_cases = ((df_classified[gt_col] == 0) & below_threshold).sum()
    false_negative_cases = ((df_classified[gt_col] == 1) & below_threshold).sum()

    return true_positive_cases, false_positive_cases, true_negative_cases, false_negative_cases


def min_prob_higher_threshold(df_classified: pd.DataFrame, objective: str, threshold: float):
    # Select the correct ground truth column based on the objective
    if objective == "temp":
        gt_col = "heat_ground_truth"
    elif objective == "ozone":
        gt_col = "ozone_ground_truth"
    else:
        raise ValueError("Objective must be either 'temp' or 'ozone'.")
    
    # Compute the minimum probability between the two channels for each row
    min_prob = df_classified[["ch0_smoothed", "ch1_smoothed"]].min(axis=1)
    
    # Create a mask for when the minimum probability exceeds the threshold
    above_threshold = min_prob > threshold
    below_threshold = ~above_threshold  # Inverse of the mask

    # Count cases based on the ground truth and whether the min prob is above the threshold
    true_positive_cases = ((df_classified[gt_col] == 1) & above_threshold).sum()
    false_positive_cases = ((df_classified[gt_col] == 0) & above_threshold).sum()
    true_negative_cases = ((df_classified[gt_col] == 0) & below_threshold).sum()
    false_negative_cases = ((df_classified[gt_col] == 1) & below_threshold).sum()

    return true_positive_cases, false_positive_cases, true_negative_cases, false_negative_cases


def mean_prob_higher_threshold(df_classified: pd.DataFrame, objective: str, threshold: float):
    # Select the correct ground truth column based on the objective
    if objective == "temp":
        gt_col = "heat_ground_truth"
    elif objective == "ozone":
        gt_col = "ozone_ground_truth"
    else:
        raise ValueError("Objective must be either 'temp' or 'ozone'.")

    # Compute the mean probability between the two channels for each row
    mean_prob = df_classified[["ch0_smoothed", "ch1_smoothed"]].mean(axis=1)
    
    # Create a mask for when the mean probability exceeds the threshold
    above_threshold = mean_prob > threshold
    below_threshold = ~above_threshold  # Inverse of the mask

    # Count cases based on the ground truth and whether the mean prob is above the threshold
    true_positive_cases = ((df_classified[gt_col] == 1) & above_threshold).sum()
    false_positive_cases = ((df_classified[gt_col] == 0) & above_threshold).sum()
    true_negative_cases = ((df_classified[gt_col] == 0) & below_threshold).sum()
    false_negative_cases = ((df_classified[gt_col] == 1) & below_threshold).sum()

    return true_positive_cases, false_positive_cases, true_negative_cases, false_negative_cases


def metrics(df_classified: pd.DataFrame, threshold: float, objective: str, validation_method: str):

    true_positive_cases = 0
    false_positive_cases = 0
    true_negative_cases = 0
    false_negative_cases = 0

    if validation_method == "both":
        true_positive_cases, false_positive_cases, true_negative_cases, false_negative_cases = both_channels_higher_threshhold(df_classified, objective, threshold)

    if validation_method == "min":
        true_positive_cases, false_positive_cases, true_negative_cases, false_negative_cases = min_prob_higher_threshold(df_classified, objective, threshold)

    if validation_method == "max":
        true_positive_cases, false_positive_cases, true_negative_cases, false_negative_cases = max_prob_higher_threshold(df_classified, objective, threshold)

    if validation_method == "mean":
        true_positive_cases, false_positive_cases, true_negative_cases, false_negative_cases = mean_prob_higher_threshold(df_classified, objective, threshold)


    return true_positive_cases, false_positive_cases, true_negative_cases, false_negative_cases

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
    pd.options.display.max_columns = None
    print("Temp:", df.head())

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
    df = pd.read_csv(file_path, index_col=0)
    pd.options.display.max_columns = None
    print("Ozone:", df.head())

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


def online_experiment(classifier, df_input_not_normalized: pd.DataFrame, normalization: str) -> pd.DataFrame:

    print("Running Online Experiment")

    df = df_input_not_normalized.copy()
    df["input_normalized_ch0"] = None
    df["input_normalized_ch1"] = None
    df["classification_ch0"] = None
    df["classification_ch1"] = None

    for index, row in df.iterrows():

        if isinstance(row["input_not_normalized_ch0"], (list, np.ndarray)):
            normalized_ch0 = apply_normalization(np.array(row["input_not_normalized_ch0"]), normalization, False)
            input_tensor_ch0 = torch.tensor(normalized_ch0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction_ch0 = classifier(input_tensor_ch0)

            df.at[index, "input_normalized_ch0"] = normalized_ch0.tolist()
            df.at[index, "classification_ch0"] = prediction_ch0.flatten().tolist()[1]
                # Use .at[] to store the list as a single object in the cell

        if isinstance(row["input_not_normalized_ch1"], (list, np.ndarray)):
            normalized_ch1 = apply_normalization(np.array(row["input_not_normalized_ch1"]), normalization, True)
            input_tensor_ch1 = torch.tensor(normalized_ch1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction_ch1 = classifier(input_tensor_ch1)

            df.at[index, "input_normalized_ch1"] = normalized_ch1.tolist()
            df.at[index, "classification_ch1"] = prediction_ch1.flatten().tolist()[1]



    return df


def main(data_dir=None, classifier_dir=None, normalization=None, prefix=None, threshold=None, objective=None, validation_method=None):
    if data_dir is None or classifier_dir is None or normalization is None or prefix is None:
        parser = argparse.ArgumentParser(description="Test forcasting methods")
        parser.add_argument("--data_dir", required=True, type=str, help="Directory with raw files.")
        parser.add_argument("--classifier_dir", required=True, type=str, help="Directory with trained CNN.")
        parser.add_argument("--normalization", required=True, type=str, help="Normalization method.")
        parser.add_argument("--prefix", required=True, type=str, help="C1, basically choose the plant.")
        parser.add_argument("--threshold", required=False, type=float, default=0.8, help="Threshold for optimization")
        parser.add_argument("--objective", type=str, choices=["temp", "ozone"], default=2)
        parser.add_argument("--validation_method", type=str, choices=["both", "min", "max", "mean"], default=2)
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
        if objective is None: 
            objective = args.objective
        if validation_method is None: 
            validation_method = args.validation_method

    classifier = load_classifier(classifier_dir)
    df_result = None
    if objective == "temp":
        df_input_not_normalized_temp = load_temp_data(data_dir, prefix)
        df_result = online_experiment(classifier, df_input_not_normalized_temp, normalization)
    
    if objective == "ozone":
        df_input_not_normalized_ozone = load_ozone_data(data_dir)
        df_result = online_experiment(classifier, df_input_not_normalized_ozone, normalization)

    df_result = smooth_classification(df_result, 100)

    true_positive, false_positive, true_negative, false_negative = metrics(df_result, threshold, objective, validation_method)
    print(validation_method)
    plot_data(df_result, threshold, normalization, objective, validation_method)

    return true_positive, false_positive, true_negative, false_negative



if __name__ == "__main__":
    main()