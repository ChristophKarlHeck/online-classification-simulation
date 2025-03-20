import csv
import main

# Define the header columns
header = [
    "threshold",
    "true_positive_10", "false_positive_10", "true_negative_10", "false_negative_10",
    "true_positive_11", "false_positive_11", "true_negative_11", "false_negative_11",
    "true_positive_12", "false_positive_12", "true_negative_12", "false_negative_12",
    "true_positive_13", "false_positive_13", "true_negative_13", "false_negative_13",
    "true_positive_14", "false_positive_14", "true_negative_14", "false_negative_14"
]

csv_filename = "results.csv" # should include the normalization

with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

# 14 must be added
data_dirs = [
    "/home/chris/experiment_data/10_2025_02_20-2025_02_27/classification",
    "/home/chris/experiment_data/11_2025_02_27-2025_03_04/classification",
    "/home/chris/experiment_data/12_2025_03_06-2025_03_10/classification",
    "/home/chris/experiment_data/13_2025_03_11-2025_03_14/classification",
]

normalization_dirs = [
    "adjusted-min-max",
    "min-max",
    "min-max-sliding-window-60-min",
    "z-score",
    "z-score-sliding-window-60-min"
]

# Example: iterate over the data directories and print them
for dir_path in data_dirs:
    print("Processing directory:", dir_path)

for normalization in normalization_dirs:
    

# Provide parameters directly; these will override command-line parsing in main()
true_positive, false_positive, true_negative, false_negative = main.main(
    data_dir="/home/chris/experiment_data/10_2025_02_20-2025_02_27/classification",
    classifier_dir="/home/chris/online-classification-simulation/adjusted-min-max",
    normalization="adjusted-min-max",
    prefix="C1",
    threshold=0.8
)