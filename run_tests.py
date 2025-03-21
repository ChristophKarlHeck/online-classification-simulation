import csv
import os
import logging
import main

# Configure logging to output messages with timestamp and level.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

header = [
    "threshold", 
    "true_positive_10", "false_positive_10", "true_negative_10", "false_negative_10",
    "true_positive_11", "false_positive_11", "true_negative_11", "false_negative_11",
    "true_positive_12", "false_positive_12", "true_negative_12", "false_negative_12",
    "true_positive_13", "false_positive_13", "true_negative_13", "false_negative_13",
    "true_positive_14", "false_positive_14", "true_negative_14", "false_negative_14"
]

data_dirs = [
    "/home/chris/experiment_data/10_2025_02_20-2025_02_27/classification",
    "/home/chris/experiment_data/11_2025_02_27-2025_03_04/classification",
    "/home/chris/experiment_data/12_2025_03_06-2025_03_10/classification",
    "/home/chris/experiment_data/13_2025_03_11-2025_03_14/classification",
    "/home/chris/experiment_data/14_2025_03_14-2025_03_21/classification"
]

normalization_dirs = [
    "/home/chris/online-classification-simulation/adjusted-min-max",
    "/home/chris/online-classification-simulation/min-max",
    "/home/chris/online-classification-simulation/min-max-sliding-window-60-min",
    "/home/chris/online-classification-simulation/z-score",
    "/home/chris/online-classification-simulation/z-score-sliding-window-60-min"
]

prefixes = ["C1", "C2"]

result_dir = "/home/chris/experiment_data/Test"
os.makedirs(result_dir, exist_ok=True)

for prefix in prefixes:
    for normalization_dir in normalization_dirs:
        # Extract the normalization name from the directory (e.g., "adjusted-min-max")
        normalization = normalization_dir.rstrip('/').split("/")[-1]

        # Create the filename and open the CSV file for writing.
        filename = f"{normalization}-{prefix}-results.csv"
        filepath = os.path.join(result_dir, filename)
        logging.info(f"Starting processing for file: {filename}")
        with open(filepath, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header to CSV file.
            writer.writerow(header)

            # Iterate over threshold values from 0.01 to 0.99 (inclusive)
            for i in range(1, 100):  # i from 1 to 99
                threshold = round(i / 100.0, 2)
                if i % 10 == 0:
                    logging.info(f"{filename}: Processing threshold {threshold}")

                # For the first data_dir (labeled with "10"), always use prefix "C1"
                tp10, fp10, tn10, fn10 = main.main(
                    data_dir=data_dirs[0],
                    classifier_dir=normalization_dir,
                    normalization=normalization,
                    prefix="C1",  # data_dirs[0] does not have a prefix C2
                    threshold=threshold
                )

                # For the remaining data_dirs, use the current prefix value.
                tp11, fp11, tn11, fn11 = main.main(
                    data_dir=data_dirs[1],
                    classifier_dir=normalization_dir,
                    normalization=normalization,
                    prefix=prefix,
                    threshold=threshold
                )
                tp12, fp12, tn12, fn12 = main.main(
                    data_dir=data_dirs[2],
                    classifier_dir=normalization_dir,
                    normalization=normalization,
                    prefix=prefix,
                    threshold=threshold
                )
                tp13, fp13, tn13, fn13 = main.main(
                    data_dir=data_dirs[3],
                    classifier_dir=normalization_dir,
                    normalization=normalization,
                    prefix=prefix,
                    threshold=threshold
                )
                tp14, fp14, tn14, fn14 = main.main(
                    data_dir=data_dirs[4],
                    classifier_dir=normalization_dir,
                    normalization=normalization,
                    prefix=prefix,
                    threshold=threshold
                )

                # Create the row of data to be written.
                row = [
                    threshold,
                    tp10, fp10, tn10, fn10,
                    tp11, fp11, tn11, fn11,
                    tp12, fp12, tn12, fn12,
                    tp13, fp13, tn13, fn13,
                    tp14, fp14, tn14, fn14
                ]
                writer.writerow(row)
        logging.info(f"Finished processing file: {filename}")
