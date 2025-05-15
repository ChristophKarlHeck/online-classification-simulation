import csv
import os
import logging
import main
import re


_RULES = [
    (re.compile(r"^none_\d+$"), "None"),
    (re.compile(r"^mm_\d+$"),   "min-max"),
    (re.compile(r"^amm_\d+$"),  "adjusted-min-max"),
    (re.compile(r"^Z-score_\d+$"),  "z-score"),
]

def map_path_tag(path_str: str) -> str:
    tag = os.path.basename(path_str.rstrip(os.sep))  # 'none_1', 'mm_1000', …

    for regex, replacement in _RULES:
        if regex.fullmatch(tag):
            return replacement

    raise ValueError(f"Unrecognised tag: {tag!r}")

# Configure logging to output messages with timestamp and level.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

header = [
    "threshold", 
    "true_positive_10", "false_positive_10", "true_negative_10", "false_negative_10",
    "true_positive_11", "false_positive_11", "true_negative_11", "false_negative_11",
    # "true_positive_12", "false_positive_12", "true_negative_12", "false_negative_12",
    # "true_positive_13", "false_positive_13", "true_negative_13", "false_negative_13",
    "true_positive_14", "false_positive_14", "true_negative_14", "false_negative_14"
]

data_dirs = [
    "/home/chris/experiment_data/10_2025_02_20-2025_02_27/classification",
    "/home/chris/experiment_data/11_2025_02_27-2025_03_04/classification",
    #"/home/chris/experiment_data/12_2025_03_06-2025_03_10/classification",
    #"/home/chris/experiment_data/13_2025_03_11-2025_03_14/classification",
    "/home/chris/experiment_data/14_2025_03_14-2025_03_21/classification"
]

normalization_dirs = [
    "/home/chris/online-classification-simulation/FCN_temperature/local_10/mm_1",
    "/home/chris/online-classification-simulation/FCN_temperature/local_10/Z-score_1",
    "/home/chris/online-classification-simulation/FCN_temperature/local_10/amm_1000",
    "/home/chris/online-classification-simulation/FCN_temperature/local_10/mm_1000",
     "/home/chris/online-classification-simulation/FCN_temperature/local_10/Z-score_1000",
]

validation_methods = ["max", "min", "mean"]

prefixes = ["C1", "C2"]

objective = "temp"

result_dir = "/home/chris/experiment_data/Test"
os.makedirs(result_dir, exist_ok=True)

threshold = 0.5

for prefix in [prefixes[0]]:
    for validation_method in validation_methods:
        for normalization_dir in normalization_dirs:
            # Extract the normalization name from the directory (e.g., "adjusted-min-max")
            model = normalization_dir.rstrip('/').split("/")[-1]
            normalization = map_path_tag(normalization_dir)

            # Create the filename and open the CSV file for writing.
            filename = f"{model}-{prefix}-{validation_method}-results.csv"
            filepath = os.path.join(result_dir, validation_method)
            os.makedirs(filepath, exist_ok=True)    
            filepath_name = os.path.join(filepath, filename)
            logging.info(f"Starting processing for file: {filename}")
            with open(filepath_name, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # Write header to CSV file.
                writer.writerow(header)

                    # For the first data_dir (labeled with "10"), always use prefix "C1"
                tp10, fp10, tn10, fn10 = main.main(
                        data_dir=data_dirs[0],
                        classifier_dir=normalization_dir,
                        normalization=normalization,
                        prefix="C1",  # data_dirs[0] does not have a prefix C2
                        threshold=threshold,
                        objective=objective,
                        validation_method=validation_method
                    )

                    # For the remaining data_dirs, use the current prefix value.
                tp11, fp11, tn11, fn11 = main.main(
                        data_dir=data_dirs[1],
                        classifier_dir=normalization_dir,
                        normalization=normalization,
                        prefix=prefix,
                        threshold=threshold,
                        objective=objective,
                        validation_method=validation_method
                    )
                # tp12, fp12, tn12, fn12 = main.main(
                #         data_dir=data_dirs[2],
                #         classifier_dir=normalization_dir,
                #         normalization=normalization,
                #         prefix=prefix,
                #         threshold=threshold,
                #         objective=objective,
                #         validation_method=validation_method
                #     )
                # tp13, fp13, tn13, fn13 = main.main(
                #         data_dir=data_dirs[3],
                #         classifier_dir=normalization_dir,
                #         normalization=normalization,
                #         prefix=prefix,
                #         threshold=threshold,
                #         objective=objective,
                #         validation_method=validation_method
                #     )
                tp14, fp14, tn14, fn14 = main.main(
                        data_dir=data_dirs[2],
                        classifier_dir=normalization_dir,
                        normalization=normalization,
                        prefix=prefix,
                        threshold=threshold,
                        objective=objective,
                        validation_method=validation_method
                    )

                    # Create the row of data to be written.
                row = [
                        threshold,
                        tp10, fp10, tn10, fn10,
                        tp11, fp11, tn11, fn11,
                        # tp12, fp12, tn12, fn12,
                        # tp13, fp13, tn13, fn13,
                        tp14, fp14, tn14, fn14
                    ]
                writer.writerow(row)
                logging.info(f"Wrote row: {threshold}")
            logging.info(f"Finished processing file: {filename}")
