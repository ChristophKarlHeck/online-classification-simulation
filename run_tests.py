import csv
import main

# Provide parameters directly; these will override command-line parsing in main()
true_positive, false_positive, true_negative, false_negative = main.main(
    data_dir="/home/chris/experiment_data/10_2025_02_20-2025_02_27/classification",
    classifier_dir="/home/chris/online-classification-simulation/adjusted-min-max",
    normalization="adjusted-min-max",
    prefix="C1",
    threshold=0.8
)