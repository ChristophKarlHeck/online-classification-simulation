# online-classification-simulation

## CNN Temperature
```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/CNN_temp/adjusted-min-max --normalization adjusted-min-max --prefix C1
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/CNN_temp/min-max-sliding-window-60-min --normalization min-max-sliding-window-60-min --prefix C1
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/CNN_temp/adjusted-min-max --normalization z-score --prefix C1
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/CNN_temp/adjusted-min-max --normalization z-score --prefix C1
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/CNN_temp/z-score-sliding-window-60-min --normalization z-score-sliding-window-60-min --prefix C1
```

## FCN Temperature and Ozone

```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/z-score-sliding-window --normalization z-score-sliding-window-60-min --prefix C1  --threshold 0.3 --num_classes 3
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/adjusted-min-max --normalization adjusted-min-max --prefix C1  --threshold 0.3 --num_classes 3
```