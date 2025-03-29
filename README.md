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

### Z-score sliding window

```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/z-score-sliding-window --normalization z-score-sliding-window-60-min --prefix C1  --threshold 0.3 --num_classes 3
```

### Adjusted Min-Max Temp

```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/adjusted-min-max --normalization adjusted-min-max --prefix C1  --threshold 0.33 --num_classes 3 --objective temp
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/11_2025_02_27-2025_03_04/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/adjusted-min-max --normalization adjusted-min-max --prefix C1  --threshold 0.33 --num_classes 3 --objective temp
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/12_2025_03_06-2025_03_10/classification  --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/adjusted-min-max --normalization adjusted-min-max --prefix C1  --threshold 0.33 --num_classes 3 --objective temp
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/13_2025_03_11-2025_03_14/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/adjusted-min-max --normalization adjusted-min-max --prefix C1  --threshold 0.33 --num_classes 3 --objective temp
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/14_2025_03_14-2025_03_21/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/adjusted-min-max --normalization adjusted-min-max --prefix C1  --threshold 0.33 --num_classes 3 --objective temp
```

### Z-score Temp
- so far, best result (avg of probabilites)
```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/z-score --normalization z-score --prefix C1  --threshold 0.33 --num_classes 3 --objective temp
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/11_2025_02_27-2025_03_04/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/z-score --normalization z-score --prefix C1  --threshold 0.33 --num_classes 3 --objective temp
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/12_2025_03_06-2025_03_10/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/z-score --normalization z-score --prefix C1  --threshold 0.33 --num_classes 3 --objective temp
```

- also good with mean of probabilties
```bash
python3 main.py --data_dir /home/chris/experiment_data/13_2025_03_11-2025_03_14/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/z-score --normalization z-score --prefix C1  --threshold 0.33 --num_classes 3 --objective temp
```

```bash
python3 main.py --data_dir /home/chris/experiment_data/14_2025_03_14-2025_03_21/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/z-score --normalization z-score --prefix C1  --threshold 0.33 --num_classes 3 --objective temp
```

### Z-score Ozone
- so far, best result (avg of probabilites)
```bash
python3 main.py --data_dir /home/chris/experiment_data/10_2025_02_20-2025_02_27/classification --classifier_dir /home/chris/online-classification-simulation/FCN_temp_ozone/z-score --normalization z-score --prefix C1  --threshold 0.33 --num_classes 3 --objective ozone
```