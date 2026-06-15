# AIE683 Cross-Dataset ECG Classification - Run Package

This folder contains numbered and more clearly named copies of the code files required to run the project from start to finish.
> **Note:** Dataset folders are not included in this package. The dataset folders can be accessed via the links below:
>
> * [PTB-XL](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset?resource=download)
> * [CPSC2018](https://www.kaggle.com/datasets/physionet/china-physiological-signal-challenge-in-2018)
> * [Chapman](https://www.kaggle.com/datasets/erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database)


## Folder Contents

| Step | File | Purpose |
|---:|---|---|
| 00 | `00_model_resnet34.py` | 1D ResNet-34 model architecture. |
| 00 | `00_model_resnet34_dann.py` | DANN version: feature extractor + label classifier + domain discriminator. |
| 01 | `01_preprocess_ecg.py` | Processes raw ECG data and converts it into 10 s / 500 Hz / 12-lead `.npz` files. |
| 02 | `02_create_splits_and_label_budgets.py` | Creates patient-based train/validation/test splits and 10/25/50/100 label-budget CSV files for PTB-XL. |
| 03 | `03_train_baseline_resnet34.py` | Trains the baseline ResNet-34 model and saves checkpoints according to the selected budget. |
| 04 | `04_train_dann_resnet34.py` | Trains the DANN model. PTB-XL is used as the source domain, while CPSC/Chapman is used as target-domain input. |
| 05 | `05_evaluate_cross_dataset_budget_checkpoints.py` | Evaluates Baseline and DANN budget checkpoints on PTB-XL, Chapman, and CPSC. |
| 06 | `06_temperature_scaling_evaluation.py` | Applies Temperature Scaling and compares No_TS and TS results. |
| metadata | `metadata/snomed_to_superclass.xlsx` | Mapping file used to convert Chapman/CPSC SNOMED labels into the PTB-XL superclass system. |

## Paths That Must Be Updated On Another Computer

If another user runs this package on a different computer, they must update the hardcoded paths inside the scripts. The most important paths are listed below.

### In `01_preprocess_ecg.py`

Update the raw dataset locations:

```python
PATHS = {
    'PTB-XL': r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\PTB-XL\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
    'Chapman': r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Chapman\WFDB_ChapmanShaoxing",
    'CPSC2018': r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\CPSC2018"
}
```

Update the processed dataset output folders:

```python
OUTPUT_DIRS = {
    "PTB-XL": r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\PTB-XL",
    "Chapman": r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\Chapman",
    "CPSC2018": r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\CPSC2018",
}
```

Update the mapping file and PTB-XL metadata CSV paths:

```python
EXCEL_MAPPING_PATH = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING\snomed_to_superclass.xlsx"
PTBXL_CSV_PATH = os.path.join(PATHS["PTB-XL"], "ptbxl_database.csv")
```

### In `02_create_splits_and_label_budgets.py`

Update these paths:

```python
PROCESSED_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\PTB-XL"
ORIGINAL_CSV = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\PTB-XL\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1\ptbxl_database.csv"
SAVE_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"
```

### In `03_train_baseline_resnet34.py`

Update the result/checkpoint directory:

```python
csv_dir = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"
```

For each label budget, update the training split and checkpoint name:

```python
train_df = pd.read_csv(os.path.join(csv_dir, "train_split_10.csv"))
save_path = os.path.join(csv_dir, 'best_baseline_resnet34_10.pth')
metrics_save_path = os.path.join(csv_dir, "baseline_test_metrics_10.json")
```

Change `10` to `25`, `50`, or `100` for the other budget runs.

### In `04_train_dann_resnet34.py`

Update the result/checkpoint directory:

```python
csv_dir = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"
```

Update the source training split according to the budget:

```python
source_train_df = pd.read_csv(os.path.join(csv_dir, "train_split_10.csv"))
source_val_df = pd.read_csv(os.path.join(csv_dir, "val_split.csv"))
```

Update the target-domain processed dataset folder:

```python
cpsc_dir = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\CPSC2018"
```

Update the DANN checkpoint name for each budget:

```python
save_path = os.path.join(csv_dir, 'best_dann_resnet34_10.pth')
```

Change `10` to `25`, `50`, or `100` for the other budget runs.

### In `05_evaluate_cross_dataset_budget_checkpoints.py`

This file imports shared paths from `06_temperature_scaling_evaluation.py` through `step06_temperature_scaling_evaluation.py`. Therefore, update the base paths in `06_temperature_scaling_evaluation.py` and `step06_temperature_scaling_evaluation.py`.

### In `06_temperature_scaling_evaluation.py` and `step06_temperature_scaling_evaluation.py`

Update:

```python
BASE_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"
DATA_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed"
```

`BASE_DIR` should point to the folder containing:

- `train_split_10.csv`
- `train_split_25.csv`
- `train_split_50.csv`
- `train_split_100.csv`
- `val_split.csv`
- `test_split.csv`
- model checkpoint `.pth` files

`DATA_DIR` should point to the folder containing:

- `PTB-XL`
- `Chapman`
- `CPSC2018`

## Run Order

First, navigate to this folder in PowerShell:

```powershell
cd "C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING\AIE683_run_package_numbered"
```

### 1. Preprocessing

Creates processed `.npz` files from the raw datasets.

```powershell
python "01_preprocess_ecg.py"
```

Run this step only if the processed datasets are missing or if you want to redo preprocessing from scratch.

### 2. Train / Validation / Test Splits and Label-Budget CSV Files

```powershell
python "02_create_splits_and_label_budgets.py"
```

Main generated files:

- `train_split_10.csv`
- `train_split_25.csv`
- `train_split_50.csv`
- `train_split_100.csv`
- `val_split.csv`
- `test_split.csv`

### 3. Baseline Training

```powershell
python "03_train_baseline_resnet34.py"
```

Important: The budget to be trained is determined inside this file using the `train_split_XX.csv` and `best_baseline_resnet34_XX.pth` lines. It should be run separately for 10, 25, 50, and 100.

Expected checkpoints:

- `best_baseline_resnet34_10.pth`
- `best_baseline_resnet34_25.pth`
- `best_baseline_resnet34_50.pth`
- `best_baseline_resnet34_100.pth`

### 4. DANN Training

```powershell
python "04_train_dann_resnet34.py"
```

Important: For DANN, the source training split changes according to the budget, while the validation set stays fixed. Target-domain labels are not used during training; only the input signals are used for domain adaptation.

Expected checkpoints:

- `best_dann_resnet34_10.pth`
- `best_dann_resnet34_25.pth`
- `best_dann_resnet34_50.pth`
- `best_dann_resnet34_100.pth`

### 5. Cross-Dataset Evaluation

```powershell
python "05_evaluate_cross_dataset_budget_checkpoints.py"
```

This step evaluates Baseline and DANN checkpoints on:

- PTB-XL test split
- Chapman
- CPSC2018

Output files:

- `cross_dataset_results_budget_checkpoints.json`
- `cross_dataset_results_budget_checkpoints.csv`

### 6. Temperature Scaling Evaluation

```powershell
python "06_temperature_scaling_evaluation.py"
```

For each checkpoint, this step:

1. Fits the temperature value `T` on the PTB-XL validation set.
2. Compares No_TS and TS results on PTB-XL, Chapman, and CPSC.

Output files:

- `temperature_scaling_results.json`
- `temperature_scaling_results.csv`

## Metric Policy

Since some superclasses are absent in the external datasets, Macro AUROC is computed only over the classes present in the test set. Therefore:

- Missing classes should be reported for Chapman.
- Missing classes should be reported for CPSC2018.
- Brier Score and ECE are computed using the model’s full five-class probability vector.

## Notes

- Dataset paths inside the code still point to the original locations.
- If you run the code on another computer, you must update the paths listed above.
- `03_train_baseline_resnet34.py` and `04_train_dann_resnet34.py` must be run manually for each budget: 10/25/50/100.
- The result files used in the final report are `cross_dataset_results_budget_checkpoints.csv` and `temperature_scaling_results.csv`.
