"""

What it does:
  1. Loads trained Baseline and DANN checkpoints for each label budget.
  2. Fits one temperature value on the fixed PTB-XL validation split only.
  3. Evaluates each model before and after temperature scaling on:
       - PTB-XL test
       - Chapman
       - CPSC2018
  4. Uses corrected present-class AUROC and present-class Macro F1 for
     external datasets with absent classes.
  5. Saves a JSON and CSV summary for final reporting.

Important:
  - Target labels are not used to fit temperature.
  - Temperature scaling should mainly affect ECE/Brier, not predictions.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from helpers.model_resnet34 import ResNet34_1D
from helpers.model_resnet34_dann import ResNet34_1D_DANN


BASE_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"
DATA_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed"

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 32
LABEL_BUDGETS = [10, 25, 50, 100]


class ProcessedECGDataset(Dataset):
    def __init__(self, metadata_df):
        self.data_df = metadata_df.reset_index(drop=True)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        with np.load(row["file_path"], allow_pickle=True) as data:
            signal = data["signal"]

        signal = np.transpose(signal, (1, 0))
        label_idx = LABEL_MAP.get(row["super_class"], -1)
        return (
            torch.tensor(signal, dtype=torch.float32),
            torch.tensor(label_idx, dtype=torch.long),
        )


class TemperatureWrapper(nn.Module):
    def __init__(self, model, init_temperature=1.5):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.clamp(min=0.05)
        return logits / temperature

    def fit(self, val_loader, device, max_iter=50):
        self.to(device)
        self.model.eval()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Fitting T: collecting logits", leave=False):
                logits_list.append(self.model(inputs.to(device)))
                labels_list.append(labels)

        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        before_nll = criterion(logits, labels).item()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        after_nll = criterion(self.temperature_scale(logits), labels).item()

        return {
            "Temperature": float(self.temperature.item()),
            "Val_NLL_Before_TS": float(before_nll),
            "Val_NLL_After_TS": float(after_nll),
        }


def scan_external_dataset(processed_dir):
    rows = []
    skipped = 0

    for file_path in tqdm(glob.glob(os.path.join(processed_dir, "*.npz")),
                          desc=f"Scanning {os.path.basename(processed_dir)}",
                          leave=False):
        with np.load(file_path, allow_pickle=True) as data:
            superclasses = data["superclasses"]

        if len(superclasses) > 0 and superclasses[0] in LABEL_MAP:
            rows.append({"file_path": file_path, "super_class": superclasses[0]})
        else:
            skipped += 1

    df = pd.DataFrame(rows)
    print(f"{os.path.basename(processed_dir)}: valid={len(df)} skipped={skipped}")
    return df


def normalize_probabilities(y_prob):
    y_prob = np.asarray(y_prob, dtype=np.float64)
    row_sums = y_prob.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return y_prob / row_sums


def multiclass_brier_score(y_true, y_prob):
    y_prob = normalize_probabilities(y_prob)
    y_true_onehot = np.eye(y_prob.shape[1])[y_true]
    return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_prob = normalize_probabilities(y_prob)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return float(ece)


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = normalize_probabilities(y_prob)

    valid_mask = y_true >= 0
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]

    y_pred = np.argmax(y_prob, axis=1)
    class_counts = {
        class_name: int(np.sum(y_true == idx))
        for idx, class_name in enumerate(CLASS_NAMES)
    }

    present_indices = [
        idx for idx, class_name in enumerate(CLASS_NAMES)
        if class_counts[class_name] > 0
    ]
    absent_classes = [
        CLASS_NAMES[idx] for idx in range(NUM_CLASSES)
        if idx not in present_indices
    ]
    present_classes = [CLASS_NAMES[idx] for idx in present_indices]

    macro_f1_present = float(f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=present_indices,
        zero_division=0,
    ))
    macro_f1_full = float(f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    ))
    per_class_values = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    per_class_f1 = {
        CLASS_NAMES[idx]: float(per_class_values[idx])
        for idx in range(NUM_CLASSES)
    }

    macro_auroc = None
    auroc_note = "All classes present."
    if len(present_indices) < 2:
        auroc_note = "AUROC undefined: fewer than two classes present."
    else:
        y_prob_present = y_prob[:, present_indices]
        y_prob_present = normalize_probabilities(y_prob_present)
        remap = {old: new for new, old in enumerate(present_indices)}
        y_true_present = np.array([remap[label] for label in y_true], dtype=np.int64)
        macro_auroc = float(roc_auc_score(
            y_true_present,
            y_prob_present,
            multi_class="ovr",
            average="macro",
        ))
        if absent_classes:
            auroc_note = (
                f"Absent classes {absent_classes} excluded; "
                f"AUROC computed over {present_classes}."
            )

    return {
        "Macro_F1": macro_f1_present,
        "Macro_F1_Full_5Class": macro_f1_full,
        "Per_Class_F1": per_class_f1,
        "Macro_AUROC": macro_auroc,
        "Brier_Score": multiclass_brier_score(y_true, y_prob),
        "ECE": expected_calibration_error(y_true, y_prob),
        "Class_Counts": class_counts,
        "Classes_in_AUROC": present_classes,
        "Absent_Classes": absent_classes,
        "AUROC_Note": auroc_note,
    }


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
        logits = model(inputs.to(device))
        probs = torch.softmax(logits, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    return all_labels, all_probs


def load_model(model_type, weights_path, device):
    if model_type == "Baseline":
        model = ResNet34_1D(num_classes=NUM_CLASSES, input_channels=12)
    elif model_type == "DANN":
        model = ResNet34_1D_DANN(num_classes=NUM_CLASSES, input_channels=12)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def build_loaders():
    val_df = pd.read_csv(os.path.join(BASE_DIR, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(BASE_DIR, "test_split.csv"))
    chapman_df = scan_external_dataset(os.path.join(DATA_DIR, "Chapman"))
    cpsc_df = scan_external_dataset(os.path.join(DATA_DIR, "CPSC2018"))

    val_loader = DataLoader(ProcessedECGDataset(val_df), batch_size=BATCH_SIZE, shuffle=False)
    test_loaders = {
        "PTB-XL (In-Dist)": DataLoader(ProcessedECGDataset(test_df), batch_size=BATCH_SIZE, shuffle=False),
        "Chapman (Mild Shift)": DataLoader(ProcessedECGDataset(chapman_df), batch_size=BATCH_SIZE, shuffle=False),
        "CPSC 2018 (Strong Shift)": DataLoader(ProcessedECGDataset(cpsc_df), batch_size=BATCH_SIZE, shuffle=False),
    }
    return val_loader, test_loaders


def checkpoint_path(model_type, budget):
    if model_type == "Baseline":
        return os.path.join(BASE_DIR, f"best_baseline_resnet34_{budget}.pth")
    if model_type == "DANN":
        return os.path.join(BASE_DIR, f"best_dann_resnet34_{budget}.pth")
    raise ValueError(f"Unknown model type: {model_type}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_loader, test_loaders = build_loaders()
    results = []

    for model_type in ["Baseline", "DANN"]:
        for budget in LABEL_BUDGETS:
            weights_path = checkpoint_path(model_type, budget)
            if not os.path.exists(weights_path):
                print(f"[SKIP] Missing checkpoint: {weights_path}")
                continue

            print("\n" + "=" * 70)
            print(f"{model_type} | budget={budget}%")
            print(f"Checkpoint: {weights_path}")
            print("=" * 70)

            base_model = load_model(model_type, weights_path, device)
            calibrated_model = TemperatureWrapper(load_model(model_type, weights_path, device))
            temp_info = calibrated_model.fit(val_loader, device)
            print(
                f"Temperature={temp_info['Temperature']:.4f} | "
                f"Val NLL {temp_info['Val_NLL_Before_TS']:.4f} -> "
                f"{temp_info['Val_NLL_After_TS']:.4f}"
            )

            for dataset_name, loader in test_loaders.items():
                for calibration_state, model in [
                    ("No_TS", base_model),
                    ("TS", calibrated_model),
                ]:
                    labels, probs = collect_predictions(model, loader, device)
                    metrics = compute_metrics(labels, probs)
                    row = {
                        "Model": model_type,
                        "Budget_pct": budget,
                        "Calibration": calibration_state,
                        "Dataset": dataset_name,
                        **temp_info,
                        **metrics,
                    }
                    results.append(row)

                    auroc = row["Macro_AUROC"]
                    auroc_str = "N/A" if auroc is None else f"{auroc:.4f}"
                    print(
                        f"{calibration_state:5s} | {dataset_name:25s} | "
                        f"AUROC={auroc_str} F1={row['Macro_F1']:.4f} "
                        f"ECE={row['ECE']:.4f} Brier={row['Brier_Score']:.4f}"
                    )

    os.makedirs(BASE_DIR, exist_ok=True)
    json_path = os.path.join(BASE_DIR, "temperature_scaling_v2_results.json")
    csv_path = os.path.join(BASE_DIR, "temperature_scaling_v2_results.csv")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    flat_rows = []
    for row in results:
        flat = row.copy()
        flat["Per_Class_F1"] = json.dumps(flat["Per_Class_F1"], ensure_ascii=False)
        flat["Class_Counts"] = json.dumps(flat["Class_Counts"], ensure_ascii=False)
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
