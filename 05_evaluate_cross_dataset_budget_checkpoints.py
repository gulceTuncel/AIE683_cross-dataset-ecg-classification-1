"""
Cross-dataset evaluation for budget-specific Baseline and DANN checkpoints.

This script is separate from the older cross_dataset_test*.py files.
It does not require the old best_baseline_resnet34.pth checkpoint.

Expected checkpoints:
  best_baseline_resnet34_10.pth
  best_baseline_resnet34_25.pth
  best_baseline_resnet34_50.pth
  best_baseline_resnet34_100.pth
  best_dann_resnet34_10.pth
  best_dann_resnet34_25.pth
  best_dann_resnet34_50.pth
  best_dann_resnet34_100.pth
"""

import json
import os

import pandas as pd
from torch.utils.data import DataLoader

from step06_temperature_scaling_evaluation import (
    BASE_DIR,
    DATA_DIR,
    BATCH_SIZE,
    LABEL_BUDGETS,
    ProcessedECGDataset,
    collect_predictions,
    compute_metrics,
    load_model,
    scan_external_dataset,
)

import torch


def checkpoint_path(model_type, budget):
    if model_type == "Baseline":
        return os.path.join(BASE_DIR, f"best_baseline_resnet34_{budget}.pth")
    if model_type == "DANN":
        return os.path.join(BASE_DIR, f"best_dann_resnet34_{budget}.pth")
    raise ValueError(f"Unknown model type: {model_type}")


def build_test_loaders():
    ptbxl_test_df = pd.read_csv(os.path.join(BASE_DIR, "test_split.csv"))
    chapman_df = scan_external_dataset(os.path.join(DATA_DIR, "Chapman"))
    cpsc_df = scan_external_dataset(os.path.join(DATA_DIR, "CPSC2018"))

    return {
        "PTB-XL (In-Dist)": DataLoader(
            ProcessedECGDataset(ptbxl_test_df),
            batch_size=BATCH_SIZE,
            shuffle=False,
        ),
        "Chapman (Mild Shift)": DataLoader(
            ProcessedECGDataset(chapman_df),
            batch_size=BATCH_SIZE,
            shuffle=False,
        ),
        "CPSC 2018 (Strong Shift)": DataLoader(
            ProcessedECGDataset(cpsc_df),
            batch_size=BATCH_SIZE,
            shuffle=False,
        ),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_loaders = build_test_loaders()
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

            model = load_model(model_type, weights_path, device)

            for dataset_name, loader in test_loaders.items():
                labels, probs = collect_predictions(model, loader, device)
                metrics = compute_metrics(labels, probs)

                row = {
                    "Model": model_type,
                    "Budget_pct": budget,
                    "Dataset": dataset_name,
                    "Checkpoint": weights_path,
                    **metrics,
                }
                results.append(row)

                auroc = row["Macro_AUROC"]
                auroc_str = "N/A" if auroc is None else f"{auroc:.4f}"
                print(
                    f"{dataset_name:25s} | "
                    f"AUROC={auroc_str} "
                    f"F1={row['Macro_F1']:.4f} "
                    f"ECE={row['ECE']:.4f} "
                    f"Brier={row['Brier_Score']:.4f}"
                )

    json_path = os.path.join(BASE_DIR, "cross_dataset_results_v5_budget_checkpoints.json")
    csv_path = os.path.join(BASE_DIR, "cross_dataset_results_v5_budget_checkpoints.csv")

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
