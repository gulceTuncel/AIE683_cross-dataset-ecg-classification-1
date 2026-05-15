"""
ablation_study.py  —  Systematic Ablation Study
================================================
Project: Cross-Dataset ECG Generalization Under Distribution Shift
Authors: E. Betül Parlak & Ş. Gülce Tuncel

Evaluates all four model configurations from the proposal across every
combination of label budget and test dataset, then writes a consolidated
results table for the paper.

Configuration matrix (2 × 2):
  ┌─────────────────┬──────────────────────┬────────────────────────┐
  │                 │   No calibration     │   Temperature Scaling  │
  ├─────────────────┼──────────────────────┼────────────────────────┤
  │ Baseline        │  CONFIG A            │  CONFIG C (Base+TS)    │
  │ DANN            │  CONFIG B            │  CONFIG D (DANN+TS)    │
  └─────────────────┴──────────────────────┴────────────────────────┘

Label budgets tested  : 10%, 25%, 50%, 100%
Test datasets         : PTB-XL test set | Chapman | CPSC 2018

Outputs (all written to BASE_DIR):
  ablation_results_full.json    —  raw results for every (config, budget, dataset) cell
  ablation_summary_table.csv   —  wide pivot table ready for the paper
  label_budget_curve.csv        —  AUROC vs budget per config × dataset (for figure)
  dataset_label_distributions.csv  —  class counts per external dataset (diagnostic)

FIXES vs. v1
  - AUROC is now computed only over classes actually present in y_true.
    sklearn raises ValueError when a one-vs-rest class has 0 positives;
    the old except returned 0.0 which is misleading. The new implementation
    passes `labels=present_classes` so absent classes are excluded rather
    than causing a silent failure.
  - Dataset label distributions are printed and saved at startup so label-
    harmonization gaps are visible before any model is loaded.
  - Per-class F1 is added to the output to expose class-collapse at low budgets.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

from resNet34      import ResNet34_1D
from resnet34_dann import ResNet34_1D_DANN

# =====================================================================
# 0. GLOBAL AYARLAR
# =====================================================================
BASE_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"
DATA_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed"

BASELINE_WEIGHTS_TEMPLATE = os.path.join(BASE_DIR, "best_baseline_resnet34_{pct}.pth")
DANN_WEIGHTS_TEMPLATE     = os.path.join(BASE_DIR, "best_dann_resnet34_{pct}.pth")

# True  → tüm bütçe seviyeleri için tek model kullan (hızlı test)
# False → her bütçe seviyesi kendi ağırlık dosyasını arar, yoksa fallback
SINGLE_WEIGHTS_MODE = False
FALLBACK_BASELINE   = os.path.join(BASE_DIR, "best_baseline_resnet34.pth")
FALLBACK_DANN       = os.path.join(BASE_DIR, "best_dann_resnet34.pth")

LABEL_BUDGETS = [10, 25, 50, 100]
LABEL_MAP     = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES   = 5
BATCH_SIZE    = 32

# =====================================================================
# 1. VERİ SETİ SINIFI
# =====================================================================
class Processed_ECG_Dataset(Dataset):
    def __init__(self, metadata_df):
        self.data_df = metadata_df.reset_index(drop=True)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        with np.load(row['file_path'], allow_pickle=True) as d:
            signal = d['signal']                   # (5000, 12)
        signal    = np.transpose(signal, (1, 0))   # → (12, 5000)
        label_idx = LABEL_MAP.get(row['super_class'], -1)
        return (
            torch.tensor(signal,    dtype=torch.float32),
            torch.tensor(label_idx, dtype=torch.long)
        )


# =====================================================================
# 2. EXTERNAL DATASET TARAYICI + ETİKET DAĞILIM RAPORU
# =====================================================================
def scan_external_dataset(processed_dir, dataset_name=""):
    """
    Klasörü tarar, UNKNOWN etiketleri eler ve etiket dağılımını raporlar.
    Harmonizasyon kalitesini değerlendirmek için kritik bir adımdır.
    """
    npz_files = glob.glob(os.path.join(processed_dir, "*.npz"))
    data_list, skipped = [], 0

    for fp in tqdm(npz_files, desc=f"Taranıyor: {dataset_name}", leave=False):
        with np.load(fp, allow_pickle=True) as d:
            sc = d['superclasses']
        if len(sc) > 0 and sc[0] != 'UNKNOWN' and sc[0] in LABEL_MAP:
            data_list.append({'super_class': sc[0], 'file_path': fp})
        else:
            skipped += 1

    df = pd.DataFrame(data_list)

    # --- Etiket dağılımını yazdır ---
    print(f"\n  [{dataset_name}] Geçerli: {len(df)} | Atlanan: {skipped}")
    if len(df) > 0:
        dist = df['super_class'].value_counts()
        for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
            cnt = dist.get(cls, 0)
            bar = '█' * int(cnt / max(dist.values) * 20)
            print(f"    {cls:5s}: {cnt:5d}  {bar}")
        absent = [c for c in LABEL_MAP if c not in dist.index]
        if absent:
            print(f"    ⚠ Bu veri setinde BULUNMAYAN sınıflar: {absent}")
            print(f"      → Bu sınıflar AUROC hesabından hariç tutulacak.")

    return df


def print_and_save_all_distributions(datasets_dict, save_dir):
    """Tüm veri setlerinin etiket dağılımını tek CSV'ye kaydeder."""
    rows = []
    for ds_name, df in datasets_dict.items():
        if df is None or len(df) == 0:
            continue
        for cls, cnt in df['super_class'].value_counts().items():
            rows.append({'Dataset': ds_name, 'Class': cls, 'Count': int(cnt)})
    dist_df = pd.DataFrame(rows)
    out_path = os.path.join(save_dir, "dataset_label_distributions.csv")
    dist_df.to_csv(out_path, index=False)
    print(f"\n  Etiket dağılımları kaydedildi: {out_path}")
    return dist_df


# =====================================================================
# 3. DEĞERLENDİRME METRİKLERİ  (AUROC HATASI DÜZELTİLDİ)
# =====================================================================
def multiclass_brier_score(y_true, y_prob):
    oh = np.eye(y_prob.shape[1])[y_true]
    return float(np.mean(np.sum((y_prob - oh) ** 2, axis=1)))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    conf  = np.max(y_prob, axis=1)
    pred  = np.argmax(y_prob, axis=1)
    acc   = (pred == y_true).astype(float)
    ece   = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        prop = np.mean(mask)
        if prop > 0:
            ece += np.abs(np.mean(acc[mask]) - np.mean(conf[mask])) * prop
    return float(ece)


def compute_metrics(y_true, y_prob):
    """
    4 metriği hesaplar.

    AUROC DÜZELTMESİ:
    sklearn.roc_auc_score(multi_class='ovr') bir sınıf y_true'da hiç
    yoksa ValueError fırlatır. Eski kod bunu 0.0 olarak döndürüyordu —
    bu yanlıştır (0.0 AUROC, mükemmel ters tahmin anlamına gelir).

    Düzeltme: y_true'daki mevcut sınıfları tespit et ve AUROC'u sadece
    bu sınıflar üzerinde hesapla. Hariç tutulan sınıflar raporda
    not olarak gösterilir.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    present_classes = np.unique(y_true)
    n_present       = len(present_classes)

    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    # --- Per-class F1 (class collapse tespiti için) ---
    per_class_f1 = f1_score(y_true, y_pred, average=None,
                            labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class_f1_dict = {INV_LABEL_MAP[i]: round(float(per_class_f1[i]), 4)
                         for i in range(NUM_CLASSES)}

    # --- Düzeltilmiş AUROC ---
    if n_present < 2:
        macro_auroc = float('nan')
        auroc_note  = f"Sadece {n_present} sınıf mevcut — AUROC hesaplanamaz."
    else:
        # labels= parametresi sklearn'e hangi sınıfların bekleneceğini söyler.
        # Mevcut olmayan sınıflar OVR hesabından otomatik hariç tutulur.
        try:
            macro_auroc = float(roc_auc_score(
                y_true,
                y_prob[:, present_classes],  # sadece mevcut sınıfların olasılıkları
                multi_class='ovr',
                average='macro',
                labels=list(range(n_present))
            ))
            absent = [INV_LABEL_MAP[c] for c in range(NUM_CLASSES)
                      if c not in present_classes]
            auroc_note = (f"Hariç tutulan sınıflar: {absent}" if absent
                          else "Tüm sınıflar mevcut.")
        except Exception as e:
            macro_auroc = float('nan')
            auroc_note  = f"AUROC hatası: {e}"

    return {
        'Macro_F1'       : macro_f1,
        'Macro_AUROC'    : macro_auroc,
        'Brier_Score'    : multiclass_brier_score(y_true, y_prob),
        'ECE'            : expected_calibration_error(y_true, y_prob),
        'Per_Class_F1'   : per_class_f1_dict,
        'AUROC_Note'     : auroc_note,
        'N_Classes_Present': int(n_present),
    }


# =====================================================================
# 4. TEMPERATURE SCALING
# =====================================================================
class TemperatureWrapper(nn.Module):
    def __init__(self, model, init_T=1.5):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.tensor([init_T]))

    def forward(self, x):
        return self.model(x) / self.temperature.clamp(min=0.1)

    def fit(self, val_loader, device, max_iter=50):
        self.to(device)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                all_logits.append(self.model(inputs.to(device)))
                all_labels.append(labels)

        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        def closure():
            optimizer.zero_grad()
            loss = criterion(all_logits / self.temperature.clamp(min=0.1),
                             all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        T_val = self.temperature.item()
        print(f"    Optimum T = {T_val:.4f}  "
              f"({'Soğutma (overconfidence düzeltildi)' if T_val > 1 else 'Isıtma (underconfidence)'})")
        return self


# =====================================================================
# 5. MODEL DEĞERLENDIRME VE YÜKLEME
# =====================================================================
@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for inputs, labels in loader:
        logits = model(inputs.to(device))
        probs  = torch.softmax(logits, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return compute_metrics(all_labels, all_probs)


def load_model(model_class, weights_path, device):
    if not os.path.exists(weights_path):
        print(f"    [UYARI] Ağırlık dosyası bulunamadı: {weights_path}")
        return None
    model = model_class(num_classes=NUM_CLASSES, input_channels=12)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model


def resolve_weights(template, fallback, pct):
    """Bütçeye özgü ağırlık varsa onu, yoksa fallback'i döndürür."""
    path = template.format(pct=pct)
    if os.path.exists(path):
        return path, False
    if os.path.exists(fallback):
        print(f"    [INFO] {pct}% ağırlık yok → fallback: {os.path.basename(fallback)}")
        return fallback, True   # True = fallback kullanıldı
    return None, True


# =====================================================================
# 6. ANA ABLASYON DÖNGÜSÜ
# =====================================================================
def run_ablation(device):
    # --- Veri setlerini yükle ---
    print("\n" + "="*60)
    print("Veri Setleri ve Etiket Dağılımları")
    print("="*60)

    ptbxl_test_df = pd.read_csv(os.path.join(BASE_DIR, "test_split.csv"))
    val_df        = pd.read_csv(os.path.join(BASE_DIR, "val_split.csv"))

    chapman_df = scan_external_dataset(
        os.path.join(DATA_DIR, "Chapman"), "Chapman")
    cpsc_df    = scan_external_dataset(
        os.path.join(DATA_DIR, "CPSC2018"), "CPSC 2018")

    # Dağılımları kaydet
    print_and_save_all_distributions(
        {"PTB-XL Test": ptbxl_test_df,
         "Chapman": chapman_df,
         "CPSC 2018": cpsc_df},
        BASE_DIR
    )

    val_loader        = DataLoader(Processed_ECG_Dataset(val_df),
                                   batch_size=BATCH_SIZE, shuffle=False)
    ptbxl_test_loader = DataLoader(Processed_ECG_Dataset(ptbxl_test_df),
                                   batch_size=BATCH_SIZE, shuffle=False)
    chapman_loader    = (DataLoader(Processed_ECG_Dataset(chapman_df),
                                   batch_size=BATCH_SIZE, shuffle=False)
                         if len(chapman_df) > 0 else None)
    cpsc_loader       = (DataLoader(Processed_ECG_Dataset(cpsc_df),
                                   batch_size=BATCH_SIZE, shuffle=False)
                         if len(cpsc_df) > 0 else None)

    test_sets = {
        "PTB-XL (In-Dist)"    : ptbxl_test_loader,
        "Chapman (Mild Shift)" : chapman_loader,
        "CPSC 2018 (Strong)"  : cpsc_loader,
    }

    all_results = []

    for pct in LABEL_BUDGETS:
        print(f"\n{'='*60}")
        print(f"LABEL BUDGET: {pct}%")
        print(f"{'='*60}")

        if SINGLE_WEIGHTS_MODE:
            baseline_path, b_fallback = FALLBACK_BASELINE, True
            dann_path,     d_fallback = FALLBACK_DANN,     True
        else:
            baseline_path, b_fallback = resolve_weights(
                BASELINE_WEIGHTS_TEMPLATE, FALLBACK_BASELINE, pct)
            dann_path,     d_fallback = resolve_weights(
                DANN_WEIGHTS_TEMPLATE,     FALLBACK_DANN,     pct)

        # -- Modelleri yükle --
        print(f"\n[A] Baseline ({pct}%)")
        baseline_model = load_model(ResNet34_1D, baseline_path, device) \
                         if baseline_path else None

        print(f"[B] DANN ({pct}%)")
        dann_model     = load_model(ResNet34_1D_DANN, dann_path, device) \
                         if dann_path else None

        print(f"[C] Baseline + TS ({pct}%)")
        baseline_ts = (TemperatureWrapper(
                          load_model(ResNet34_1D, baseline_path, device))
                       .fit(val_loader, device)
                       if baseline_path else None)

        print(f"[D] DANN + TS ({pct}%)")
        dann_ts     = (TemperatureWrapper(
                          load_model(ResNet34_1D_DANN, dann_path, device))
                       .fit(val_loader, device)
                       if dann_path else None)

        configs = {
            "A_Baseline"    : (baseline_model, b_fallback),
            "B_DANN"        : (dann_model,     d_fallback),
            "C_Baseline+TS" : (baseline_ts,    b_fallback),
            "D_DANN+TS"     : (dann_ts,        d_fallback),
        }

        for cfg_name, (model, used_fallback) in configs.items():
            for ds_name, loader in test_sets.items():
                if loader is None:
                    continue

                if model is None:
                    all_results.append({
                        'Config': cfg_name, 'Budget_pct': pct,
                        'Used_Fallback': used_fallback, 'Dataset': ds_name,
                        'Macro_F1': None, 'Macro_AUROC': None,
                        'Brier_Score': None, 'ECE': None,
                    })
                    continue

                metrics = evaluate_loader(model, loader, device)

                auroc_str = (f"{metrics['Macro_AUROC']:.4f}"
                             if not np.isnan(metrics['Macro_AUROC'])
                             else "  N/A ")
                print(f"  {cfg_name:18s} | {ds_name:25s} | "
                      f"AUROC={auroc_str}  F1={metrics['Macro_F1']:.4f}  "
                      f"ECE={metrics['ECE']:.4f}  "
                      f"Brier={metrics['Brier_Score']:.4f}")
                if metrics['AUROC_Note'] and 'Hariç' in metrics['AUROC_Note']:
                    print(f"    ↳ {metrics['AUROC_Note']}")

                all_results.append({
                    'Config'           : cfg_name,
                    'Budget_pct'       : pct,
                    'Used_Fallback'    : used_fallback,
                    'Dataset'          : ds_name,
                    'Macro_F1'         : metrics['Macro_F1'],
                    'Macro_AUROC'      : metrics['Macro_AUROC'],
                    'Brier_Score'      : metrics['Brier_Score'],
                    'ECE'              : metrics['ECE'],
                    'N_Classes_Present': metrics['N_Classes_Present'],
                    'AUROC_Note'       : metrics['AUROC_Note'],
                    'Per_Class_F1'     : json.dumps(metrics['Per_Class_F1']),
                })

    return all_results


# =====================================================================
# 7. TABLO VE EĞRİ OLUŞTURMA
# =====================================================================
def build_and_print_summary(results):
    df = pd.DataFrame(results)

    # Fallback uyarısı: eğer tüm satırlar fallback kullandıysa bütçe analizi geçersiz
    all_fallback = df.groupby('Budget_pct')['Used_Fallback'].all()
    if all_fallback.all():
        print("\n" + "!"*60)
        print("  UYARI: Tüm bütçe seviyeleri aynı fallback modelini kullandı.")
        print("  Label-budget eğrisi anlamsız — her bütçe için ayrı model")
        print("  eğitip kaydetmeniz gerekiyor:")
        print("    best_baseline_resnet34_10.pth")
        print("    best_baseline_resnet34_25.pth")
        print("    best_baseline_resnet34_50.pth")
        print("    best_dann_resnet34_10.pth")
        print("    best_dann_resnet34_25.pth")
        print("    best_dann_resnet34_50.pth")
        print("!"*60)

    # AUROC pivot (NaN değerleri bırak — 0.0 ile karıştırma)
    pivot = df.pivot_table(
        index   =['Budget_pct', 'Config'],
        columns ='Dataset',
        values  ='Macro_AUROC',
        aggfunc ='first'
    ).round(4)

    print("\n" + "="*70)
    print("  ABLASYON — MACRO AUROC (NaN = o veri setinde sınıf eksik)")
    print("="*70)
    print(pivot.to_string(na_rep='  N/A'))
    print("="*70)

    # Wide summary table
    wide = df.pivot_table(
        index   =['Config', 'Budget_pct'],
        columns ='Dataset',
        values  =['Macro_AUROC', 'Macro_F1', 'Brier_Score', 'ECE'],
        aggfunc ='first'
    ).round(4)

    # Label-budget curve (sadece non-fallback satırlar anlamlı)
    curve = df[['Config', 'Budget_pct', 'Dataset',
                'Macro_AUROC', 'Used_Fallback']].copy()
    curve = curve.sort_values(['Config', 'Dataset', 'Budget_pct'])

    return wide, curve


# =====================================================================
# ANA ÇALIŞTIRMA BLOĞU
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Donanım: {device}  |  Bütçeler: {LABEL_BUDGETS}")
    print(f"Konfigürasyonlar: A=Baseline  B=DANN  C=Base+TS  D=DANN+TS")

    results = run_ablation(device)

    wide_table, curve_df = build_and_print_summary(results)

    os.makedirs(BASE_DIR, exist_ok=True)

    json_path = os.path.join(BASE_DIR, "ablation_results_full.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    summary_path = os.path.join(BASE_DIR, "ablation_summary_table.csv")
    wide_table.to_csv(summary_path)

    curve_path = os.path.join(BASE_DIR, "label_budget_curve.csv")
    curve_df.to_csv(curve_path, index=False)

    print(f"\n✅ ablation_results_full.json  → {json_path}")
    print(f"✅ ablation_summary_table.csv  → {summary_path}")
    print(f"✅ label_budget_curve.csv      → {curve_path}")