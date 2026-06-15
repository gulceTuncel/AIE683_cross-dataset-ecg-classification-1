"""
ECG Preprocessing Pipeline
  1. Sinyal okuma + ADC → mV dönüşümü
  2. Baseline wander removal (4th-order zero-phase Butterworth, 0.5 Hz)
  3. Temporal normalization (center-crop / zero-pad → 10s)
  4. Z-score normalization (per-lead)
  5. Label harmonization (SCP → superclass, SNOMED-CT → superclass, multi-label)

Kullanım: python preprocess_ecg.py
"""

import os
import glob
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from collections import Counter
import ast
import warnings
import traceback

warnings.filterwarnings("ignore")


PATHS = {
    'PTB-XL': r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\PTB-XL\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
    'Chapman': r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Chapman\WFDB_ChapmanShaoxing",
    'CPSC2018': r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\CPSC2018"
}


OUTPUT_DIRS = {
    "PTB-XL":   r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\PTB-XL",
    "Chapman":  r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\Chapman",
    "CPSC2018": r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\CPSC2018",
}

EXCEL_MAPPING_PATH = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING\snomed_to_superclass.xlsx"
PTBXL_CSV_PATH = os.path.join(PATHS["PTB-XL"], "ptbxl_database.csv")

TARGET_FS = 500
TARGET_SECONDS = 10
TARGET_SAMPLES = TARGET_FS * TARGET_SECONDS


STATS = {"processed": 0, "skipped_unmapped": 0, "errors": 0,
         "cropped": 0, "padded": 0, "exact": 0}





SCP_TO_SUPERCLASS = {
    "NORM": "NORM",

    "AMI": "MI", "IMI": "MI", "LMI": "MI", "ALMI": "MI",
    "ASMI": "MI", "ILMI": "MI", "IPLMI": "MI", "IPMI": "MI",
    "PMI": "MI", "QWAVE": "MI",

    "NST_": "STTC", "DIG": "STTC", "NDT": "STTC",
    "ISCA": "STTC", "ISCI": "STTC", "ISC_": "STTC", "STTC": "STTC",

    "LBBB": "CD", "RBBB": "CD", "LAFB": "CD", "LPFB": "CD",
    "WPW": "CD", "IVCD": "CD", "CRBBB": "CD", "CLBBB": "CD",
    "1AVB": "CD", "2AVB": "CD", "3AVB": "CD", "AVB": "CD",
    "CHB3": "CD", "CHB2": "CD", "CHB": "CD", "BIVD": "CD",

    "LVH": "HYP", "RVH": "HYP", "SEHYP": "HYP",
    "LAO/LAE": "HYP", "RAO/RAE": "HYP",
}


def load_snomed_mapping(excel_path):
    mapping = {}
    df = pd.read_excel(excel_path, dtype=str)
    for _, row in df.iterrows():
        code = str(row["SNOMED_Code"]).strip()
        superclass = str(row["superclass"]).strip()

        code = code.replace(" ", "").split(".")[0]
        if code and superclass and superclass != "nan":
            mapping[code] = superclass
    print(f"Excel yüklendi: {len(mapping)} SNOMED kodu haritalandı.")
    return mapping


def load_ptbxl_labels(csv_path):
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    label_dict = {}
    for ecg_id, row in df.iterrows():
        label_dict[ecg_id] = list(row["scp_codes"].keys())
    return label_dict


def harmonize_ptbxl(scp_codes):
    """
    PTB-XL SCP kodlarını superclass'lara çevirir.
    Multi-label: birden fazla superclass döndürebilir.
    """
    superclasses = set()
    for code in scp_codes:
        if code in SCP_TO_SUPERCLASS:
            superclasses.add(SCP_TO_SUPERCLASS[code])
    return sorted(superclasses) if superclasses else None


def harmonize_snomed(snomed_codes, snomed_mapping):

    superclasses = set()
    for code in snomed_codes:
        code = code.strip()
        if code in snomed_mapping:
            superclasses.add(snomed_mapping[code])
    return sorted(superclasses) if superclasses else None


def extract_snomed_from_hea(hea_path):
    """Header dosyasından SNOMED-CT tanı kodlarını çıkarır."""
    codes = []
    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("#dx:") or line.lower().startswith("# dx:"):
                dx_part = line.split(":", 1)[1].strip()
                for c in dx_part.split(","):
                    c = c.strip()

                    digits = "".join(ch for ch in c if ch.isdigit())
                    if digits:
                        codes.append(digits)
                break
    return codes if codes else ["UNKNOWN"]



def read_ecg_signal(record_path):
    record = wfdb.rdrecord(record_path)
    ecg = record.p_signal
    fs = record.fs
    return ecg, fs



def remove_baseline_wander(ecg, fs):
    fc = 0.5
    b, a = butter(4, fc / (fs / 2), btype="high")
    filtered = np.zeros_like(ecg)
    for lead in range(ecg.shape[1]):
        filtered[:, lead] = filtfilt(b, a, ecg[:, lead])
    return filtered


def temporal_normalize(ecg, cur_fs, target_fs, target_samples):
    """
    1. Gerekirse resampling (cur_fs → target_fs)
    2. Center-crop (uzun kayıtlar) veya zero-pad (kısa kayıtlar) → target_samples

    Returns:
        out: (target_samples, num_leads) array
        action: "cropped" / "padded" / "exact"
    """

    if cur_fs != target_fs:
        from scipy.signal import resample
        num_target = int(ecg.shape[0] * target_fs / cur_fs)
        ecg = resample(ecg, num_target, axis=0)

    cur_samples = ecg.shape[0]
    num_leads = ecg.shape[1]
    out = np.zeros((target_samples, num_leads))

    if cur_samples > target_samples:

        start = (cur_samples - target_samples) // 2
        out = ecg[start : start + target_samples, :]
        action = "cropped"
    elif cur_samples < target_samples:

        pad_start = (target_samples - cur_samples) // 2
        out[pad_start : pad_start + cur_samples, :] = ecg
        action = "padded"
    else:
        out = ecg.copy()
        action = "exact"

    return out, action


def zscore_normalize_per_lead(ecg):
    """
    Per-lead Z-score normalization.
    Her lead için: x̂ = (x - μ) / σ

    Sabit sinyallerde (σ=0) bölme hatasını önlemek için küçük epsilon eklenir.
    """
    eps = 1e-8
    normalized = np.zeros_like(ecg)
    for lead in range(ecg.shape[1]):
        mu = np.mean(ecg[:, lead])
        sigma = np.std(ecg[:, lead])
        normalized[:, lead] = (ecg[:, lead] - mu) / (sigma + eps)
    return normalized



def preprocess_single_record(record_path, cur_fs=None):
    """
    Tek bir ECG kaydı için tüm preprocessing adımlarını uygular:
      1. Sinyal okuma (ADC → mV)
      2. Baseline wander removal
      3. Temporal normalization (resample + crop/pad)
      4. Z-score normalization (per-lead)

    Returns:
        processed_ecg: (5000, 12) numpy array
        fs: orijinal sampling rate
        action: "cropped" / "padded" / "exact"
    """

    ecg, fs = read_ecg_signal(record_path)


    ecg = remove_baseline_wander(ecg, fs)


    ecg, action = temporal_normalize(ecg, fs, TARGET_FS, TARGET_SAMPLES)


    ecg = zscore_normalize_per_lead(ecg)

    return ecg, fs, action


def process_ptbxl(base_path, output_dir, ptbxl_labels):
    print("\n" + "=" * 70)
    print("PTB-XL İŞLENİYOR")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)


    records_dir = os.path.join(base_path, "records500")
    hea_files = glob.glob(os.path.join(records_dir, "**", "*.hea"), recursive=True)
    print(f"Toplam kayıt: {len(hea_files)}")

    for i, hea_file in enumerate(hea_files):
        record_name = os.path.splitext(os.path.basename(hea_file))[0]
        record_path = os.path.join(os.path.dirname(hea_file), record_name)

        try:

            ecg_id = int("".join(c for c in record_name if c.isdigit()))


            if ecg_id not in ptbxl_labels:
                STATS["skipped_unmapped"] += 1
                continue

            scp_codes = ptbxl_labels[ecg_id]
            superclasses = harmonize_ptbxl(scp_codes)

            if superclasses is None:
                STATS["skipped_unmapped"] += 1
                continue


            ecg, fs, action = preprocess_single_record(record_path)
            STATS[action] += 1

            out_path = os.path.join(output_dir, f"{record_name}.npz")
            np.savez_compressed(
                out_path,
                signal=ecg,
                original_labels=scp_codes,
                superclasses=superclasses,
                original_fs=fs,
            )
            STATS["processed"] += 1

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(hea_files)} tamamlandı...")

        except Exception as e:
            STATS["errors"] += 1
            print(f"  HATA [{record_name}]: {e}")


def process_chapman_cpsc(base_path, output_dir, ds_name, snomed_mapping):
    """Chapman veya CPSC2018 veri setini işler."""
    print("\n" + "=" * 70)
    print(f"{ds_name} İŞLENİYOR")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    hea_files = glob.glob(os.path.join(base_path, "**", "*.hea"), recursive=True)
    print(f"Toplam kayıt: {len(hea_files)}")

    for i, hea_file in enumerate(hea_files):
        record_name = os.path.splitext(os.path.basename(hea_file))[0]
        record_path = os.path.join(os.path.dirname(hea_file), record_name)

        try:

            snomed_codes = extract_snomed_from_hea(hea_file)


            superclasses = harmonize_snomed(snomed_codes, snomed_mapping)

            if superclasses is None:
                STATS["skipped_unmapped"] += 1
                continue


            ecg, fs, action = preprocess_single_record(record_path)
            STATS[action] += 1


            out_path = os.path.join(output_dir, f"{record_name}.npz")
            np.savez_compressed(
                out_path,
                signal=ecg,
                original_labels=snomed_codes,
                superclasses=superclasses,
                original_fs=fs,
            )
            STATS["processed"] += 1

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(hea_files)} tamamlandı...")

        except Exception as e:
            STATS["errors"] += 1
            print(f"  HATA [{record_name}]: {e}")



if __name__ == "__main__":
    print("=" * 70)
    print("ECG PREPROCESSING PIPELINE")
    print("=" * 70)


    for name, path in PATHS.items():
        status = "BULUNDU" if os.path.exists(path) else "BULUNAMADI!"
        print(f"  {name}: {status} --> {path}")


    print("\nSözlükler yükleniyor...")
    ptbxl_labels = load_ptbxl_labels(PTBXL_CSV_PATH)
    snomed_mapping = load_snomed_mapping(EXCEL_MAPPING_PATH)


    STATS = {"processed": 0, "skipped_unmapped": 0, "errors": 0,
             "cropped": 0, "padded": 0, "exact": 0}
    process_ptbxl(PATHS["PTB-XL"], OUTPUT_DIRS["PTB-XL"], ptbxl_labels)
    print(f"\nPTB-XL Özet: {STATS}")


    STATS = {"processed": 0, "skipped_unmapped": 0, "errors": 0,
             "cropped": 0, "padded": 0, "exact": 0}
    process_chapman_cpsc(PATHS["Chapman"], OUTPUT_DIRS["Chapman"], "Chapman", snomed_mapping)
    print(f"\nChapman Özet: {STATS}")


    STATS = {"processed": 0, "skipped_unmapped": 0, "errors": 0,
             "cropped": 0, "padded": 0, "exact": 0}
    process_chapman_cpsc(PATHS["CPSC2018"], OUTPUT_DIRS["CPSC2018"], "CPSC2018", snomed_mapping)
    print(f"\nCPSC2018 Özet: {STATS}")

    print("\n" + "=" * 70)
    print("İşlem Tamamlandı!")
    print("=" * 70)
