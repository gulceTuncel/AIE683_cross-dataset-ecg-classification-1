"""
data_split.py  —  Patient-Based Stratified Split + Label Budget Subsampling

Key design decisions:
  - Patient-based split prevents data leakage (same patient never straddles sets)
  - Stratified by super_class at every level to preserve class distribution
  - Four label budget levels: 10%, 25%, 50%, 100% (proposal §Objectives)
  - All splits are saved to CSV for full reproducibility
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split




def build_metadata_from_npz(processed_dir, original_csv_path):
    """
    Tüm EKG sinyallerini RAM'e yüklemek yerine sadece dosya yollarını ve
    etiketlerini içeren bir harita (metadata) çıkarır.
    """
    print("İşlenmiş NPZ dosyaları taranıyor...")

    npz_files = glob.glob(os.path.join(processed_dir, "*.npz"))
    ptbxl_csv  = pd.read_csv(original_csv_path, index_col='ecg_id')

    data_list = []

    for file_path in npz_files:
        filename   = os.path.basename(file_path)
        ecg_id_str = "".join(c for c in filename.split('_')[0] if c.isdigit())

        if not ecg_id_str:
            continue

        ecg_id = int(ecg_id_str)

        if ecg_id not in ptbxl_csv.index:
            continue

        patient_id = ptbxl_csv.loc[ecg_id, 'patient_id']

        with np.load(file_path, allow_pickle=True) as data:
            superclasses = data['superclasses']

            if len(superclasses) > 0:
                primary_class = superclasses[0]
                data_list.append({
                    'ecg_id':     ecg_id,
                    'patient_id': patient_id,
                    'super_class': primary_class,
                    'file_path':  file_path
                })

    metadata_df = pd.DataFrame(data_list)
    print(f"Toplam {len(metadata_df)} geçerli dosya bulundu ve eşleştirildi.")
    return metadata_df





def patient_based_stratified_split(df):
    """
    Veri Sızıntısını (Data Leakage) önleyen hasta bazlı bölme.
    Split oranları  →  Train 70% | Val 10% | Test 20%
    """
    print("\nHasta bazlı veri bölme işlemi başlatılıyor...")

    patient_df = df.drop_duplicates(subset=['patient_id']).copy()


    train_val_patients, test_patients = train_test_split(
        patient_df, test_size=0.20, random_state=42,
        stratify=patient_df['super_class']
    )


    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.125, random_state=42,
        stratify=train_val_patients['super_class']
    )

    train_df = df[df['patient_id'].isin(train_patients['patient_id'])]
    val_df   = df[df['patient_id'].isin(val_patients['patient_id'])]
    test_df  = df[df['patient_id'].isin(test_patients['patient_id'])]

    print(f"Eğitim Seti (Train 100%): {len(train_df)} dosya")
    print(f"Doğrulama Seti (Val)    : {len(val_df)} dosya")
    print(f"Test Seti (Test)        : {len(test_df)} dosya")

    return train_df, val_df, test_df





def create_label_budget_splits(train_df, budgets=(0.10, 0.25, 0.50), random_state=42):
    """
    Tüm eğitim setinden belirtilen oranlarda tabakalı alt küme üretir.

    Args:
        train_df     : Tam eğitim seti (100% budget)
        budgets      : Üretilecek bütçe oranları tuple'ı  (proposal: 10%, 25%, 50%)
        random_state : Tekrarlanabilirlik için sabit seed

    Returns:
        dict  →  { '10': DataFrame, '25': DataFrame, '50': DataFrame }
    """
    budget_splits = {}

    for ratio in budgets:
        pct_label = int(ratio * 100)

        subset, _ = train_test_split(
            train_df,
            train_size=ratio,
            random_state=random_state,
            stratify=train_df['super_class']
        )

        budget_splits[str(pct_label)] = subset
        print(f"Yüzde {pct_label:3d}% Eğitim Bütçesi: {len(subset):>6} dosya "
              f"({len(subset)/len(train_df)*100:.1f}% of full train set)")

    return budget_splits





class Processed_ECG_Dataset(Dataset):
    """
    'Lazy Loading' mantığıyla çalışır: ihtiyaç duyulan batch kadar
    dosyayı diskten okur.
    """
    def __init__(self, metadata_df):
        self.data_df  = metadata_df.reset_index(drop=True)
        self.label_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row        = self.data_df.iloc[idx]
        file_path  = row['file_path']
        label_text = row['super_class']

        with np.load(file_path, allow_pickle=True) as data:
            signal = data['signal']

        signal    = np.transpose(signal, (1, 0))
        label_idx = self.label_map.get(label_text, -1)

        return (
            torch.tensor(signal,    dtype=torch.float32),
            torch.tensor(label_idx, dtype=torch.long)
        )





def print_class_distribution(df, split_name):
    """Her split için sınıf dağılımını yazdırır."""
    counts = df['super_class'].value_counts()
    total  = len(df)
    print(f"\n  {split_name} sınıf dağılımı:")
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls:5s}: {cnt:5d}  ({cnt/total*100:.1f}%)")





if __name__ == "__main__":

    PROCESSED_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\PTB-XL"
    ORIGINAL_CSV  = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\PTB-XL\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1\ptbxl_database.csv"
    SAVE_DIR      = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"




    meta_df = build_metadata_from_npz(PROCESSED_DIR, ORIGINAL_CSV)




    train_df, val_df, test_df = patient_based_stratified_split(meta_df)




    print("\nLabel budget alt kümeleri oluşturuluyor...")
    budget_splits = create_label_budget_splits(
        train_df, budgets=(0.10, 0.25, 0.50)
    )




    print_class_distribution(train_df,              "Train 100%")
    print_class_distribution(budget_splits['50'],   "Train  50%")
    print_class_distribution(budget_splits['25'],   "Train  25%")
    print_class_distribution(budget_splits['10'],   "Train  10%")
    print_class_distribution(val_df,                "Val        ")
    print_class_distribution(test_df,               "Test       ")




    print("\nDataLoader sanity check (10% budget)...")
    loader = DataLoader(Processed_ECG_Dataset(budget_splits['10']),
                        batch_size=32, shuffle=True)
    signals, labels = next(iter(loader))
    print(f"  Sinyal boyutu (beklenen [32, 12, 5000]): {signals.shape}")
    print(f"  Etiket boyutu (beklenen [32])          : {labels.shape}")




    print("\nCSV dosyaları kaydediliyor...")
    os.makedirs(SAVE_DIR, exist_ok=True)


    train_df.to_csv(os.path.join(SAVE_DIR, "train_split_100.csv"), index=False)
    val_df  .to_csv(os.path.join(SAVE_DIR, "val_split.csv"),       index=False)
    test_df .to_csv(os.path.join(SAVE_DIR, "test_split.csv"),      index=False)


    for pct, df_budget in budget_splits.items():
        out_path = os.path.join(SAVE_DIR, f"train_split_{pct}.csv")
        df_budget.to_csv(out_path, index=False)
        print(f"  Kaydedildi: {out_path}")

    print("\n✅ Tüm veri bölme listeleri (10%, 25%, 50%, 100%) başarıyla kaydedildi!")
    print(f"   Kayıt konumu: {SAVE_DIR}")
