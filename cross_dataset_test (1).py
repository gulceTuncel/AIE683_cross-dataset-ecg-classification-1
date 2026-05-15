import torch
import numpy as np
import pandas as pd
import os
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

# =====================================================================
# 1. GEREKLİ KÜTÜPHANELER VE SINIFLAR
# =====================================================================
# Kendi yazdığımız 1D ResNet mimarisini içe aktarıyoruz.
from resNet34 import ResNet34_1D
from resnet34_dann import ResNet34_1D_DANN

class Processed_ECG_Dataset(Dataset):
    """
    PyTorch'un verileri modelin içine (GPU'ya) taşımasını sağlayan "Kurye" sınıfıdır.
    Test verilerini de tıpkı eğitim verilerinde olduğu gibi tensörlere dönüştürür.
    """
    def __init__(self, metadata_df):
        self.data_df = metadata_df.reset_index(drop=True)
        # Sınıfları rakamlara dönüştürme sözlüğü (Chapman ve CPSC'nin etiketlerini
        # projenizdeki "Label Harmonization" adımıyla bu 5 sınıfa uydurmuştuk).
        self.label_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        file_path = row['file_path']
        label_text = row['super_class']
        
        # Önceden işlenmiş (Z-score ve Crop/Pad uygulanmış) EKG sinyalini disken okur
        with np.load(file_path, allow_pickle=True) as data:
            signal = data['signal']
        
        # Sinyal boyutunu (Zaman, Kanal) -> (Kanal, Zaman) olarak çevirir (PyTorch Conv1d formatı)
        signal = np.transpose(signal, (1, 0)) 
        label_idx = self.label_map.get(label_text, -1)

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)

# =====================================================================
# 2. DEĞERLENDİRME METRİKLERİ (Distribution Shift Analizi İçin)
# =====================================================================
def multiclass_brier_score(y_true, y_prob):
    """
    Modelin olasılık tahminlerinin isabetliliğini ölçer.
    Model PTB-XL'den çıkıp Çin verisine geçtiğinde "yanlış kararlara" çok yüksek
    olasılık (özgüven) vermeye başlarsa, Brier skoru hızla kötüleşir (yükselir).
    """
    num_classes = y_prob.shape[1]
    y_true_onehot = np.eye(num_classes)[y_true]
    return np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Modelin kalibrasyonunu (özgüven vs. gerçeklik) ölçer.
    Beklentimiz: Model PTB-XL'de düşük ECE veriyordu. CPSC veri setine geçince 
    "Aşırı Özgüven" (Overconfidence) problemi yaşayacak ve ECE fırlayacak.
    İşte bu metrik, o fırlamayı sayısal olarak ispatlamamızı sağlar.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def evaluate_all_metrics(y_true, y_prob, dataset_name):
    """
    Tüm test verisi bittikten sonra sonuçları derleyip konsola raporlayan fonksiyon.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    try:
        macro_auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        macro_auroc = 0.0
        
    brier = multiclass_brier_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    
    print("\n" + "="*50)
    print(f"      {dataset_name} TEST SONUÇLARI")
    print("="*50)
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Macro AUROC      : {macro_auroc:.4f}")
    print(f"Brier Score      : {brier:.4f}")
    print(f"ECE (10 Bins)    : {ece:.4f}")
    print("="*50 + "\n")
    
    return {'Dataset': dataset_name, 'Macro_F1': macro_f1, 'Macro_AUROC': macro_auroc, 'Brier_Score': brier, 'ECE': ece}

# =====================================================================
# 3. YENİ VERİ SETLERİNİ TARAMA FONKSİYONU
# =====================================================================
import glob
def scan_external_dataset(processed_dir):
    """
    PTB-XL dışındaki (Chapman ve CPSC) veri setlerini tarayıp içindeki dosyaları listeler.
    Bu veri setlerinde bazı sinyallerin etiketleri sistemimize uymayabilir (UNKNOWN),
    bu fonksiyon o geçersiz kayıtları testten hariç tutar.
    """
    print(f"\n{processed_dir} taranıyor...")
    npz_files = glob.glob(os.path.join(processed_dir, "*.npz"))
    
    data_list = []
    skipped_count = 0
    
    for file_path in tqdm(npz_files, desc="Dosyalar Haritalanıyor"):
        with np.load(file_path, allow_pickle=True) as data:
            superclasses = data['superclasses']
            
            # Eğer dosyanın geçerli bir etiketi (NORM, MI vb.) varsa listeye ekle
            if len(superclasses) > 0 and superclasses[0] != 'UNKNOWN':
                primary_class = superclasses[0]
                data_list.append({
                    'super_class': primary_class,
                    'file_path': file_path
                })
            else:
                 skipped_count += 1 # Etiketi anlaşılamayanları atla (Label Harmonization hatası vs.)
                
    df = pd.DataFrame(data_list)
    print(f"Bulunan Geçerli Kayıt: {len(df)} | Atlanan Kayıt: {skipped_count}")
    return df

# =====================================================================
# 4. TEST FONKSİYONU
# =====================================================================
def run_cross_dataset_test(model, data_loader, dataset_name, device):
    """
    Dışarıdan gelen veri setlerini modelin içine sokup tahminleri (olasılıkları) toplar.
    """
    model.eval() # Modeli dondur, test moduna al (BatchNorm vb. kilitlenir)
    all_labels = []
    all_probs = []
    
    with torch.no_grad(): # Hafıza tasarrufu: Türev almayı kapat (Eğitim yapmıyoruz)
        test_bar = tqdm(data_loader, desc=f"Test Ediliyor ({dataset_name})")
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1) # Logitleri yüzdelik olasılığa çevir
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Tüm tahminler bitince o veri seti için metrikleri hesapla
    metrics = evaluate_all_metrics(all_labels, all_probs, dataset_name)
    return metrics

# =====================================================================
# ANA ÇALIŞTIRMA BLOK (ANA PİPELİNE)
# =====================================================================
if __name__ == "__main__":
    # --- AYARLAR ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. En iyi model ağırlıklarınızı yükleyin (PTB-XL'de eğitilen Altın Model)
    weights_path = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING\best_dann_resnet34.pth" 
    model = ResNet34_1D_DANN(num_classes=5, input_channels=12)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    print(f"Model {weights_path} konumundan başarıyla yüklendi.")

    # 2. Veri Seti Klasör Yolları
    CHAPMAN_PROCESSED_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\Chapman"
    CPSC_PROCESSED_DIR = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\CPSC2018"
    
    all_results = []

    # --- TEST 1: CHAPMAN (MILD SHIFT - HAFİF KAYMA) ---
    # Chapman veri seti, PTB-XL ile benzer cihazlarla kaydedilmiştir ancak Çinli hastaları içerir.
    # Burada sadece "Popülasyon" kaynaklı ufak bir performans düşüşü bekliyoruz.
    if os.path.exists(CHAPMAN_PROCESSED_DIR):
        chapman_df = scan_external_dataset(CHAPMAN_PROCESSED_DIR)
        if len(chapman_df) > 0:
            chapman_loader = DataLoader(Processed_ECG_Dataset(chapman_df), batch_size=32, shuffle=False)
            chapman_metrics = run_cross_dataset_test(model, chapman_loader, "Chapman (Mild Shift)", device)
            all_results.append(chapman_metrics)
    else:
        print(f"HATA: {CHAPMAN_PROCESSED_DIR} bulunamadı.")

    # --- TEST 2: CPSC 2018 (STRONG SHIFT - GÜÇLÜ KAYMA) ---
    # CPSC 2018 hem Çin popülasyonunu içerir, hem de 11 farklı hastanedeki farklı cihazlardan gelir.
    # Boyutları değişkendir. Proje hipotezine göre asıl "çöküşü" burada göreceğiz.
    if os.path.exists(CPSC_PROCESSED_DIR):
        cpsc_df = scan_external_dataset(CPSC_PROCESSED_DIR)
        if len(cpsc_df) > 0:
            cpsc_loader = DataLoader(Processed_ECG_Dataset(cpsc_df), batch_size=32, shuffle=False)
            cpsc_metrics = run_cross_dataset_test(model, cpsc_loader, "CPSC 2018 (Strong Shift)", device)
            all_results.append(cpsc_metrics)
    else:
        print(f"HATA: {CPSC_PROCESSED_DIR} bulunamadı.")

    # --- SONUÇLARI KAYDETME ---
    # Tüm testler bittikten sonra sonuçları makaleniz için rapor olarak (JSON) kaydeder.
    if all_results:
        save_dir = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"
        results_file = os.path.join(save_dir, "cross_dataset_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
             json.dump(all_results, f, indent=4)
        print(f"\nÇapraz Veri Seti (Cross-Dataset) test sonuçları kaydedildi: {results_file}")