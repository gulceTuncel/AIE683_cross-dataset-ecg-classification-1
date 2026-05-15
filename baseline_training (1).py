import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from resNet34 import ResNet34_1D, BasicBlock1D
import os # Dosya yolları işlemleri için
import json # Sonuçları kaydetmek için

# =====================================================================
# 1. ÖZEL DEĞERLENDİRME METRİKLERİ (Test Aşamasında Kullanılacak)
# =====================================================================
def multiclass_brier_score(y_true, y_prob):
    """
    Brier Score: Modelin tahmin olasılıklarının gerçeklikle ne kadar örtüştüğünü (Calibration) ölçer.
    Düşük olması modelin kendine güveninde ne kadar "haklı" olduğunu gösterir.
    """
    num_classes = y_prob.shape[1]
    # Gerçek etiketleri (örn: 2) One-Hot vektörüne dönüştürür (örn: [0, 0, 1, 0, 0])
    y_true_onehot = np.eye(num_classes)[y_true]
    # Tahmin olasılıkları ile gerçek vektör arasındaki karesel hatanın ortalamasını alır
    brier_score = np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
    return brier_score

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    ECE (Expected Calibration Error): Model "Bu %90 ihtimalle Kalp Krizi (MI)" dediğinde, 
    gerçekten vakaların %90'ında MI çıkıp çıkmadığını kontrol eder.
    Distribution Shift (Alan kayması) durumlarında en çok bozulan metrik budur.
    """
    confidences = np.max(y_prob, axis=1) # Modelin verdiği en yüksek olasılık (Güven)
    predictions = np.argmax(y_prob, axis=1) # Modelin seçtiği sınıf
    accuracies = (predictions == y_true) # Tahmin doğru mu yanlış mı? (True/False)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1) # %0'dan %100'e 10 adet kutu (bin) oluşturur
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Bu güven aralığına (örn: %80-%90) düşen tahminleri bul
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin]) # Bu kutudaki gerçek doğruluk
            avg_confidence_in_bin = np.mean(confidences[in_bin]) # Bu kutudaki ortalama özgüven
            # Farkın mutlak değerini ağırlığıyla çarpıp ana ECE skoruna ekle
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def evaluate_all_metrics(y_true, y_prob):
    """
    Tüm tahminleri alıp, projenizin metodolojisinde belirttiğiniz 4 ana metriği hesaplar.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Sınıf dengesizliğine karşı dirençli olan Macro F1 skoru
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Modelin sınıf ayırma gücünü eşik değerinden (threshold) bağımsız ölçen Macro AUROC
    try:
        macro_auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        # Eğitim başında model sadece 1 sınıfı tahmin ederse çökmemesi için güvenlik önlemi
        macro_auroc = 0.0
        
    brier = multiclass_brier_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    
    # Konsol ekranına şık bir rapor yazdır
    print("\n" + "="*40)
    print("      TEST SETİ SONUÇ RAPORU")
    print("="*40)
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Macro AUROC      : {macro_auroc:.4f}")
    print(f"Brier Score      : {brier:.4f}  (Düşük olması iyidir)")
    print(f"ECE (10 Bins)    : {ece:.4f}  (Düşük olması iyidir)")
    print("="*40 + "\n")
    
    return {'Macro_F1': macro_f1, 'Macro_AUROC': macro_auroc, 'Brier_Score': brier, 'ECE': ece}

# =====================================================================
# 2. SINIF AĞIRLIKLARI HESAPLAMA (Eğitim Aşamasında Kullanılacak)
# =====================================================================
def get_class_weights(train_df):
    """
    Azınlık sınıflarına (nadir hastalıklara) daha yüksek ceza (ağırlık) vererek,
    modelin sadece çoğunluk sınıfını (NORM) ezberlemesini engeller (Class Imbalance çözümü).
    """
    labels = train_df['super_class'].values
    unique_labels = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=labels)
    
    label_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
    weight_tensor = torch.zeros(len(label_map))
    
    for cls_name, idx in label_map.items():
        if cls_name in unique_labels:
            cls_idx_in_unique = np.where(unique_labels == cls_name)[0][0]
            weight_tensor[idx] = weights[cls_idx_in_unique]
        else:
            weight_tensor[idx] = 1.0 # Eğer alt-kümelerde o sınıf hiç yoksa nötr etki yapar
    return weight_tensor

# =====================================================================
# 3. ANA EĞİTİM DÖNGÜSÜ (Training & Validation)
# =====================================================================
def train_baseline_model(model, train_loader, val_loader, class_weights, num_epochs=20, save_path='best_baseline.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim başlatılıyor. Donanım: {device}")
    
    model.to(device)
    class_weights = class_weights.to(device)
    
    # Ağırlıklı CrossEntropyLoss kullanımı
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # L2 regülarizasyonu (weight_decay) içeren AdamW optimizasyonu (Ezberlemeyi zorlaştırır)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_auroc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # ----------------- EĞİTİM AŞAMASI -----------------
        model.train() # BatchNorm ve Dropout gibi katmanları aktif eder
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc="Eğitim", leave=False)
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() # Eski batch'ten kalan türevleri (gradyanları) temizle
            outputs = model(inputs) # İleri besleme (Forward)
            loss = criterion(outputs, labels) # Hatayı hesapla
            loss.backward() # Hatayı geriye yay (Backward)
            optimizer.step() # Model ağırlıklarını güncelle
            
            train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        
        # ----------------- DOĞRULAMA (VALIDATION) AŞAMASI -----------------
        model.eval() # BatchNorm gibi katmanları dondurur, eğitim yapmaz
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad(): # Hafıza tasarrufu için türev hesaplamayı tamamen kapat
            val_bar = tqdm(val_loader, desc="Doğrulama", leave=False)
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1) # Çıktıları olasılığa (0-1 arasına) dönüştür
                preds = torch.argmax(probs, dim=1) # En yüksek olasılıklı sınıfı seç
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        try:
            val_auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            val_auroc = 0.0 
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Macro AUROC: {val_auroc:.4f}")
        
        # CHECKPOINT (En iyi modeli kaydetme)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), save_path)
            print(f"--> Yeni en iyi model kaydedildi! (AUROC: {best_val_auroc:.4f})")

# =====================================================================
# 4. TEST VE DEĞERLENDİRME FONKSİYONU (Eğitim Sonrası Çalışır)
# =====================================================================
def test_and_evaluate_model(model, test_loader, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Kaydettiğimiz "En İyi" ağırlıkları bilgisayardan okuyup modele yüklüyoruz
    print(f"\n[{weights_path}] ağırlıkları yükleniyor ve Test Setinde değerlendiriliyor...")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval() # Test için modeli değerlendirme moduna al
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Test Ediliyor")
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Toplanan sonuçları 4 metrikli ana değerlendirme fonksiyonuna gönderiyoruz
    metrics = evaluate_all_metrics(all_labels, all_probs)
    return metrics

# =====================================================================
# ÇALIŞTIRMA AKIŞI (Ana Pipeline)
# =====================================================================
if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader

    # ---------------------------------------------------------
    # 1. VERİ SETİ SINIFI (PyTorch'un veriyi okuma formatı)
    # ---------------------------------------------------------
    class Processed_ECG_Dataset(Dataset):
        def __init__(self, metadata_df):
            self.data_df = metadata_df.reset_index(drop=True)
            self.label_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}

        def __len__(self):
            return len(self.data_df)

        def __getitem__(self, idx):
            row = self.data_df.iloc[idx]
            file_path = row['file_path']
            label_text = row['super_class']
            
            # NPZ'den sinyali çek
            with np.load(file_path, allow_pickle=True) as data:
                signal = data['signal']
            
            # PyTorch Conv1d formatı için eksenleri değiştir (Length, Channels) -> (Channels, Length)
            signal = np.transpose(signal, (1, 0)) 
            label_idx = self.label_map.get(label_text, -1)

            return torch.tensor(signal, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)

    # ---------------------------------------------------------
    # 2. KAYDEDİLEN CSV DOSYALARINI OKUMA
    # ---------------------------------------------------------
    csv_dir = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"
    
    print("Veri listeleri yükleniyor...")
    train_df = pd.read_csv(os.path.join(csv_dir, "train_split_100.csv"))
    val_df = pd.read_csv(os.path.join(csv_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(csv_dir, "test_split.csv"))
    
    # ---------------------------------------------------------
    # 3. DATALOADER'LARI OLUŞTURMA
    # ---------------------------------------------------------
    print("DataLoader'lar hazırlanıyor...")
    train_dataset = Processed_ECG_Dataset(train_df)
    val_dataset = Processed_ECG_Dataset(val_df)
    test_dataset = Processed_ECG_Dataset(test_df)
    
    # Sadece eğitim seti karıştırılır (Modelin ezberlememesi için)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ---------------------------------------------------------
    # 4. MODELİ BAŞLATMA VE EĞİTİM
    # ---------------------------------------------------------
    weights = get_class_weights(train_df)
    
    # ResNet34_1D modelini başlat
    model = ResNet34_1D(num_classes=5, input_channels=12)
    save_path = os.path.join(csv_dir, 'best_baseline_resnet34.pth')
    
    # Eğitimi Başlat (20 epoch)
    train_baseline_model(model, train_loader, val_loader, weights, num_epochs=20, save_path=save_path)
    
    # ---------------------------------------------------------
    # 5. TEST ETME VE SONUÇLARI KAYDETME (EKLENEN KISIM)
    # ---------------------------------------------------------
    # Eğitim bitince, "en iyi" modeli test setiyle değerlendirir
    final_metrics = test_and_evaluate_model(model, test_loader, weights_path=save_path)
    
    # Sonuçların JSON formatında bilgisayara kaydedilmesi
    metrics_save_path = os.path.join(csv_dir, "baseline_test_metrics.json")
    
    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"✅ Tüm metrikler kalıcı olarak kaydedildi: {metrics_save_path}")