import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

# Daha önce yazdığımız DANN modelini çağırıyoruz
from resnet34_dann import ResNet34_1D_DANN

# =====================================================================
# 1. VERİ SETİ SINIFI (PyTorch Dataset)
# =====================================================================
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
        
        with np.load(file_path, allow_pickle=True) as data:
            signal = data['signal']
            
        signal = np.transpose(signal, (1, 0))
        label_idx = self.label_map.get(label_text, -1)
        
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)

# =====================================================================
# 2. ALPHA (α) GÜNCELLEYİCİ - DANN'ın Matematiksel Sırrı
# =====================================================================
def get_alpha(current_step, total_steps):
    """
    Eğitimin en başında GRL (Gradient Reversal Layer) kapalıya yakın başlar (alpha ≈ 0).
    Çünkü model önce bir EKG'nin neye benzediğini öğrenmelidir.
    Eğitim ilerledikçe alpha değeri 1'e doğru çıkar ve model 
    "Alan (Domain) Unutma" konusunda giderek daha agresifleşir.
    Formül: Yarının-Maksimumu (Half-Max) tarzı bir S-Eğrisi (Sigmoid) kullanır.
    """
    p = float(current_step) / total_steps
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    return alpha

# =====================================================================
# 3. DANN EĞİTİM DÖNGÜSÜ
# =====================================================================
def train_dann_model(model, source_loader, target_loader, val_loader, num_epochs=20, save_path='best_dann_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\nDANN Eğitimi Başlatılıyor... Donanım: {device}")
    
    # İki farklı Loss (Kayıp) Fonksiyonu
    criterion_class = nn.CrossEntropyLoss() # Hastalıkları ayırmak için (5 sınıf)
    criterion_domain = nn.CrossEntropyLoss() # Kaynak vs Hedef ayrımı için (2 sınıf: 0 veya 1)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_val_auroc = 0.0
    
    # Toplam adım sayısını hesapla (Alpha formülü için)
    # İki veri setinden hangisi KÜÇÜKSE ona göre tur atılır (Denge bozulmasın diye)
    len_dataloader = min(len(source_loader), len(target_loader))
    total_steps = num_epochs * len_dataloader

    for epoch in range(num_epochs):
        model.train()
        total_class_loss = 0.0
        total_domain_loss = 0.0
        
        # İki dataloader'ı aynı anda iterasyona sok (zip)
        # Biri Almanya (Source), diğeri Çin (Target)
        data_zip = zip(source_loader, target_loader)
        batch_bar = tqdm(data_zip, total=len_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Eğitim]", leave=False)
        
        for i, ((source_inputs, source_labels), (target_inputs, _)) in enumerate(batch_bar):
            # Adım hesaplama (Alpha'yı güncelle)
            current_step = epoch * len_dataloader + i
            alpha = get_alpha(current_step, total_steps)
            
            source_inputs, source_labels = source_inputs.to(device), source_labels.to(device)
            target_inputs = target_inputs.to(device)
            
            optimizer.zero_grad()
            
            # -------------------------------------------------------------
            # ADIM 1: SOURCE (KAYNAK - Almanya) VERİSİ İLE İŞLEM
            # -------------------------------------------------------------
            # PTB-XL verilerini modele sokuyoruz. Hem hastalık hem alan tahmini alıyoruz.
            class_preds_s, domain_preds_s = model(source_inputs, alpha=alpha)
            
            # Hastalık Hatası
            loss_s_class = criterion_class(class_preds_s, source_labels)
            
            # Alan Hatası (Almanya için etiketler SIFIR (0) olarak belirlenir)
            domain_labels_s = torch.zeros(source_inputs.size(0), dtype=torch.long).to(device)
            loss_s_domain = criterion_domain(domain_preds_s, domain_labels_s)
            
            # -------------------------------------------------------------
            # ADIM 2: TARGET (HEDEF - Çin) VERİSİ İLE İŞLEM
            # -------------------------------------------------------------
            # Chapman verilerini modele sokuyoruz. HASTALIKLARI GİZLİYORUZ (Unsupervised).
            _, domain_preds_t = model(target_inputs, alpha=alpha)
            
            # Alan Hatası (Çin için etiketler BİR (1) olarak belirlenir)
            domain_labels_t = torch.ones(target_inputs.size(0), dtype=torch.long).to(device)
            loss_t_domain = criterion_domain(domain_preds_t, domain_labels_t)
            
            # -------------------------------------------------------------
            # ADIM 3: TOPLAM HATA (TOTAL LOSS) VE OPTİMİZASYON
            # -------------------------------------------------------------
            # Model şunları yapmaya çalışıyor:
            # 1. Hastalıkları doğru bil (loss_s_class düşsün)
            # 2. Alanları GRL sayesinde ayırt edeme (domain loss'lar matematiksel olarak tersine çalışır)
            loss = loss_s_class + loss_s_domain + loss_t_domain
            
            loss.backward()
            optimizer.step()
            
            total_class_loss += loss_s_class.item()
            total_domain_loss += (loss_s_domain.item() + loss_t_domain.item())
            
        # --- DOĞRULAMA (VALIDATION) AŞAMASI ---
        # Modelin gerçek hastalığı öğrenip öğrenmediğini (PTB-XL val setiyle) kontrol ediyoruz
        model.eval()
        all_labels, all_probs = [], []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Doğrulama]", leave=False)
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Test sırasında GRL kapalıdır (alpha=None)
                class_outputs = model(inputs, alpha=None)
                probs = torch.softmax(class_outputs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        try:
            val_auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            val_auroc = 0.0
            
        avg_class_loss = total_class_loss / len_dataloader
        avg_domain_loss = total_domain_loss / len_dataloader
        
        print(f"Alpha: {alpha:.3f} | Sınıf Loss: {avg_class_loss:.4f} | Alan Loss: {avg_domain_loss:.4f} | Val AUROC: {val_auroc:.4f}")
        
        # En iyi DANN modelini kaydet
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), save_path)
            print(f"--> YENİ EN İYİ DANN MODELİ KAYDEDİLDİ! (AUROC: {best_val_auroc:.4f})")

# =====================================================================
# 4. ÇALIŞTIRMA BLOKU
# =====================================================================
if __name__ == "__main__":
    # --- 1. Veri Dosyalarını Yükleme ---
    csv_dir = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"
    
    # SOURCE (KAYNAK): PTB-XL (%10'luk kısım veya tamamı)
    source_train_df = pd.read_csv(os.path.join(csv_dir, "train_split_10.csv")) # Veya 100.csv
    source_val_df = pd.read_csv(os.path.join(csv_dir, "val_split.csv"))
    
    # TARGET (HEDEF): Chapman
    # Çin veri setini tarayıp bir dataframe oluşturuyoruz
    chapman_dir = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\Chapman"
    npz_files = glob.glob(os.path.join(chapman_dir, "*.npz"))
    target_list = [{'file_path': f, 'super_class': 'UNKNOWN'} for f in npz_files] 
    target_train_df = pd.DataFrame(target_list)
    
    print(f"Kaynak Veri (PTB-XL): {len(source_train_df)} | Hedef Veri (Chapman): {len(target_train_df)}")
    
    # --- 2. DataLoader'ları Hazırlama ---
    source_loader = DataLoader(Processed_ECG_Dataset(source_train_df), batch_size=32, shuffle=True)
    target_loader = DataLoader(Processed_ECG_Dataset(target_train_df), batch_size=32, shuffle=True)
    val_loader = DataLoader(Processed_ECG_Dataset(source_val_df), batch_size=32, shuffle=False)
    
    # --- 3. Modeli Başlatma ve Eğitme ---
    model = ResNet34_1D_DANN(num_classes=5, input_channels=12)
    save_path = os.path.join(csv_dir, 'best_dann_resnet34.pth')
    
    # Eğitimi başlat
    train_dann_model(model, source_loader, target_loader, val_loader, num_epochs=20, save_path=save_path)