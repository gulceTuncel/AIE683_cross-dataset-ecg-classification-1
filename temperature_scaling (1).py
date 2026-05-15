import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Kendi mimarilerimizi import ediyoruz
from resnet34_dann import ResNet34_1D_DANN
from baseline_training import expected_calibration_error # ECE fonksiyonumuzu kullanacağız

# =====================================================================
# 1. VERİ SETİ SINIFI 
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
# 2. TEMPERATURE SCALING MİMARİSİ
# =====================================================================
class ModelWithTemperature(nn.Module):
    """
    Eğitilmiş modelimizi (DANN) içine alan ve çıktılarını T (Temperature)
    parametresi ile ölçeklendiren sarma (wrapper) sınıfı.
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        # T (Temperature) başlangıçta 1.5 olarak başlatılır
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        # DANN modelinden ham çıktıları (Logits) al. 
        # (Test modunda olduğumuz için DANN sadece hastalık tahminini döndürür)
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """ Ham Logit'leri T değerine böler """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

# =====================================================================
# 3. EN İDEAL 'T' DEĞERİNİ BULMA (Optimizasyon)
# =====================================================================
def set_temperature(valid_loader, model, device):
    """
    L-BFGS optimizasyon algoritmasını kullanarak, Doğrulama (Validation) seti 
    üzerinde NLL (Negative Log Likelihood) hatasını en aza indiren en iyi 
    T (Temperature) değerini bulur.
    """
    model.to(device)
    model.eval()
    
    nll_criterion = nn.CrossEntropyLoss().to(device)
    
    # 1. Validation Setinden tüm Logit ve Etiketleri topla
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Logitler Toplanıyor"):
            inputs = inputs.to(device)
            # Logitleri al (Henüz T ile bölünmemiş ham halleri)
            logits = model.model(inputs) 
            all_logits.append(logits)
            all_labels.append(labels)
            
    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)

    # Ölçeklendirilmemiş (Ham) durumun NLL ve ECE'sini hesapla
    before_temperature_nll = nll_criterion(all_logits, all_labels).item()
    before_temperature_ece = expected_calibration_error(all_labels.cpu().numpy(), torch.softmax(all_logits, dim=1).cpu().numpy())
    
    print(f"\n[Kalibrasyon Öncesi - Validation] NLL: {before_temperature_nll:.4f} | ECE: {before_temperature_ece:.4f}")

    # 2. Sadece 'temperature' parametresini eğitecek optimizörü kur
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)

    # 3. L-BFGS için özel kayıp (loss) hesaplama fonksiyonu
    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(model.temperature_scale(all_logits), all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval)

    # 4. Kalibrasyon Sonrası (Yeni) değerleri hesapla
    after_temperature_nll = nll_criterion(model.temperature_scale(all_logits), all_labels).item()
    after_temperature_ece = expected_calibration_error(all_labels.cpu().numpy(), torch.softmax(model.temperature_scale(all_logits), dim=1).detach().cpu().numpy())
    
    optimal_T = model.temperature.item()
    print(f"[Kalibrasyon Sonrası - Validation] İdeal T Değeri: {optimal_T:.4f}")
    print(f"[Kalibrasyon Sonrası - Validation] NLL: {after_temperature_nll:.4f} | ECE: {after_temperature_ece:.4f}")

    return model

# =====================================================================
# 4. YENİ 'T' DEĞERİNİ ÇİN VERİ SETİNDE TEST ETME
# =====================================================================
def test_calibrated_model(model, test_loader, dataset_name, device):
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Test Ediliyor ({dataset_name})"):
            inputs = inputs.to(device)
            
            # Artık model bize T'ye bölünmüş, soğutulmuş mantıklı logitler veriyor
            scaled_logits = model(inputs)
            probs = torch.softmax(scaled_logits, dim=1) 
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    ece = expected_calibration_error(np.array(all_labels), np.array(all_probs))
    
    print("\n" + "="*50)
    print(f"      {dataset_name} KALİBRE EDİLMİŞ SONUÇ")
    print("="*50)
    print(f"Yeni ECE (Kalibrasyon Hatası) : {ece:.4f}")
    print("="*50 + "\n")

# =====================================================================
# ANA BLOK
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Veri Yükleyicileri Hazırla
    csv_dir = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"
    cpsc_dir = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET\Datasets_Processed\CPSC2018"
    
    # Validation için PTB-XL kullanıyoruz (Model burada doğrulandı)
    val_df = pd.read_csv(os.path.join(csv_dir, "val_split.csv"))
    val_loader = DataLoader(Processed_ECG_Dataset(val_df), batch_size=32, shuffle=False)
    
    # Test için problemli olan CPSC 2018 (Çin) verisini yüklüyoruz
    npz_files = glob.glob(os.path.join(cpsc_dir, "*.npz"))
    target_list = []
    for f in npz_files:
        with np.load(f, allow_pickle=True) as data:
            sc = data['superclasses']
            if len(sc) > 0 and sc[0] != 'UNKNOWN':
                target_list.append({'file_path': f, 'super_class': sc[0]})
                
    cpsc_loader = DataLoader(Processed_ECG_Dataset(pd.DataFrame(target_list)), batch_size=32, shuffle=False)
    
    # 2. Eğitilmiş DANN Modelini Yükle
    print("\n1. DANN Modeli Yükleniyor...")
    dann_model = ResNet34_1D_DANN(num_classes=5, input_channels=12)
    dann_model.load_state_dict(torch.load(os.path.join(csv_dir, 'best_dann_resnet34.pth'), map_location=device, weights_only=True))
    
    # Modeli Temperature Scaling zırhıyla kapla
    calibrated_model = ModelWithTemperature(dann_model)
    
    # 3. İdeal Sıcaklığı (T) Bul
    print("\n2. Sıcaklık Ölçeklendirmesi (Temperature Scaling) Uygulanıyor...")
    calibrated_model = set_temperature(val_loader, calibrated_model, device)
    
    # 4. Çin (CPSC) Verisinde Aşırı Özgüvenin Kırıldığını Test Et
    print("\n3. Çin Veri Setinde (CPSC 2018) Yeni Kalibrasyon Test Ediliyor...")
    test_calibrated_model(calibrated_model, cpsc_loader, "CPSC 2018 (Strong Shift)", device)