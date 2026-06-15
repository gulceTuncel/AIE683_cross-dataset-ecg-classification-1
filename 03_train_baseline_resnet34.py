import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from helpers.model_resnet34 import ResNet34_1D, BasicBlock1D
import os
import json




def multiclass_brier_score(y_true, y_prob):
    """
    Brier Score: Modelin tahmin olasılıklarının gerçeklikle ne kadar örtüştüğünü (Calibration) ölçer.
    Düşük olması modelin kendine güveninde ne kadar "haklı" olduğunu gösterir.
    """
    num_classes = y_prob.shape[1]

    y_true_onehot = np.eye(num_classes)[y_true]

    brier_score = np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1))
    return brier_score

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    ECE (Expected Calibration Error): Model "Bu %90 ihtimalle Kalp Krizi (MI)" dediğinde,
    gerçekten vakaların %90'ında MI çıkıp çıkmadığını kontrol eder.
    Distribution Shift (Alan kayması) durumlarında en çok bozulan metrik budur.
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

def evaluate_all_metrics(y_true, y_prob):
    """
    Tüm tahminleri alıp, projenizin metodolojisinde belirttiğiniz 4 ana metriği hesaplar.
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


    print("\n" + "="*40)
    print("      TEST SETİ SONUÇ RAPORU")
    print("="*40)
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Macro AUROC      : {macro_auroc:.4f}")
    print(f"Brier Score      : {brier:.4f}  (Düşük olması iyidir)")
    print(f"ECE (10 Bins)    : {ece:.4f}  (Düşük olması iyidir)")
    print("="*40 + "\n")

    return {'Macro_F1': macro_f1, 'Macro_AUROC': macro_auroc, 'Brier_Score': brier, 'ECE': ece}




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
            weight_tensor[idx] = 1.0
    return weight_tensor




def train_baseline_model(model, train_loader, val_loader, class_weights, num_epochs=20, save_path='best_baseline.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim başlatılıyor. Donanım: {device}")

    model.to(device)
    class_weights = class_weights.to(device)


    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_auroc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")


        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc="Eğitim", leave=False)

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)


        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Doğrulama", leave=False)
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

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


        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), save_path)
            print(f"--> Yeni en iyi model kaydedildi! (AUROC: {best_val_auroc:.4f})")




def test_and_evaluate_model(model, test_loader, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    print(f"\n[{weights_path}] ağırlıkları yükleniyor ve Test Setinde değerlendiriliyor...")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

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


    metrics = evaluate_all_metrics(all_labels, all_probs)
    return metrics




if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader




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




    csv_dir = r"C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING"

    print("Veri listeleri yükleniyor...")
    train_df = pd.read_csv(os.path.join(csv_dir, "train_split_10.csv"))
    val_df = pd.read_csv(os.path.join(csv_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(csv_dir, "test_split.csv"))




    print("DataLoader'lar hazırlanıyor...")
    train_dataset = Processed_ECG_Dataset(train_df)
    val_dataset = Processed_ECG_Dataset(val_df)
    test_dataset = Processed_ECG_Dataset(test_df)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




    weights = get_class_weights(train_df)


    model = ResNet34_1D(num_classes=5, input_channels=12)
    save_path = os.path.join(csv_dir, 'best_baseline_resnet34_10.pth')


    train_baseline_model(model, train_loader, val_loader, weights, num_epochs=20, save_path=save_path)





    final_metrics = test_and_evaluate_model(model, test_loader, weights_path=save_path)


    metrics_save_path = os.path.join(csv_dir, "baseline_test_metrics_10.json")

    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)

    print(f"✅ Tüm metrikler kalıcı olarak kaydedildi: {metrics_save_path}")
