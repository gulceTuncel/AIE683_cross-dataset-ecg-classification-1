# AIE683 Cross-Dataset ECG Classification - Run Package


> Not: Dataset klasorleri bu pakete dahil edilmemistir. Kodlar, orijinal dataset ve cikti yollarinin ayni kaldigini varsayar.

## Klasor Icerigi

| Sira | Dosya | Amac |
|---:|---|---|
| 00 | `00_model_resnet34.py` | 1D ResNet-34 model mimarisi. |
| 00 | `00_model_resnet34_dann.py` | DANN versiyonu: feature extractor + label classifier + domain discriminator. |
| 01 | `01_preprocess_ecg.py` | Raw ECG verilerini isler, 10 sn / 500 Hz / 12 lead `.npz` dosyalarina donusturur. |
| 02 | `02_create_splits_and_label_budgets.py` | PTB-XL icin patient-based train/val/test split ve 10/25/50/100 budget CSV'leri olusturur. |
| 03 | `03_train_baseline_resnet34.py` | Baseline ResNet-34 egitimi. Budget dosyasina gore checkpoint kaydeder. |
| 04 | `04_train_dann_resnet34.py` | DANN egitimi. PTB-XL source, CPSC/Chapman target-domain input olarak kullanilir. |
| 05 | `05_evaluate_cross_dataset_budget_checkpoints.py` | Baseline ve DANN budget checkpointlerini PTB-XL, Chapman, CPSC uzerinde degerlendirir. |
| 06 | `06_temperature_scaling_evaluation.py` | Temperature Scaling uygular; No_TS ve TS sonuclarini karsilastirir. |
| metadata | `metadata/snomed_to_superclass.xlsx` | Chapman/CPSC SNOMED etiketlerini PTB-XL superclass sistemine eslemek icin kullanilan mapping dosyasi. |

## Calistirma Sirasi

PowerShell'de once bu klasore gelin:

```powershell
cd "C:\Users\sgtun\OneDrive\Desktop\BYM-DONEM 2\AIE683\Paper Writing\DATASET_HANDLING\AIE683_run_package_numbered"
```

### 1. Preprocessing

Raw datasetlerden islenmis `.npz` dosyalari olusturulur.

```powershell
python "01_preprocess_ecg.py"
```

Bu adimi yalnizca islenmis datasetler yoksa veya preprocessing'i bastan almak istiyorsaniz calistirin.

### 2. Train / Validation / Test split ve label budget CSV'leri

```powershell
python "02_create_splits_and_label_budgets.py"
```

Uretilen temel dosyalar:

- `train_split_10.csv`
- `train_split_25.csv`
- `train_split_50.csv`
- `train_split_100.csv`
- `val_split.csv`
- `test_split.csv`

### 3. Baseline egitimi

```powershell
python "03_train_baseline_resnet34.py"
```

Onemli: Bu dosyada hangi budget'in egitilecegi `train_split_XX.csv` ve `best_baseline_resnet34_XX.pth` satirlariyla belirlenir. 10, 25, 50 ve 100 icin ayri ayri calistirilmalidir.

Beklenen checkpointler:

- `best_baseline_resnet34_10.pth`
- `best_baseline_resnet34_25.pth`
- `best_baseline_resnet34_50.pth`
- `best_baseline_resnet34_100.pth`

### 4. DANN egitimi

```powershell
python "04_train_dann_resnet34.py"
```

Onemli: DANN icin source train split budget'e gore degisir, validation sabit kalir. Target-domain label'lari egitimde kullanilmaz; sadece input sinyalleri domain adaptation icin kullanilir.

Beklenen checkpointler:

- `best_dann_resnet34_10.pth`
- `best_dann_resnet34_25.pth`
- `best_dann_resnet34_50.pth`
- `best_dann_resnet34_100.pth`

### 5. Cross-dataset evaluation

```powershell
python "05_evaluate_cross_dataset_budget_checkpoints.py"
```

Bu adim Baseline ve DANN checkpointlerini su datasetlerde degerlendirir:

- PTB-XL test split
- Chapman
- CPSC2018

Cikti dosyalari:

- `cross_dataset_results_budget_checkpoints.json`
- `cross_dataset_results_budget_checkpoints.csv`

### 6. Temperature Scaling evaluation

```powershell
python "06_temperature_scaling_evaluation.py"
```

Bu adim her checkpoint icin:

1. PTB-XL validation set uzerinde temperature `T` degerini fit eder.
2. No_TS ve TS sonuclarini PTB-XL, Chapman, CPSC uzerinde karsilastirir.

Cikti dosyalari:

- `temperature_scaling_results.json`
- `temperature_scaling_results.csv`


## Metrik Politikasi

External datasetlerde bazi superclass'lar bulunmadigi icin Macro AUROC yalnizca test setinde mevcut siniflar uzerinden hesaplanir. Bu nedenle:

- Chapman icin eksik siniflar raporlanmalidir.
- CPSC2018 icin eksik siniflar raporlanmalidir.
- Brier Score ve ECE, modelin tam 5 sinifli olasilik vektoru uzerinden hesaplanir.

## Orijinal Dosyalar Hakkinda

Bu klasordeki dosyalar kopyadir. Asil proje klasorundeki orijinal dosyalar degistirilmemistir.

## Dikkat Edilecek Noktalar

- Dataset path'leri kodlarin icinde halen orijinal konumlari gostermektedir.
- Kodlari baska bilgisayarda calistiracaksaniz path'leri guncellemeniz gerekir.
- `03_train_baseline_resnet34.py` ve `04_train_dann_resnet34.py` budget bazli egitim icin manuel olarak 10/25/50/100 seklinde calistirilmalidir.
- Final raporda kullanılan sonuc dosyalari `cross_dataset_results_budget_checkpoints.csv` ve `temperature_scaling_results.csv` dosyalaridir.
