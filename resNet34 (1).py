import torch
import torch.nn as nn

# =====================================================================
# 1. TEMEL YAPI TAŞI: ARTIK BLOK (Residual Block)
# =====================================================================
class BasicBlock1D(nn.Module):
    """
    ResNet (Residual Network) mimarisinin yapı taşıdır. 
    İçerisinde 2 adet evrişim (convolution) katmanı ve 1 adet "kestirme yol" (skip connection) bulunur.
    Kestirme yol, bilginin kaybolmadan ağın derinliklerine aktarılmasını sağlar.
    """
    # Her bloğun çıkış kanal sayısının giriş kanal sayısına oranını belirler (ResNet-34 için 1'dir)
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        
        # --- 1. Evrişim Katmanı ---
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=7, # DİKKAT: Görüntülerde kernel_size=3 kullanılırken, EKG sinyalleri 
                           # hızlı akan veriler olduğu için formu (QRS kompleksi vs.) 
                           # yakalayabilmek adına pencere boyutu (kernel) 7 olarak geniş tutulmuştur.
            stride=stride, padding=3, bias=False
        )
        # Sinyalleri normalize ederek eğitimin daha hızlı ve stabil (kararlı) olmasını sağlar
        self.bn1 = nn.BatchNorm1d(out_channels)
        # Negatif değerleri sıfırlayarak modele "doğrusal olmayan" (non-linear) öğrenme yeteneği katar
        self.relu = nn.ReLU(inplace=True)
        
        # --- 2. Evrişim Katmanı ---
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # --- KESTİRME YOL (SKIP CONNECTION / SHORTCUT) ---
        # ResNet'i ResNet yapan kısımdır. H(x) = F(x) + x formülündeki "+ x" kısmıdır.
        self.shortcut = nn.Sequential()
        
        # Eğer giriş sinyalinin boyutu (stride != 1) veya kanal sayısı (in_channels != out_channels)
        # ana yoldan çıkacak sonuçla eşleşmiyorsa, kestirme yolu 1x1 evrişim ile ana yola uyduruyoruz (projeksiyon).
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # 1. Ana Yol (Main Path): Sinyal iki evrişimden geçer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 2. Kestirme Yol'un Ana Yola Eklenmesi
        # Sinyalin işlenmemiş (veya boyutlandırılmış) hali, işlenmiş haline doğrudan eklenir.
        out += self.shortcut(x)
        
        # Toplamanın ardından ReLU aktivasyonu uygulanır
        out = self.relu(out)
        
        return out


# =====================================================================
# 2. ANA MİMARİ: 1D ResNet-34
# =====================================================================
class ResNet34_1D(nn.Module):
    """
    1 Boyutlu ResNet-34 Mimarisi.
    Projenizdeki 12 derivasyonlu EKG sinyallerini (12, 5000) alır, 
    özelliklerini çıkarır ve 5 ana hastalık sınıfına (NORM, MI, vb.) ait olasılık skorları üretir.
    """
    def __init__(self, num_classes=5, input_channels=12):
        super(ResNet34_1D, self).__init__()
        self.in_channels = 64
        
        # --- GİRİŞ KATMANI (STEM) ---
        # 12 kanallı EKG sinyalini alır ve 64 filtreli kalın bir katmana yayar.
        # kernel_size=15: Sinyalin büyük bir kısmını tek seferde görerek genel gürültüye karşı direnç sağlar.
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # Sinyalin uzunluğunu (5000) yarı yarıya düşürerek önemli özellikleri öne çıkarır
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # --- RESNET BLOKLARI ---
        # ResNet-34'ün orijinal mimari dizilimi (3, 4, 6, 3 blok)
        # Katmanlar ilerledikçe kanal sayısı artar (64->128->256->512), ancak sinyal uzunluğu kısalır.
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        # --- SINIFLANDIRICI (CLASSIFIER HEAD) ---
        # Sinyalin zaman boyutunu tamamen yok eder, her kanal için 1 adet ortalama değer çıkarır.
        # Bu, modelin farklı uzunluktaki sinyallere adapte olabilmesini sağlar (Eğer ilerde 10 sn değil de 15 sn gelirse çökmez).
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Çıkarılan 512 tane özelliği, projenizdeki 5 ana sınıfa bağlar
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Belirtilen sayıda 'BasicBlock1D' bloğunu zincirleme olarak birbirine bağlar.
        """
        # Sadece ilk bloğun adımı (stride) parametreden gelir (boyutu küçültmek için),
        # Geri kalan blokların adımı her zaman 1'dir.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock1D(self.in_channels, out_channels, stride=s))
            # Bir sonraki bloğun giriş kanalı, mevcut bloğun çıkış kanalı olur
            self.in_channels = out_channels * BasicBlock1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Verinin ağ içindeki seyahat rotası (İleri Besleme)
        x'in başlangıç boyutu: (Batch_Size, 12, 5000)
        """
        # 1. Giriş işlemleri
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 2. Derin özellik çıkarımı (Feature Extraction)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Buraya geldiğinde x'in boyutu yaklaşık: (Batch_Size, 512, zaman_adımı)
        
        # 3. Havuzlama (Pooling) ve Düzleştirme (Flatten)
        x = self.avgpool(x)  # Boyut: (Batch_Size, 512, 1)
        x = x.view(x.size(0), -1) # Tensor'u tek boyutlu hale getir. Boyut: (Batch_Size, 512)
        
        # 4. Sınıflandırma
        out = self.fc(x) # Boyut: (Batch_Size, 5)
        
        return out
    
# =====================================================================
# TEST BLOĞU (Sanity Check)
# =====================================================================
if __name__ == "__main__":
    # Bu blok sadece bu dosya (resnet34_1d.py) tek başına çalıştırıldığında aktif olur.
    # Diğer dosyalardan 'import' edildiğinde çalışmaz.
    print("ResNet34_1D Modeli test ediliyor...")
    
    # Modeli başlat (5 Hastalık Sınıfı, 12 Derivasyonlu EKG girişi)
    model = ResNet34_1D(num_classes=5, input_channels=12)

    # DataLoader'dan gelecek veri formatını taklit eden sahte (dummy) bir tensör oluştur
    # (32 adet EKG, 12 Kanal, 5000 Zaman Adımı)
    dummy_ecg_data = torch.randn(32, 12, 5000)

    # Modeli test et
    output = model(dummy_ecg_data)

    print(f"Giriş Boyutu (Beklenen: 32, 12, 5000)  : {dummy_ecg_data.shape}")
    print(f"Çıkış Boyutu (Beklenen: 32, 5)         : {output.shape}")
    print("Tebrikler! Model mimarisi hatasız çalışıyor.")