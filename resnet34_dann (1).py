import torch
import torch.nn as nn
from torch.autograd import Function
from resNet34 import BasicBlock1D # Daha önce yazdığımız temel ResNet bloğunu kullanıyoruz

# =====================================================================
# 1. GRL: GRADIENT REVERSAL LAYER (Gradyan Ters Çevirme Katmanı)
# =====================================================================
class GradientReversalLayer(Function):
    """
    İleri beslemede (Forward) hiçbir şeye dokunmadan veriyi geçirir.
    Geri yayılımda (Backward) ise türevi eksi (negatif) alfa ile çarpar.
    Bu sayede model, "Alan (Domain)" bilgisini öğrenmek yerine onu UNUTMAYI öğrenir.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradyanı tersine çeviren sihirli satır:
        output = grad_output.neg() * ctx.alpha
        return output, None

# =====================================================================
# 2. DANN MİMARİSİ: ResNet34_1D_DANN
# =====================================================================
class ResNet34_1D_DANN(nn.Module):
    def __init__(self, num_classes=5, input_channels=12):
        super(ResNet34_1D_DANN, self).__init__()
        
        # ---------------------------------------------------------
        # A) FEATURE EXTRACTOR (Özellik Çıkarıcı - Orijinal ResNet34)
        # ---------------------------------------------------------
        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # ---------------------------------------------------------
        # B) LABEL PREDICTOR (Hastalık Sınıflandırıcı)
        # ---------------------------------------------------------
        # Çıkarılan 512 özelliği alıp 5 farklı kalp hastalığına dönüştürür.
        self.class_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Aşırı öğrenmeyi (overfitting) engellemek için
            nn.Linear(128, num_classes)
        )

        # ---------------------------------------------------------
        # C) DOMAIN DISCRIMINATOR (Alan Ayrıştırıcı / Kötü Polis)
        # ---------------------------------------------------------
        # Verinin PTB-XL'den (0) mi yoksa Chapman/CPSC'den (1) mi geldiğini bulmaya çalışır.
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # Sadece 2 sınıf: Kaynak (Source) ve Hedef (Target)
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock1D(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * BasicBlock1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x, alpha=None):
        """
        Model artık 1 değil, 2 farklı çıktı üretiyor!
        - Eğitim aşamasında hem Hastalık (class_output) hem de Alan (domain_output) üretir.
        - Test aşamasında (alpha=None) sadece Hastalık üretmesi yeterlidir.
        """
        # 1. Ortak Özellik Çıkarımı
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        features = x.view(x.size(0), -1) # Tensor düzleştirilir: (Batch, 512)

        # 2. Hastalık Tahmini (Ana Görev)
        class_output = self.class_classifier(features)

        # 3. Alan Tahmini (Sadece Eğitimde alpha verilirse çalışır)
        if alpha is not None:
            # GRL Devrede: Özellikler GRL'den geçip Alan Ayrıştırıcıya gider
            reverse_features = GradientReversalLayer.apply(features, alpha)
            domain_output = self.domain_classifier(reverse_features)
            return class_output, domain_output
        else:
            # Test sırasında Alan Tahminine gerek yoktur
            return class_output

# =====================================================================
# TEST BLOĞU
# =====================================================================
if __name__ == "__main__":
    print("DANN Mimarisi Test Ediliyor...")
    model = ResNet34_1D_DANN(num_classes=5, input_channels=12)
    
    dummy_data = torch.randn(32, 12, 5000)
    
    # Sadece test modunda çalışma (1 çıktı)
    out_class = model(dummy_data)
    print("Test Çıktısı Boyutu (Beklenen: 32, 5):", out_class.shape)
    
    # Eğitim modunda çalışma (2 çıktı - Alpha değeri ile)
    out_class, out_domain = model(dummy_data, alpha=1.0)
    print("Eğitim - Sınıf Çıktısı (Beklenen: 32, 5):", out_class.shape)
    print("Eğitim - Alan Çıktısı  (Beklenen: 32, 2):", out_domain.shape)
    
    print("Mükemmel! DANN Modeli eğitime hazır.")