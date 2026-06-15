import torch
import torch.nn as nn




class BasicBlock1D(nn.Module):
    """
    ResNet (Residual Network) mimarisinin yapı taşıdır.
    İçerisinde 2 adet evrişim (convolution) katmanı ve 1 adet "kestirme yol" (skip connection) bulunur.
    Kestirme yol, bilginin kaybolmadan ağın derinliklerine aktarılmasını sağlar.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()


        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=7,


            stride=stride, padding=3, bias=False
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)


        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)



        self.shortcut = nn.Sequential()



        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)



        out += self.shortcut(x)


        out = self.relu(out)

        return out





class ResNet34_1D(nn.Module):
    """
    1 Boyutlu ResNet-34 Mimarisi.
    Projenizdeki 12 derivasyonlu EKG sinyallerini (12, 5000) alır,
    özelliklerini çıkarır ve 5 ana hastalık sınıfına (NORM, MI, vb.) ait olasılık skorları üretir.
    """
    def __init__(self, num_classes=5, input_channels=12):
        super(ResNet34_1D, self).__init__()
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


        self.fc = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Belirtilen sayıda 'BasicBlock1D' bloğunu zincirleme olarak birbirine bağlar.
        """


        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock1D(self.in_channels, out_channels, stride=s))

            self.in_channels = out_channels * BasicBlock1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Verinin ağ içindeki seyahat rotası (İleri Besleme)
        x'in başlangıç boyutu: (Batch_Size, 12, 5000)
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)



        x = self.avgpool(x)
        x = x.view(x.size(0), -1)


        out = self.fc(x)

        return out




if __name__ == "__main__":


    print("ResNet34_1D Modeli test ediliyor...")


    model = ResNet34_1D(num_classes=5, input_channels=12)



    dummy_ecg_data = torch.randn(32, 12, 5000)


    output = model(dummy_ecg_data)

    print(f"Giriş Boyutu (Beklenen: 32, 12, 5000)  : {dummy_ecg_data.shape}")
    print(f"Çıkış Boyutu (Beklenen: 32, 5)         : {output.shape}")
    print("Tebrikler! Model mimarisi hatasız çalışıyor.")
