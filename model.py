import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.down1 = downStep(1, 64, 568, 392)
        self.down2 = downStep(64, 128, 280, 200)
        self.down3 = downStep(128, 256, 136, 104)
        self.down4 = downStep(256, 512, 64, 56)
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.up1 = upStep(1024, 512, withReLU=True)
        self.up2 = upStep(512, 256, withReLU=True)
        self.up3 = upStep(256, 128, withReLU=True)
        self.up4 = upStep(128, 64, withReLU=False)
        self.up5 = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(n_classes)

    def forward(self, x):
        x, x_down1 = self.down1(x)
        x, x_down2 = self.down2(x)
        x, x_down3 = self.down3(x)
        x, x_down4 = self.down4(x)
        x = self.down5(x)
        
        x = self.up1(x, x_down4)
        x = self.up2(x, x_down3)
        x = self.up3(x, x_down2)
        x = self.up4(x, x_down1)
        x = self.up5(x)
        x = self.bn(x)
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC, inS, outS):
        super(downStep, self).__init__()
        self.upper_bound = round((inS - outS) / 2)
        self.lower_bound = inS - self.upper_bound
        self.conv1 = nn.Conv2d(inC, outC, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(outC)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(outC)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x_down = x[:, :, self.upper_bound:self.lower_bound, self.upper_bound:self.lower_bound]
        x = self.pool(x)
        return x, x_down

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        self.withReLU = withReLU
        self.up_conv = nn.ConvTranspose2d(inC, outC, kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(inC, outC, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(outC)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(outC)
        self.relu2 = nn.ReLU()

    def forward(self, x, x_down):
        x = self.up_conv(x)
        x = self.relu1(self.bn1(self.conv1(torch.cat((x, x_down), 1)))) if self.withReLU else self.bn1(self.conv1(torch.cat((x, x_down), 1)))
        x = self.relu2(self.bn2(self.conv2(x))) if self.withReLU else self.bn2(self.conv2(x))
        return x