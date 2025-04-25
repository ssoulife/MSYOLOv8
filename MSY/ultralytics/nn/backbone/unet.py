import torch
import torch.nn as nn
import torch.nn.functional as F

# Double Convolution
class DoubleConv2d(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.ReLU(True),
            nn.Conv2d(outputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


# Down Sampling
class DownSampling(nn.Module):
    def __init__(self):
        super(DownSampling, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out = self.down(x)
        return out


# Up Sampling
class UpSampling(nn.Module):

    # Use the deconvolution
    def __init__(self, inputChannel, outputChannel):
        super(UpSampling, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(inputChannel, outputChannel, kernel_size=2, stride=2),
            nn.BatchNorm2d(outputChannel)
        )

    def forward(self, x, y):
        x =self.up(x)
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        out = torch.cat([y, x], dim=1)
        return out


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.layer1 = DoubleConv2d(1, 64)
        self.layer2 = DoubleConv2d(64, 128)
        self.layer3 = DoubleConv2d(128, 256)
        self.layer4 = DoubleConv2d(256, 512)
        self.layer5 = DoubleConv2d(512, 1024)
        self.layer6 = DoubleConv2d(1024, 512)
        self.layer7 = DoubleConv2d(512, 256)
        self.layer8 = DoubleConv2d(256, 128)
        self.layer9 = DoubleConv2d(128, 64)

        self.layer10 = nn.Conv2d(64, 2, kernel_size=3, padding=1)  # The last output layer

        self.down = DownSampling()
        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)

    def forward(self, x):
        conv1 = self.layer1(x)
        down1 = self.down(conv1)
        conv2 = self.layer2(down1)
        down2 = self.down(conv2)
        conv3 = self.layer3(down2)
        down3 = self.down(conv3)
        conv4 = self.layer4(down3)
        down4 = self.down(conv4)
        conv5 = self.layer5(down4)
        up1 = self.up1(conv5, conv4)
        conv6 = self.layer6(up1)
        up2 = self.up2(conv6, conv3)
        conv7 = self.layer7(up2)
        up3 = self.up3(conv7, conv2)
        conv8 = self.layer8(up3)
        up4 = self.up4(conv8, conv1)
        conv9 = self.layer9(up4)
        out = self.layer10(conv9)
        return out


# Test part
if __name__ == '__main__':
    mynet = Unet()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # mynet.to(device)
    input = torch.rand(3, 1, 572, 572)
    # output = mynet(input.to(device))
    output = mynet(input)
    print(output.shape)  # (3,2,572,572)
