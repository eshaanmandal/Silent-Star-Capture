import torch
import torch.nn as nn

def main_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, padding=(1,1)),
        nn.ReLU(inplace=True)
    )
    return conv

def side_conv(in_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, 3, 3, 1, padding=(1,1)),
        nn.ReLU(inplace=True)
    )
    return conv

def simple_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, padding=(1,1))
    )
    return conv

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.main_conv1 = main_conv(3, 63) # the first convolution 
        self.main_conv2  = main_conv(63, 63) # the repeated convolutions

        self.side_conv1 = side_conv(3) # the first side convolution
        self.side_conv2 = side_conv(63) # the repeated side convolution
 
        self.simple_conv = simple_conv(63, 63) # the simple convolution without any RELU activation
           
    def forward(self, img):
        x1 = self.main_conv1(img)
        x2 = self.side_conv1(img) #


        x3 = self.main_conv2(x1)
        x4 = self.side_conv2(x1)  #

        x5 = self.main_conv2(x3)
        x6 = self.side_conv2(x3) #

        x7 = self.main_conv2(x5)
        x8 = self.side_conv2(x5) #

        x9 = self.main_conv2(x7)
        x10 = self.side_conv2(x7) #

        x11 = self.main_conv2(x9)
        x12 = self.side_conv2(x9)  #

        x13 = self.simple_conv(x11)
        x14 = self.side_conv2(x11) #

        x15 = self.simple_conv(x13)
        x16 = self.side_conv2(x13) #

        y = img + (x2 + x4 + x6 + x8 + x10 + x12 + x14 + x16)/ 8
        
        return y



net = PNet().to('cpu')

img = torch.rand((1,3, 256, 256))

print(net(img))