import torch
import torch.nn as nn


class Conv_transition(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        # [1,3,5], int, int
        super(Conv_transition, self).__init__()
        if not kernel_size:
            kernel_size = [1, 3, 5]
        paddings = [int(a / 2) for a in kernel_size]
        # print(paddings)#[0,1,2]
        # self.Conv0=nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1)
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[0], stride=1, padding=paddings[0])
        self.Conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[1], stride=1, padding=paddings[1])
        self.Conv3 = nn.Conv2d(in_channels, out_channels, kernel_size[2], stride=1, padding=paddings[2])
        #self.Conv4 = nn.Conv2d(in_channels, out_channels, kernel_size[3], stride=1, padding=paddings[3])
        self.Conv_f = nn.Conv2d(3 * out_channels, out_channels, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        # print('x.shape',x.shape)#tensor:[4,3/5/7/9,128,128]
        # x = self.Conv0(x)
        x1 = self.act(self.Conv1(x))
        x2 = self.act(self.Conv2(x))
        x3 = self.act(self.Conv3(x))
        #x4 = self.act(self.Conv4(x))
        x = torch.cat([x1, x2, x3], dim=1)
        return self.act(self.bn(self.Conv_f(x)))
    



class Inception(nn.Module):
    def __init__(self,in_ch):
        super(Inception, self).__init__()

        self.incep1_2 = self.transition(in_ch, 2)
        self.incep1_3 = self.transition(in_ch, 3)
        self.incep1_4 = self.transition(in_ch, 4)
        self.incep1_5 = self.transition(in_ch, 5)
        self.incep1_6 = self.transition(in_ch, 6)
        
    def forward(self,x):
        incep1_2 = self.incep1_2(x)
        incep1_3 = self.incep1_3(x)
        incep1_4 = self.incep1_4(x)
        incep1_5 = self.incep1_5(x)
        incep1_6 = self.incep1_6(x)

        x = torch.cat((incep1_2, incep1_3, incep1_4, incep1_5, incep1_6), dim=1)
        
        return x
    def transition(self,in_channels, out_channels):
        layers = []
        layers.append(Conv_transition([1, 3, 5], in_channels, out_channels))
        return nn.Sequential(*layers)
    
if __name__ == "__main__":
    model = Inception()
    a = torch.randn(1,3,128,128)
    b = model(a)
    print(b.shape)
    