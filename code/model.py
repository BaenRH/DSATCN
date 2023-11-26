
import torch
import torch.nn as nn

class f_Conv(nn.Module):
    def __init__(self,input_dim,output_dim,stride=1,padding=1,kernel_size=3,pool_size=2):
        super(f_Conv,self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim,output_dim,stride=stride,padding=padding,kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, stride=stride, padding=padding, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
        )

    def forward(self,x):
        out =self.conv_block(x)

        return out


class Novel_CNN(nn.Module):
    def __init__(self,win=3,channel=1,filters=[32,64,128,256,512,1024,2048],len_signal=1024):
        super(Novel_CNN,self).__init__()

        self.conv1 = f_Conv(channel,filters[0],kernel_size=win,padding=(win-1)//2)
        self.conv2 = f_Conv(filters[0],filters[1], kernel_size=win, padding=(win - 1) // 2)
        self.conv3 = f_Conv(filters[1], filters[2], kernel_size=win, padding=(win - 1) // 2)
        self.conv4 = f_Conv(filters[2], filters[3], kernel_size=win, padding=(win - 1) // 2)
        self.conv5 = f_Conv(filters[3], filters[4], kernel_size=win, padding=(win - 1) // 2)
        self.conv6 = f_Conv(filters[4], filters[5], kernel_size=win, padding=(win - 1) // 2)
        self.conv7 = nn.Sequential(
            nn.Conv1d(filters[5],filters[6],stride=1,padding=(win - 1) // 2,kernel_size=win),
            nn.ReLU(),
            nn.Conv1d(filters[6], filters[6], stride=1, padding=(win - 1) // 2, kernel_size=win),
            nn.ReLU(),
            )
        self.flatten =nn.Flatten()
        self.dense = nn.Linear(filters[6]*(len_signal//2**6),1024)       #6是池化的层数

    def forward(self,input):
        if input.dim() == 2:
            input = input.unsqueeze(1)
        layer1out = self.conv1(input)
        layer2out = self.conv2(layer1out)
        layer3out = self.conv3(layer2out)
        layer4out = self.conv4(layer3out)
        layer5out = self.conv5(layer4out)
        layer6out = self.conv6(layer5out)
        layer7out = self.conv7(layer6out)
        flatten1 = self.flatten(layer7out)
        out = self.dense(flatten1)


        return out


if __name__ == "__main__":
    data = torch.ones((2, 1, 1024))
    model = Novel_CNN()
    out = model(data)
    print(out.shape)



