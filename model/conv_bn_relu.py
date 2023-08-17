import torch.nn as nn

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, kernel_size=3, pad=1 ,stride=1):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=pad),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class ConvBNReluDrop(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNReluDrop, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.02)
        )

    def forward(self, x):
        return self.layers(x)


class Resblock(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    expansion = 1
    def __init__(self, channels_in, channels_out, stride=1):

        super(Resblock, self).__init__()
        
        self.reslayers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1,bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out*Resblock.expansion, 3, stride, padding=1,bias=False),
            nn.BatchNorm2d(channels_out*Resblock.expansion)
        )

        self.shorcut = nn.Sequential()
        if stride!=1 or channels_in != channels_out * Resblock.expansion:
            self.shorcut = nn.Sequential(
                    nn.Conv2d(channels_in , channels_out*Resblock.expansion,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm2d(channels_out*Resblock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.reslayers(x)+self.shorcut(x))
    
class SEblock(nn.Module):
    def __init__(self,channels,reduction) -> None:
        super(SEblock,self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                    nn.Linear(channels,channels//reduction,bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels//reduction,channels,bias=False),
                    nn.Sigmoid()
        )
    def forward(self,x):
        b,c = x.shape[:2]
        v = self.global_pooling(x).view(b,c)
        v = self.fc(v).view(b,c,1,1)
        return x * v.expand_as(x)

class SE_Resblock(nn.Module):
    expansion = 1
    def __init__(self,reduction, channels_in, channels_out, stride=1) -> None:
        super(SE_Resblock,self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                    nn.Linear(channels_out,channels_out//reduction,bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels_out//reduction,channels_out,bias=False),
                    nn.Sigmoid()
        )
        
        self.reslayers = nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, stride, padding=1,bias=False),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels_out, channels_out*SE_Resblock.expansion, 3, stride, padding=1,bias=False),
        nn.BatchNorm2d(channels_out*SE_Resblock.expansion)
    )

        self.shorcut = nn.Sequential()
        if stride!=1 or channels_in != channels_out * Resblock.expansion:
            self.shorcut = nn.Sequential(
                    nn.Conv2d(channels_in , channels_out*SE_Resblock.expansion,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm2d(channels_out*SE_Resblock.expansion)
            )
    def forward(self,x):
        y = x
        x = self.reslayers(x)
        b,c = x.shape[:2]
        v = self.global_pooling(x).view(b,c)
        v = self.fc(v).view(b,c,1,1)
        scaled = x * v.expand_as(x)
        return nn.ReLU(inplace=True)(scaled+self.shorcut(y))

class Image_SEblock(nn.Module):
    def __init__(self,channels,reduction,h=640,w=640) -> None:
        super(Image_SEblock,self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size=(17,17),stride=16,padding=8)
        self.fc = nn.Sequential(
                    nn.Linear(channels*h//16*w//16,channels//reduction,bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels//reduction,channels*h//16*w//16,bias=False),
                    nn.Sigmoid()
        )
        self.unpool = nn.Upsample((h,w))
    def forward(self,x):
        b,c,h,w= x.shape
        v = self.pooling(x).view(b,c*h//16*w//16)
        # print(v.shape)
        v = self.fc(v).view(b,c,h//16,w//16)
        # print(v.shape)
        v_up = self.unpool(v)
        return x * v_up
    
class ConvBNRelu_LN(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, H=256,W=256,kernel_size=3, pad=1 ,stride=1):

        super(ConvBNRelu_LN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=pad),
            nn.LayerNorm([channels_out,H,W]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)



class Resblock_LN(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    expansion = 1
    def __init__(self, channels_in, channels_out, H=256,W=256,stride=1):

        super(Resblock_LN, self).__init__()
        
        self.reslayers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1,bias=False),
            nn.LayerNorm([channels_out,H,W]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out*Resblock.expansion, 3, stride, padding=1,bias=False),
            nn.LayerNorm([channels_out*Resblock.expansion,H,W])
        )

        self.shorcut = nn.Sequential()
        if stride!=1 or channels_in != channels_out * Resblock.expansion:
            self.shorcut = nn.Sequential(
                    nn.Conv2d(channels_in , channels_out*Resblock.expansion,kernel_size=1,stride=stride,bias=False),
                    nn.LayerNorm([channels_out*Resblock.expansion,H,W])
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.reslayers(x)+self.shorcut(x))
    
