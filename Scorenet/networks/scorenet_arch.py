import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift

class SCORE(nn.Module):
    def __init__(self, in_channels, num_features, act_type = 'relu', norm_type = None):
    
        super(SCORE, self).__init__()
        #define a siaemese network
        
        
        ##both 2 input will be feed in this conv
        self.conv_1 = ConvBlock(in_channels, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)
        self.conv_2 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)

        self.pool = nn.MaxPool2d(2, 2)
                         
        
        ##2 branch will be merge into 1 
        self.conv_3 = ConvBlock(2*num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)
        self.conv_4 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)
        
        self.conv_5 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)
        self.conv_6 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)
        self.conv_7 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)
        
        self.conv_8 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)
        self.conv_9 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)
        self.conv_10 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)
        
        self.conv_11 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)
        self.conv_12 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)   
        self.conv_13 = ConvBlock(num_features, num_features, kernel_size=3, padding=1, act_type=act_type, norm_type=norm_type)

        self.fc1 = nn.Linear( 7*7*32, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, lr, sr):
        
        #block 1 branch 1
        x1 = self.conv_2(self.conv_1(lr))
        x1 = self.pool(x1)
        
        #block 1 branch 2
        x2 = self.conv_2(self.conv_1(sr))
        x2 = self.pool(x2)
        
        #block 3 merge branch
        x = self.conv_4(self.conv_3(torch.cat((x1,x2),1)))
        x = self.pool(x)
        
        x = self.conv_7(self.conv_6(self.conv_5(x)))
        x = self.pool(x)
        
        x = self.conv_10(self.conv_9(self.conv_8(x)))
        x = self.pool(x)
        
        x = self.conv_13(self.conv_12(self.conv_11(x)))
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1, 7*7*32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x # return output of every timesteps
