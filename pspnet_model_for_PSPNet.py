# 標準


# サードパーティ
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms

class PSPNet(nn.Module):

    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        # パラメータ設定
        block_config = [3, 4, 6, 3] # resnet50
        img_size = 475  # 入力サイズ
        img_size_8 = 60 # img_sizeの1/8

        # Encoder
        # 4つのモジュールを構成するサブネットワークの用意
        self.feature_conv = FeatureMap_convolution()

        self.feature_res_1 = ResidualBlockPSP(n_blocks=block_config[0], 
                                              in_channels=128,
                                              mid_channels=64,
                                              out_channels=256,
                                              stride=1, 
                                              dilation=1)

        self.feature_res_2 = ResidualBlockPSP(n_blocks=block_config[1],
                                              in_channels=256,
                                              mid_channels=128,
                                              out_channels=512,
                                              stride=2, 
                                              dilation=1)

        self.feature_dilated_res_1 = ResidualBlockPSP(n_blocks=block_config[2],
                                                      in_channels=512,
                                                      mid_channels=256,
                                                      out_channels=1024,
                                                      stride=1, 
                                                      dilation=2)

        self.feature_dilated_res_2 = ResidualBlockPSP(n_blocks=block_config[3],
                                                      in_channels=1024,
                                                      mid_channels=512,
                                                      out_channels=2048,
                                                      stride=1,
                                                      dilation=4)

        # Pyramid pooling
        self.pyramid_pooling = PyramidPooling(in_channels=2048,
                                              pool_sizes[6, 3, 2, 1],
                                              height=img_size_8,
                                              width=img_size_8)

        # Decoder
        self.decode_feature = DecodePSPFeature(height=img_size,
                                               width=img_size,
                                               n_classes=n_classes)

        # Auxilliary-Loss
        self.aux = AuxilliaryPSPlayers(in_channels=1024,
                                       height=img_size,
                                       width=img_size,
                                       n_classes=n_classes)

        
    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)
        output_aux = self.aux(x) # Encoderの途中出力をAuxレイヤーに入力
        x = self.feature_dilated_res_2(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return (output, output_aux)


"""FeatureMap_convolutionの実装

input <- (3, 475, 475)
1) Conv2DBatchNormRelu
    conv 3x3
    batch_norm
    relu
output -> (64, 238, 238)

input <- (64, 238, 238)
2) Conv2DBatchNormRelu
    conv 3x3
    batch_norm
    relu
output -> (64, 238, 238)

input <- (64, 238, 238)
3) Conv2DBatchNormRelu
    conv 3x3
    batch_norm
    relu
output -> (128, 238, 238)

input <- (128, 238, 238)
4) MaxPooling 2x2
output -> (128, 119, 119)

"""

# Conv2dBatchNormRelu
class Conv2dBatchNormRelu(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2dBatchNormRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, stride, 
                              padding, dilation, bias=bias)
        
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # inplace設定で入力をメモリに保存せずに出力を計算してメモリを削減する
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        outputs = self.relu(x)
        return outputs

# FeatureMap_convolution
class FeatureMap_convolution(nn.Module):

    def __init__(self):
        super(FeatureMap_convolution, self).__init__()

        # 畳み込み層1
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        stride = 2
        padding = 1
        dilation = 1
        bias = False
        self.chnr_1 = Conv2dBatchNormRelu(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          padding,
                                          dilation,
                                          bias)
        
        # 畳み込み層2
        in_channels = 64
        out_channels = 64
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        bias = False
        self.chnr_2 = Conv2dBatchNormRelu(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          padding,
                                          dilation,
                                          bias)
        
        # 畳み込み層3
        in_channels = 64
        out_channels = 128
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        bias = False
        self.chnr_3 = Conv2dBatchNormRelu(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          padding,
                                          dilation,
                                          bias)

        # MaxPooling層
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.chnr_1.forward(x)
        x = self.chnr_2.forward(x)
        x = self.chnr_3.forward(x)
        outputs = self.maxpooling(x)
        return outputs
    

""" ResidualBlockPSPの実装
    1) bottleNeckPSP
    2) bottoeNeckIdentifyPSP x N 回
    NはResidualBlockPSPサブネットワークを4回繰り返すが、
    それぞれ、3, 4, 6, 3回繰り返す.
"""
class ResidualBlockPSP(nn.Sequential):

    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        # bottleNeckPSPの用意
        self.add_module(
            "block1",
            BottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation)
        )
        
        # bottleNeckIdentifyPSPの繰り返しを用意
        for i in range(n_blocks - 1):
            self.add_module(
                "bottle" + str(i + 2),
                BottleNeckIdentifyPSP(out_channels, mid_channels, out_channels, stride, dilation)
            )


""" BottleNeckPSPとBottleNeckIdentifyPSPの実装
    1) BottleNeckPSP
        input
        |------------------------------------
        |                                   |
        |                                  Conv2dBatchNormRelu(C1 -> BatchNorm -> Relu)
        |                                   |
        Conv2dBatchNorm(C1 -> BatchNoru)   Conv2dBatchNormRelu(C3 -> BatchNorm -> Relu)
        |                                   |
        |                                  Conv2dBatchNorm(C1 -> BatchNorm)
        |                                   |
        -------------------------------------
        |
        output

    2) BottleNeckIdentifyPSP
        input
        |------------------------------------
        |                                   |
        |                                  Conv2dBatchNormRelu(C1 -> BatchNorm -> Relu)
        |                                   |
        y=f(x)                             Conv2dBatchNormRelu(C3 -> BatchNorm -> Relu)
        |                                   |
        |                                  Conv2dBatchNorm(C1 -> BatchNorm)
        |                                   |
        -------------------------------------
        |
        output
"""
class Conv2dNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2dNorm, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)
        
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)
        return outputs

    
class BottleNeckPSP(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(BottleNeckPSP, self).__init__()

        self.cbr_1 = Conv2dBatchNormRelu(in_channels=in_channels,
                                         out_channels=mid_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         bias=False)
                                
        self.cbr_2 = Conv2dBatchNormRelu(in_channels=mid_channels,
                                         out_channels=mid_channels,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=dilation,
                                         dilation=dilation,
                                         bias=False)
        
        self.cb_3 = Conv2dBatchNormRelu(in_channels=mid_channels,
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         bias=False)

        # ショートカットコネクション
        self.cb_residual = Conv2dNorm(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      dilation=1,
                                      bias=False)
        
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)
        output = self.relu(conv + residual)
        return output

    
class BottleNeckIdentifyPSP(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(BottleNeckIdentifyPSP, self).__init__()

        self.cbr_1 = Conv2dBatchNormRelu(in_channels=in_channels,
                                         out_channels=mid_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         bias=False)
                                
        self.cbr_2 = Conv2dBatchNormRelu(in_channels=mid_channels,
                                         out_channels=mid_channels,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=dilation,
                                         dilation=dilation,
                                         bias=False)
        
        self.cb_3 = Conv2dBatchNormRelu(in_channels=mid_channels,
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         bias=False)

        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = x
        output = self.relu(conv + residual)
        return output

    
                                
