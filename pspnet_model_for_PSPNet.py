# 標準


# サードパーティ
import torch
import torch.nn as nn
from torch.functional import F
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
                                              pool_sizes=[6, 3, 2, 1],
                                              height=img_size_8,
                                              width=img_size_8)

        # Decoder
        self.decode_feature = DecodePSPFeature(height=img_size,
                                               width=img_size,
                                               n_classes=n_classes)

        # Auxilliary-Loss
        self.aux = AuxiliaryPSPLayers(in_channels=1024,
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
        self.add_module("block1",
            BottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))
        
        # bottleNeckIdentifyPSPの繰り返しを用意
        for i in range(n_blocks - 1):
            self.add_module("bottle" + str(i + 2),
                BottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))


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
class Conv2dBatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2dBatchNorm, self).__init__()

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
        
        self.cb_3 = Conv2dBatchNorm(in_channels=mid_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    dilation=1,
                                    bias=False)

        # ショートカットコネクション
        self.cb_residual = Conv2dBatchNorm(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=1,
                                           stride=stride,
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

    def __init__(self, in_channels, mid_channels, stride, dilation):
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
                                         stride=1,
                                         padding=dilation,
                                         dilation=dilation,
                                         bias=False)
        
        self.cb_3 = Conv2dBatchNorm(in_channels=mid_channels,
                                    out_channels=in_channels,
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

    
                                
# PyramidPooling
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()
        
        # forwardで使用する画像サイズ
        self.height = height
        self.width = width

        # 各畳込み層の出力チャネル数
        out_channels = int(in_channels / len(pool_sizes))

        # 各畳込み層を作成
        self.avg_pool_list = []
        self.cbr_list = []
        for divide in pool_sizes:
            avg_pool = nn.AdaptiveAvgPool2d(output_size=divide)
            cbr = Conv2dBatchNormRelu(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      dilation=1,
                                      bias=False)
            self.avg_pool_list.append(avg_pool)
            self.cbr_list.append(cbr)


    def forward(self, x):
        pyramid_poolings = [x]
        for avg_pool, cbr in zip(self.avg_pool_list, self.cbr_list):
            out = cbr(avg_pool(x)) # (512,h,w)
            # Deconvolution(Upsampling)
            out = F.interpolate(out, size=(self.height, self.width), mode="bilinear", align_corners=True)
            pyramid_poolings.append(out)

        # pyramid_poolingの4つの出力の各チャネル数は512, h:60, w:60
        # PyramidPoolingの入力前と4つの出力を(N,C,H,W)のCの次元で連結
        output = torch.cat(pyramid_poolings, dim=1)
        return output



# Decoder
class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()

        self.height = height
        self.width = width

        self.cbr = Conv2dBatchNormRelu(in_channels=4096,
                                       out_channels=512,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       bias=False)
        
        self.dropout = nn.Dropout2d(p=0.1)

        self.classification = nn.Conv2d(in_channels=512,
                                        out_channels=n_classes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        # upsampling
        output = F.interpolate(x,
                               size=(self.height, self.width),
                               mode='bilinear',
                               align_corners=True)
        return output


# Aux
class AuxiliaryPSPLayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super(AuxiliaryPSPLayers, self).__init__()

        self.height = height
        self.width = width

        self.cbr = Conv2dBatchNormRelu(in_channels=in_channels,
                                       out_channels=256,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       bias=False)
        
        self.dropout = nn.Dropout2d(p=0.1)

        self.classification = nn.Conv2d(in_channels=256,
                                        out_channels=n_classes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        # upsampling
        output = F.interpolate(x,
                               size=(self.height, self.width),
                               mode='bilinear',
                               align_corners=True)
        return output


