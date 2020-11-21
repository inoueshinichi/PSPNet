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
                                              stride=1, dilation=1)
        self.feature_res_2 = ResidualBlockPSP(n_blocks=block_config[1],
                                              in_channels=256,
                                              mid_channels=128,
                                              out_channels=512,
                                              stride=2, dilation=1)
        self.feature_dilated_res_1 = ResidualBlockPSP(n_blocks=block_config[2],
                                                      in_channels=512,
                                                      mid_channels=256,
                                                      out_channels=1024,
                                                      stride=1, dilation=2)
        self.feature_dilated_res_2 = ResidualBlockPSP(n_blocks=block_config[3],
                                                      in_channels=1024,
                                                      mid_channels=512,
                                                      out_channels=2048,
                                                      stride=1, dilation=4)

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



    

        
