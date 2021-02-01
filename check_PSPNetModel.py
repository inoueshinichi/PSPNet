"""PSPNetの動作確認
"""

from pspnet_model_for_PSPNet import PSPNet

import torch


def main():
    # ネットワークアーキテクチャのチェック
    net = PSPNet(n_classes=21)
    print(net)

    # ダミーデータの作成
    batch_size = 2
    dummy_img = torch.rand(batch_size, 3, 475, 475)

    # 計算
    outputs = net(dummy_img)
    print(outputs)
    print(f"outputs[0].shape {outputs[0].size()}")
    print(f"outputs[1].shape {outputs[1].size()}")

if __name__ == "__main__":
    main()