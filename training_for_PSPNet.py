"""PSPNetの学習と検証
    ネットワークパラメータの初期値は,weights/pspnet50_ADE20K.pthを利用する.
    この重みはPSPNet発明者の`Hengshuang Zhao`のGithubで公開しているcaffeネットワークモデルのパラメータを
    pytorchで読み込めるように変換したもの.
    ADE20Kデータセット(150クラス, 約2万枚の画像からなるセマセグ用データセット by MIT Computer Visionチーム)
"""

import math
import time

import pandas as pd

import torch
from torch import nn, optim
from torch.utils import data

from datapath_for_PSPNet import make_datapath_list
from dataset_for_PSPNet import DataTransform, VOCDataset

from pspnet_model_for_PSPNet import PSPNet
from loss_for_PSPNet import PSPLoss


def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):

    # GPUが使えるか確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("使用デバイス:", device)

    # ネットワークをGPUのメモリに転送
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloaders_dict['train'].dataset)
    num_val_imgs = len(dataloaders_dict['val'].dataset)
    batch_size = dataloaders_dict['train'].batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # multiple minibatch
    batch_multiplier = 3

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()

        # epochの損失和
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print("----------")
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("----------")

        # epoch毎の訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                scheduler.step() # 最適化シュケジューラの更新
                optimizer.zero_grad()
                print("(train)")

            else:
                if ((epoch + 1) % 5 == 0):
                    net.eval()
                    print("----------")
                    print("(val)")

                else:
                    # 検証は5回に1回だけ行う
                    continue

            # データローダーからminibatchづつ取り出すループ
            count = 0
            for imgs, anno_class_imgs in dataloaders_dict[phase]:

                # ミニバッチサイズが1だとBatchNormalizationでエラーになるので回避
                if imgs.size()[0] == 1:
                    continue

                # GPUが使えるならGPUメモリに転送
                imgs = imgs.to(device)
                anno_class_imgs = anno_class_imgs.to(device)

                # multiple minibatchでのパラメータ更新
                if (phase == 'train') and (count == 0):
                    optimizer.step() # ネットワークパラメータ更新
                    optimizer.zero_grad() # パラメータの勾配値をzeroにする
                    count = batch_multiplier
                
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(imgs)
                    loss = criterion(outputs, anno_class_imgs.long()) / batch_multiplier

                    if phase == 'train':
                        loss.backward() # 勾配の計算
                        count -= 1 # multiple minibatch

                        if (iteration % 10 == 0): # 10iterに一度lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print("Iteration {} || Loss {:.4f} || 10iter {:.4f} sec".format(
                                iteration,
                                loss.item() / batch_size * batch_multiplier,
                                duration
                            ))
                            t_iter_start = time.time()
                        
                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1

                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

                    
        # epochのphase毎のlossと正解率
        t_epoch_finish = time.time()
        print("--------------------")
        print("epoch {} || Epoch_TRAIN_Loss::{:.4f} || Epoch_VAL_Loss::{:.4f}".format(
            epoch + 1,
            epoch_train_loss / num_train_imgs,
            epoch_val_loss / num_val_imgs
        ))
        print("timer: {:.4f} sec.".format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss / num_train_imgs,
            'val_loss': epoch_val_loss / num_val_imgs
        }
        
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log/pspnet_log_output.csv")

    # 最後のネットワークを保存する
    torch.save(net.state_dict(), "weights/pspnet50_" + str(epoch + 1) + ".pth")

def main():

    """データセット"""
    
    # ファイルパスリストの作成
    rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    # Dataset作成
    # (RGB)の平均値と標準偏差
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    train_transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)
    train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train', transform=train_transform)

    val_transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val', transform=val_transform)

    # DataLoader作成
    batch_size = 8

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {
        'train' : train_dataloader,
        'val' : val_dataloader
    }

    """ネットワークモデル"""

    # ファインチューニングでPSPNetを作成
    # ADE20Kデータセットの学習済みモデルを使用する
    # ADK20Kはクラス数が150であることに注意
    net = PSPNet(n_classes=150)

    # ADE20K学習済みパラメータを読み込む
    state_dict = torch.load("./weights/pspnet50_ADK20K.pth")
    net.load_state_dict(state_dict)

    # 分類用の畳み込み数を出力数21のものに付け替える
    n_classes = 21
    net.decode_feature.classification = nn.Conv2d(in_channels=512, 
                                                  out_channels=n_classes, 
                                                  kernel_size=1, 
                                                  stride=1, 
                                                  padding=0)
    net.aux.classification = nn.Conv2d(in_channels=256, 
                                       out_channels=n_classes, 
                                       kernel_size=1, 
                                       stride=1, 
                                       padding=0)

    # 付け替えた畳み込み層をXavierの初期値で初期化(活性化関数がシグモイドだから)
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None: # バイアス項がある場合
                nn.init.constant_(n.bias, 0.0)
    
    net.decode_feature.classification.apply(weights_init)
    net.aux.classification.apply(weights_init)

    print("ネットワーク設定完了 : 学習済みの重みを読み込みました.")

    """損失関数"""
    criterion = PSPLoss(aux_weight=0.4)

    """オプティマイザ&スケジューラ"""
    # ファインチューニングなので、入力に近い層は学習率を小さく、逆に出力に近い層は大きくする
    optimizer = optim.SGD([
        {'params': net.feature_conv.parameters(), 'lr': 1e-3},
        {'params': net.feature_res_1.parameters(), 'lr': 1e-3},
        {'params': net.feature_res_2.parameters(), 'lr': 1e-3},
        {'params': net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
        {'params': net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
        {'params': net.pyramid_pooling.parameters(), 'lr': 1e-3},
        {'params': net.decode_feature.parameters(), 'lr': 1e-2},
        {'params': net.aux.parameters(), 'lr': 1e-2},
    ], momentum=0.9, weight_decay=0.0001)

    # スケジューラの設定
    def lambda_epoch(epoch):
        max_epoch = 30
        return math.pow((1 - epoch / max_epoch), 0.9)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    # 学習と評価
    train_model()







if __name__ == "__main__":
    main()