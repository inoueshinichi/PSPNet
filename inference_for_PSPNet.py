"""PSPNetの推論を行う
   前処理クラスの関数を動作させてエラー回避させるために、
   必要ないがアノテーション画像を1枚用意しておく. このアノテーション画像は処理で使用されない.
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.utils import data

from datapath_for_PSPNet import make_datapath_list, DataTransform
from pspnet_model_for_PSPNet import PSPNet


def main():

    # ファイルパスリスト作成
    rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
    # 後ほどアノテーション画像のみを使用する

    # PSPNetを用意
    net = PSPNet(n_classes=21)
    state_dict = torch.load("./weights/pspnet50_30.pth", map_location={'cuda:0': 'cpu'})
    net.load_state_dict(state_dict)
    print("ネットワーク設定完了 : 学習済みの重みを読み込みました.")

    """推論"""

    # 元画像の表示
    img_file_path = "./data/cowboy-757575_640.jpg"
    img = Image.open(img_file_path)
    img_width, img_height = img.size
    plt.imshow(img)
    plt.show()

    # 前処理クラス作成
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)

    # 前処理
    # 適当なアノテーション画像を用意し、さらにカラーパレットの情報を抜き出す
    anno_file_path = val_anno_list[0]
    anno_class_img = Image.open(anno_file_path)
    p_palette = anno_class_img.getpalette()
    phase = 'val'
    img, anno_class_img = transform(phase, img, anno_class_img)

    # PSPNetで推論する
    net.eval()
    x = img.unsqueeze(0) # ミニバッチ化 torch.Size([1, 3, 475, 475])
    outputs = net(x)
    y = outputs[0] # AuxLoss側は無視. yのサイズはtorch.Size([1, 21, 475, 475])

    # PSPNetの出力から最大クラスを求め、カラーパレット形式にして、画像サイズをもとに戻す
    y = y[0].detach().numpy()
    y = np.argmax(y, axis=0)
    anno_class_img = Image.fromarray(np.uint8(y), mode="P")
    anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palette)
    plt.imshow(anno_class_img)
    plt.show()

    # 画像を透過させて重ねる
    trans_img = Image.new("RGBA", anno_class_img.size, (0, 0, 0, 0))
    anno_class_img = anno_class_img.convert("RGBA")

    for x in range(img_width):
        for y in range(img_height):
            # 推論結果画像のピクセルデータを取得
            pixel = anno_class_img.getpixel((x,y))
            r,g,b,a= pixel

            # (0,0,0)の背景ならそのままにして透過させる
            if r == 0 and g == 0 and b == 0:
                continue
            else:
                # それ以外の色は用意した画像にピクセルを書き込む
                trans_img.putpixel((x,y), (r,g,b,150))
                # 透過度150
    
    img = Image.open(img_file_path)
    result = Image.alpha_composite(img.convert("RGBA"), trans_img) # 原画像と推論結果画像を重ねる
    plt.imshow(result)
    plt.show()
