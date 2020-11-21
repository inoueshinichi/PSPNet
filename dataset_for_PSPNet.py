from data_argmentation_for_PSPNet import (
    Compose,
    Scale,
    RandomRotation,
    RandomMirror,
    Resize,
    Normalize_Tensor
)


class DataTransform():
    """
    画像とアノテーションの前処理クラス.訓練時と検証時で異なる動作をする.
    画像サイズ(input_size, input_size)にする.
    訓練時は水増しを行う.
    
    Attributes
    ----------
    input_size : int リサイズ先の大きさ.
    color_mean : (R,G,B) チャネルの平均値
    color_std  : (R,G,B) チャネルの標準偏差
    """
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5,1.5]),
                RandomRotation(angle=[-10,10]),
                RandomMirror(),
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ]),
            'val': Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }
        
    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)
        
from torch.utils.data import Dataset
from PIL import Image

class VOCDataset(Dataset):
    """
    VOC2012のDatasetを作成するクラス.PytorchのDatasetクラスを継承
    
    Attributes
    ----------
    img_list : list 画像のパスを格納したリスト
    anno_list : list アノテーションへのパスを格納したリスト
    phase : 'train' or 'val'
    transform : object 前処理クラス
    """
    
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, anno_class = self.pull_item(index)
        return img, anno_class
        
    def pull_item(self, index):
        """
        画像のTensor形式データ、アノテーションを取得
        """
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path) # (RGB)
        
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path) #[h][w][カラーパレット番号]
        
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)
#         print(img, anno_class_img)
        
        return img, anno_class_img