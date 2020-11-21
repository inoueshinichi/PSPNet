import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np

"""
torchvisionのTransformsオブジェクトを用いて水増しするインターフェースクラス
注意) アノテーション画像はカラーパレット形式(index_color画像).
"""
class Compose(object):
    """
    引数transformsに格納された変形を順番に実行するクラス
    対象画像とアノテーション画像を同時に変換する
    """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img
        
        
# 画像のスケールを変換する
class Scale(object):

    def __init__(self, scale):
        self.scale = scale # [min_ratio, max_ratio]
        
    def __call__(self, img, snno_class_img):
        
        # pil_img.size = [幅][高さ]
        width = img.size[0] 
        height = img.size[1]
        
        # 拡大倍率をランダムに設定
        scale = np.random.uniform(self.scale[0], self.scale[1])
        
        scaled_w = int(width * scale)
        scaled_h = int(height * scale)
        
        # 画像のりサイズ
        img = img.resize((scaled_w, scaled_h), \
                         Image.BICUBIC) # 双対二次補完
        
        # アノテーションサイズのりサイズ
        anno_class_img = anno_class_img.resize((scaled_w, scaled_h), \
                                               Image.NEAREST) # 最近棒補完
        
        # 画像を元の大きさに切り出して位置を求める
        if scale > 1.0:
            # 左上
            left = scaled_w - width
            left = int(np.random.uniform(0, left))
            top = scaled_h - height
            top = int(np.random.uniform(0, top))
            
            # クロップ
            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop(
                            (left, top, left+width, top+height))
            
        else:
            # input_sizeより短い辺はpaddingする
            p_palette = anno_class_img.copy().getpalette()
            
            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()
            
            # 左上
            pad_width = width - scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))
            pad_height = height - scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))
            
            # 黒画像を作成
            img = Image.new(img.mode, (width, height), (0, 0, 0))
            
            # 埋め込み
            img.paste(img_original, (pad_width_left, pad_height_top))
            
            # 黒画像作成 for カラーパレット形式
            anno_class_img = Image.new(
                                anno_class_img.mode, (width, height), (0))
            
            # 埋め込み
            anno_class_img.paste(anno_class_img_original,
                                (pad_width_left, pad_height_top))
            
            # カラーパレット情報のコピー
            anno_class_img.putpalette(p_palette)
            
        return img, anno_class_img
    

#ランダムに回転させる
class RandomRotation(object):
 	
   def __init__(self, angle):
       self.angle = angle # [min_deg, max_deg]
        
   def __call__(self, img, anno_class_img):
        
       # 回転角度を決める
       rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))
        
       # 回転
       img = img.rotate(rotate_angle, Image.BILINEAR) # 双対1次補完
       anno_class_img = img.rotate(rotate_angle, Image.NEAREST) # 最近棒補完
        
       return img, anno_class_img
        

# 50%の確率で左右にフリップ
class RandomMirror(object):
    
    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
            
        return img, anno_class_img
        
        
# 画像形状をリサイズする
class Resize(object):
    """
    引数input_sizeに大きさを変形するクラス
    """
    def __init__(self, input_size):
        self.input_size = input_size # width == height
        
    def __call__(self, img, anno_class_img):
        
        img = img.resize((self.input_size, self.input_size), Image.BICUBIC) # 双対二次補完
        anno_class_img = anno_class_img.resize((self.input_size, self.input_size), Image.NEAREST) # 最近棒補完
        
        return img, anno_class_img
        
        
# 0-255 -> 0-1に正規化してtorch.Tensorに変換
class Normalize_Tensor(object):
    
   def __init__(self, color_mean, color_std):
       self.color_mean = color_mean
       self.color_std = color_std
        
   def __call__(self, img, anno_class_img):
       # PIL画像をtorch.Tensorに変換. 大きさを最大1に規格化すru
       img = transforms.functional.to_tensor(img)
        
       # 色情報の標準化
       img = transforms.functional.normalize(
               img, self.color_mean, self.color_std)
        
       # アノテーション画像をNumpyに変換
       anno_class_img = np.array(anno_class_img) # [高さ][幅]
        
       # 'ambigious'には255が格納されているので、0の背景にしておく
       index = np.where(anno_class_img == 255)
       anno_class_img[index] = 0
        
       # アノテーション画像をtorch.Tensorに変換
       anno_class_img = torch.from_numpy(anno_class_img)
        
       return img, anno_class_img
        
                             
