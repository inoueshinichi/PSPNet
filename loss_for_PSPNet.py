"""PSPNetの損失関数
"""

# サードパーティ
import torch
import torch.nn as nn
from torch.functional import F
import torch.optim as optim
import torchvision
from torchvision import models, transforms

class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        """損失関数の計算

        Args:
            outputs PSPNetの出力(tuple): (output=torch.Size([num_batch, 21, 475, 475]),
                                         outupt_aux=torch.Size([num_batch, 21, 475, 475]))

            targets [num_batch, 475, 475]: 正解のアノテーション情報

        Returns:
            loss : テンソル 損失
        """

        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')

        return loss + self.aux_weight * loss_aux

    