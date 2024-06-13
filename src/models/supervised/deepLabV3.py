import pytorch_lightning as pl
import torch.nn as nn
from torchvision import models

class DeepLabV3Module(pl.LightningModule):
    def __init__(self, in_channels, out_channels, scale_factor: int = 50):
        super(DeepLabV3Module, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        
        self.model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Adjust classifier for num of classes
        self.model.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=(1, 1), stride=(1, 1))

        # create a MaxPool2d of kernel size scale_factor as the final pooling layer
        self.pool = nn.MaxPool2d(kernel_size=scale_factor)

    
    def forward(self, x):
        x = self.model(x)['out']
        x = self.pool(x)
        return x