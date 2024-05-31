import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer


class ESDSegmentation(pl.LightningModule):
    def __init__(
        self,
        model_type,
        in_channels,
        out_channels,
        learning_rate=1e-3,
        model_params: dict = {},
    ):
        """
        Constructor for ESDSegmentation class.
        """
        # call the constructor of the parent class
        super(ESDSegmentation, self).__init__()
        # use self.save_hyperparameters to ensure that the module will load
        self.save_hyperparameters()
        # store in_channels and out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate
        # if the model type is segmentation_cnn, initalize a segmentation_cnn as self.model
        if model_type == "segmentation_cnn" or model_type.lower() == "segmentationcnn":
            self.model = SegmentationCNN(in_channels=in_channels, out_channels=out_channels, **model_params)
        elif model_type.lower() == "unet":         # if the model type is unet, initialize a unet as self.model
            self.model = UNet(in_channels=in_channels, out_channels=out_channels, **model_params)
        elif model_type == "fcn_resnet_transfer" or model_type.lower() == "fcnresnettransfer": # if the model type is fcn_resnet_transfer, initialize a fcn_resnet_transfer as self.model
            self.model = FCNResnetTransfer(in_channels=in_channels, out_channels=out_channels, **model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # initialize the accuracy metrics for the semantic segmentation task
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_channels)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_channels)
    def forward(self, X):
        X = torch.nan_to_num(X)
        # evaluate self.model
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch
        # evaluate batch
        predictions = self(sat_img)
        # calculate cross entropy loss
        loss = nn.CrossEntropyLoss()(predictions, mask.long())
        
        # calculate training accuracy
        preds = torch.argmax(predictions, dim=1)
        train_acc = self.train_accuracy(preds, mask.long())
        
        # log training loss and accuracy
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch
        # evaluate batch for validation
        predictions = self(sat_img)
        # get the class with the highest probability
        preds = torch.argmax(predictions, dim=1)
        # calculate validation accuracy
        val_acc = self.val_accuracy(preds, mask.long())
        
        # log validation accuracy
        self.log('val_accuracy', val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # calculate validation loss
        val_loss = nn.CrossEntropyLoss()(predictions, mask.long())
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return val_loss
    
    def configure_optimizers(self):
        # initialize optimizer
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
