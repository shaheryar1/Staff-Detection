from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch 
import pytorch_lightning as pl
import torch.optim as optim

def get_custom_mobilenet(num_classes = 2):
    model = mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Modify the last layer for two classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 256)
    model.classifier.append(nn.Dropout(0.2))
    model.classifier.append(nn.Linear(256,num_classes))
    return model


class ImageClassifier(pl.LightningModule):
    def __init__(self, model ,  learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate 
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define model
        self.model = model

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log('train_loss', loss,prog_bar = True)
        self.log('train_acc', acc,prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log('val_loss', loss,prog_bar = True)
        self.log('val_acc', acc, prog_bar = True)

