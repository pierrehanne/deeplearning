from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
import torch

   
class MLPClassifier(LightningModule):
    """Multi Layer Perceptron Architecture"""
    def __init__(self, input_size: int, hidden_input: int, hidden_output: int, output_size: int) -> None:
        """Initialize the MLPClassifier"""
        super().__init__()
        
        # add layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_input),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_input, hidden_output),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_output, output_size)
        )
        
        # add loss function
        self.loss = torch.nn.CrossEntropyLoss()
        
        # add metrics
        self.train_acc = Accuracy()            
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        
        
    def forward(self, x):
        return self.layers(x)
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self.layers(x), y)
        y_pred = torch.argmax(self.layers(x), dim=1)
        self.train_acc(y_pred, y)     
        self.log("train_accuracy", self.train_acc.compute(), on_step=False, on_epoch=True)
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}
            
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self.layers(x), y)       
        y_pred = torch.argmax(self.layers(x), dim=1)
        self.val_acc(y_pred, y)     
        self.log("val_accuracy", self.val_acc.compute(), on_step=False, on_epoch=True)
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self.layers(x), y)        
        y_pred = torch.argmax(self.layers(x), dim=1)
        self.test_acc(y_pred, y)     
        self.log("test_accuracy", self.test_acc.compute(), on_step=False, on_epoch=True)
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y} 