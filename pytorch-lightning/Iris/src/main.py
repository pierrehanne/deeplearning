from dataset import IrisDataModule
from model import MLPClassifier
from pytorch_lightning import Trainer

if __name__ == '__main__':
      
    # Create the dataset    
    dm = IrisDataModule()
    
    # Prepare the dataloaders
    dm.setup()
    
    # Create the model
    mlp = MLPClassifier(input_size = 4, hidden_input = 32, hidden_output = 64, output_size = 3)    
    trainer = Trainer(accelerator = "gpu", gpus = 1, max_epochs = 10, enable_progress_bar = True, progress_bar_refresh_rate = 1, log_every_n_steps=1)
    
    # Fit the model on the training dataloader
    trainer.fit(mlp, dm.train_dataloader())
    
    # Evaluate the model on the validation dataloader
    trainer.validate(mlp, dm.val_dataloader())
        
    # Evaluate the model on the testing dataloader
    trainer.test(mlp, dm.test_dataloader())