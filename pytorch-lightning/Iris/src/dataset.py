# Libraries
from os import cpu_count
from sklearn.datasets import load_iris
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import Tensor


class IrisDataModule(LightningDataModule):
    """Iris Custom Lightning Data Module"""
    def __init__(self, nb_classes: int = 3, train_split_ratio: float = .7, batch_size: int = 16) -> None:
        """Initialize the IrisDataModule"""
        super().__init__()  
        
        # init dataset parameters
        self.data = None
        self.features = None
        self.labels = None  
        
        # define the number of classes          
        self.nb_classes = nb_classes
        
        # init batch size
        self.batch_size = batch_size
        
        # init dataloaders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        
    def prepare_data(self):
        """prepare the data"""
        # load the data
        dataset = load_iris()
        
        # define the features and labels
        self.features = dataset.data
        self.labels = dataset.target
        
        # convert the features and labels to tensors
        self.features = Tensor(self.features).float()
        self.labels = Tensor(self.labels).long()
        
        # create the dataset
        self.data = TensorDataset(self.features, self.labels)        
        return self.data
        
        
    def setup(self, stage: str = None):
        """setup the data"""
        
        # prepare the data
        self.data = self.prepare_data()        

        # setup reproducible results
        RANDOM_SEED = 42
        seed_everything(RANDOM_SEED)
        
        # split the dataset into train, val and test
        if stage == "train" or stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = random_split(self.data, [130, 20])
        
        if stage == "test" or stage is None:
            self.train_dataset, self.test_dataset = random_split(self.train_dataset, [110, 20])


    def train_dataloader(self):
        """return the train dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=min(8, cpu_count()), shuffle=False, pin_memory=True)

    
    def val_dataloader(self):
        """return the val dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=min(8, cpu_count()), shuffle=False, pin_memory=True)

    
    def test_dataloader(self):
        """return the test dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=min(8, cpu_count()), shuffle=False, pin_memory=True)