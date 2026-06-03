import pandas as pd
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import mlflow
import json

# Classes
class FakeNewsDataset(Dataset):
    def __init__(self, feature, label):
        super().__init__()
        self.feature = feature
        self.label = label
        
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        feature = self.feature[idx]
        label = self.label[idx]
        
        return feature, label
    
class FakeNewsDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_id, conv_dim, kernel_size, max_seq):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, pad_id)
        self.conv1d = nn.Conv1d(embed_dim, conv_dim, kernel_size)
        self.pool1d = nn.MaxPool1d(kernel_size)
        self.flat = nn.Flatten()
        
        # Calculate Flattened shape
        with torch.no_grad():
            dummy = torch.zeros(1, max_seq, dtype = torch.int64)
            embed = self.embed(dummy)
            conv1d = self.conv1d(
                embed.transpose(1,2)
            )
            pool1d = self.pool1d(conv1d)
            flat = self.flat(pool1d)
            flattened_size = flat.size(1)
        
        self.fc = nn.Linear(flattened_size, 1)
        
    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(1,2)
        x = self.conv1d(x)
        x = self.pool1d(x)
        x = self.flat(x)
        x = self.fc(x)
        return x
    

# Functions
def feature_tensor_pad(train, test, feature_col: str, 
                       pad_idx: int = 3, batch_first: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a padded tensor using the feature column of both train and test.

    Args:
        train (_type_): The train dataset
        test (_type_): The test dataset
        feature_col (str): Name of the feature column
        pad_idx (int, optional): Value to use as pad in tensor. Defaults to 3.
        batch_first (bool, optional): Whether or not to have the batch size as the first dimension of the resulting tensor shape. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Train and Test tensors with padded values.
    """
      
    merged = pd.concat([
        train[feature_col],
        test[feature_col]
    ])
    
    padded = pad_sequence(merged, batch_first = batch_first, padding_value = pad_idx)
    
    train, test = padded[:len(train), :], padded[len(train):, :]
    
    return train, test

def create_dataloaders(train, test, feature_col: str, label_col: str, **kwargs) -> tuple[DataLoader, DataLoader]:
    """Create the train and test dataloaders for the model.

    Args:
        train (_type_): The train dataset.
        test (_type_): The test dataset.
        feature_col (str): The name of the feature column for both datasets.
        label_col (str): The name of the label column for both datasets.

    Returns:
        tuple[DataLoader, DataLoader]: Dataloaders for both train and test set.
    """
    
    # Kwargs
    pad_idx = kwargs.get('pad_idx', 0)
    batch_first = kwargs.get('batch_first', True)
    batch_size = kwargs.get('batch_size', 32)
    shuffle = kwargs.get('shuffle', True)
    
    # Convert Feature and Labels to tensors
    train[feature_col] = train[feature_col].apply(lambda x: torch.tensor(x))
    train[label_col] = train[label_col].apply(lambda x: torch.tensor(x))
    
    test[feature_col] = test[feature_col].apply(lambda x: torch.tensor(x))
    test[label_col] = test[label_col].apply(lambda x: torch.tensor(x))
    
    train_x, test_x = feature_tensor_pad(
        train = train, 
        test = test,
        feature_col = feature_col,
        pad_idx = pad_idx,
        batch_first = batch_first
    )
    
    train_set = FakeNewsDataset(train_x, train[label_col])
    test_set = FakeNewsDataset(test_x, test[label_col])
    
    train_loader = DataLoader(
        dataset = train_set,
        batch_size = batch_size,
        shuffle = shuffle
    )
    
    test_loader = DataLoader(
        dataset = test_set,
        batch_size = batch_size,
        shuffle = shuffle
    )
    
    return train_loader, test_loader

def get_experiment(exp_name: str, uri_path: str, artifact_location: str = None) -> mlflow.entities.Experiment:
    """Retrieves an existing experiment with the given experiment name. If there is none, it creates an experiment name with the given artifact location.

    Args:
        exp_name (str): Name of the experiment to get. If there is none, it creates an experiment using this name.
        uri_path (str): Path to the sqlite database file that stores mlflow experiments and runs.
        artifact_location (str, optional): Location to store the artifact if an experiment is created. Defaults to None.

    Returns:
        mlflow.entities.Experiment: MLFlow experiment object used to track metrics and artifacts.
    """
    
    # Set Tracking Uri
    mlflow.set_tracking_uri('sqlite:///' + uri_path)
    
    # Get Existing Experiment
    experiment = mlflow.get_experiment_by_name(exp_name)
    
    # Create Experiment if None
    if experiment is None:
        experiment = mlflow.create_experiment(exp_name, artifact_location)
    
    return experiment

def load_latest_model(model: FakeNewsDetector, model_dir: str, 
                      return_runs: bool = False) -> pd.DataFrame:
    """Load the latest FakeNewsDetector model that was tracked by an mlflow experiment.

    Args:
        model (FakeNewsDetector): A FakeNewsDetector model that would be used to load the latest model.
        model_dir (str): The directory path to where the models are saved.
        return_runs (bool): Whether to return the records of finished runs. Defaults to False.
    """
    
    # Get all recent finished runs
    all_runs = mlflow.search_runs(
        filter_string = "status = 'FINISHED'",
        order_by = ['end_time DESC', 'metrics.test_loss ASC', 'metrics.train_loss ASC'],
        search_all_experiments = True
    )
    
    # Get latest run id
    latest_run_id = all_runs.run_id[0]
    
    # Load latest model artifacts
    latest_artifact_dir = os.path.join(model_dir, latest_run_id, 'artifacts')
    
    model.load_state_dict(torch.load(
        os.path.join(latest_artifact_dir, 'weights.pt')
    ))
    
    return all_runs if return_runs else None

def train_model(model: FakeNewsDetector, model_config: dict, epochs: int, 
                datasets: tuple[pd.DataFrame, pd.DataFrame], model_version: float = None,
                batch_size: int = 32, lr: float = 0.001, optim = None, loss_func = None, 
                verbose: bool = False, **kwargs) -> None:
    # Experiment Kwargs
    exp_name = kwargs.get('exp_name', 'FakeNewsDetector')
    model_dir = os.path.join('..','models',exp_name)
    
    uri_path = kwargs.get(
        'uri_path', 
        os.path.join(model_dir, 'mlflow.db')
    )
    
    artifact_location = kwargs.get('artifact_location', model_dir)
    
    if verbose:
        print('Experiment Kwargs Loaded.')
    
    # DataLoader Kwargs
    train, test = datasets[0], datasets[1]
    feature_col = kwargs.get('feature_col', 'Content')
    label_col = kwargs.get('label_col', 'Label')
    pad_idx = kwargs.get('pad_idx', 0)
    batch_first = kwargs.get('batch_first', True)
    shuffle = kwargs.get('shuffle', True)
    
    if verbose:
        print('DataLoader Kwargs Loaded.')
        
    # Set Experiment
    experiment = get_experiment(exp_name, uri_path, artifact_location)
    experiment = mlflow.set_experiment(experiment_id = experiment.experiment_id)
    
    if verbose:
        print(f'Experiment set to {experiment.experiment_id}.')
        
    # Process DataLoaders
    trainloader, testloader = create_dataloaders(
        train = train,
        test = test,
        feature_col = feature_col,
        label_col = label_col,
        pad_idx = pad_idx,
        batch_size = batch_size,
        batch_first = batch_first,
        shuffle = shuffle
    )
    
    if verbose:
        print('DataLoaders generated.')
        
    # Load Latest Model
    all_runs = load_latest_model(model, model_dir, return_runs = True)
    
    if verbose:
        print('Latest Model Loaded.')
        
    # Model Optimizer, Loss, and Version
    if optim is None:
        optim = torch.optim.Adam(model.parameters(), lr)
        
    if loss_func is None:
        loss_func = nn.BCEWithLogitsLoss()
        
    if model_version is None:
        model_version = all_runs['tags.version'][0]
        
    # Main Loop
    if verbose:
        print('All configurations completed. Initiating Main Loop.')
        
    with mlflow.start_run():
        
        # Log Model Configuration
        mlflow.log_params(model_config)
            
        for epoch in range(epochs):
            train_loss = 0
            for batch in trainloader:
                X, y = batch[0], batch[1]
                
                logits = model(X)
                loss = loss_func(
                    logits.reshape(-1),
                    y.to(torch.float32)
                )
                
                train_loss += loss.item()
                
                optim.zero_grad()
                loss.backward()
                optim.step()
            
            # Log metric to current run
            mlflow.log_metric('train_loss', train_loss/len(trainloader), step = epoch)
            
            test_loss = 0
            for batch in testloader:
                X, y = batch[0], batch[1]
                
                logits = model(X)
                loss = loss_func(
                    logits.reshape(-1),
                    y.to(torch.float32)
                )
                
                test_loss += loss.item()
            
            # Log metric to current run
            mlflow.log_metric('test_loss', test_loss/len(testloader))
            
            if verbose:
                print(
                    f'Epoch: {epoch} |',
                    f'Train Loss: {train_loss/len(trainloader)} |',
                    f'Test Loss: {test_loss/len(testloader)}'
                )
        
        # Create temporary files for artifact storing
        torch.save(
            model.state_dict(),
            'weights.pt'
        )
        
        with open('config.json', 'w') as f:
            json.dump(model_config, f, indent = 2)
            
        mlflow.log_artifact('weights.pt')
        mlflow.log_artifact('config.json')
        
        os.remove('weights.pt')
        os.remove('config.json')
        
        mlflow.set_tag('version', model_version)
    
    return