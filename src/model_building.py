import pandas as pd
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import mlflow
import json
import sentencepiece as spm

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

def load_model_weights(artifacts_dir: str, return_config: bool = False) -> tuple[FakeNewsDetector, dict]:
    """Load existing model weights through saved artifacts.

    Args:
        artifacts_dir (str): Path to model artifacts that will be loaded to model.
        return_config (bool, optional): Whether or not to return the model config retrieved from artifact directory. Defaults to False.

    Returns:
        tuple[FakeNewsDetector, dict]: an instance of FakeNewsDetector class that's been loaded with existing model weights. 
        If return_config is True, this also returns a config.
    """
    # Get config
    with open(os.path.join(artifacts_dir, 'config.json')) as f:
        config = json.load(f)
    
    # Load model
    model = FakeNewsDetector(**config)
    
    # Load weights
    model.load_state_dict(torch.load(
        os.path.join(artifacts_dir, 'weights.pt')
    ))
    
    return (model, config) if return_config else model

def load_latest_model(model_dir: str, return_runs: bool = False) -> tuple[FakeNewsDetector, pd.DataFrame]:
    """Load the latest FakeNewsDetector model that was tracked by an mlflow experiment.

    Args:
        model_dir (str): The directory path to where the models are saved.
        return_runs (bool): Whether to return the records of finished runs. Defaults to False.

    Returns:
        tuple[FakeNewsDetector, dict, pd.DataFrame]: latest trained model along with its parameter configuration
        and all recent runs dataframe if return_runs is set to True.
    """
    
    # Get all recent finished runs
    all_runs = mlflow.search_runs(
        filter_string = "status = 'FINISHED'",
        order_by = ['end_time DESC', 'metrics.test_loss ASC', 'metrics.train_loss ASC']
    )
    
    # Get latest run id
    latest_run_id = all_runs.run_id[0]
    
    # Get latest model artifacts
    latest_artifact_dir = os.path.join(model_dir, latest_run_id, 'artifacts')
    
    model, config = load_model_weights(latest_artifact_dir, return_config = True)
    
    return (model, config, all_runs) if return_runs else (model, config)

def train_model(model: FakeNewsDetector, model_config: dict, epochs: int, 
                datasets: tuple[pd.DataFrame, pd.DataFrame], model_version: float,
                batch_size: int = 32, lr: float = 0.001, optim = None, loss_func = None, 
                verbose: bool = False, **kwargs) -> None:
    """Train a FakeNewsDetector model under the given parameters.

    Args:
        model (FakeNewsDetector): The model to use in training. Must be an instance of FakeNewsDetector.
        model_config (dict): Model Configuration used in the given model.
        epochs (int): Number of times to train the model.
        datasets (tuple[pd.DataFrame, pd.DataFrame]): Tuple of Dataframes containing the train and test respectively.
        model_version (float): Version number to save the model as within the database.
        batch_size (int, optional): Size of the batches created by the dataloaders in training. Defaults to 32.
        lr (float, optional): Learning rate used by the optimizers. Defaults to 0.001.
        optim (_type_, optional): Optimizers used to gradually change model weights. If None, Adam is used. Defaults to None.
        loss_func (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.
    """
    # DataLoader Kwargs
    train, test = datasets[0], datasets[1]
    feature_col = kwargs.get('feature_col', 'Content')
    label_col = kwargs.get('label_col', 'Label')
    pad_idx = kwargs.get('pad_idx', 0)
    batch_first = kwargs.get('batch_first', True)
    shuffle = kwargs.get('shuffle', True)
    
    if verbose:
        print('DataLoader Kwargs Loaded.')
        
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
        
    # Model Optimizer, Loss, and Version
    if optim is None:
        optim = torch.optim.Adam(model.parameters(), lr)
        
    if loss_func is None:
        loss_func = nn.BCEWithLogitsLoss()
        
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

def get_tokenizer(tokenizer_dir: str) -> spm.SentencePieceProcessor:
    """Load an spm tokenizer used for the model's training and predictions.

    Args:
        tokenizer_dir (str): Relative or Absolute path to the tokenizer model directory.

    Returns:
        spm.SentencePieceProcessor: latest trained model of the tokenizer.
    """
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(os.path.join(tokenizer_dir, 'spm.model'))
    
    return tokenizer

def model_predict(inputs: dict, model: FakeNewsDetector, max_seq: int, **kwargs) -> int:
    """Use a model to predict the probability that a given Philippine author and news content is likely to be fake.

    Args:
        inputs (dict): A dictionary containing an 'author' and 'content'.
        model (FakeNewsDetector): A trained FakeNewsDetector model that is used for the raw logits prediction.
        max_seq (int): Max sequence length to pad the content.

    Returns:
        int: Integer 1 or 0 to predict whether the given set of author and content is fake (1) or authentic (0).
    """
    
    # Get Model Configs
    tokenizer_dir = kwargs.get('tokenizer_dir',os.path.join('..','models','bpe'))
    pad_id = model.embed.padding_idx
    
    # Get tokenizer
    tokenizer = get_tokenizer(tokenizer_dir)
    
    # Format
    formatted = f'{inputs.get('author','Unknown')}: {inputs.get('content')}'
    encoded = torch.tensor([1] + tokenizer.encode(formatted))
    batched = encoded.unsqueeze(0)
    
    # Pad 
    pad_tensor = torch.tensor([pad_id for i in range(max_seq - batched.size(1))])
    pad_tensor = pad_tensor.unsqueeze(0)
    padded = torch.concat([
        batched,
        pad_tensor
    ], dim = -1)
    
    # Prediction
    with torch.inference_mode():
        logits = model(padded)
        prediction = 1 if logits.item() > 0 else 0
    
    return prediction