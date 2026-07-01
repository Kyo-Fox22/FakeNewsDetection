import os
import pandas as pd
import mlflow
import argparse
from src import data_processing
from src import model_building

# Setup Argument Parser for command line interface
parser = argparse.ArgumentParser(description = 'Train a Filipino Fake News Detector')
parser.add_argument('-e', '--epochs', type = int, help = 'Number of times the model sees the whole dataset in training')
parser.add_argument('-ed', '--embed_dim', type = int, default = 5, help = 'Number of embedding dimensions for the model')
parser.add_argument('-cd', '--conv_dim', type = int, default = 4, help = 'Number of convolutional dimensions for the model')
parser.add_argument('-k', '--kernel_size', type = int, default = 5, help = 'Kernel size of the Convolutional and Pooling Layers')
parser.add_argument('-vc', '--vocab_size', type = int, default = 8000, help = 'Size of the vocabulary the model is trained on')
parser.add_argument('-pid', '--pad_id', type = int, default = 3, help = 'Integer used to register as the padding token')
parser.add_argument('-b', '--batch_size', type = int, default = 32, help = 'Size of the batches for the dataloader the model uses')
parser.add_argument('-v', '--verbose', type = bool, default = True, help = 'Whether or not the program should output progress reports')
parser.add_argument('-rs', '--random_state', type = int, default = 42, help = 'Seed used for the psuedo-random functions in the program')


# TODO
#? Should model default to base 1.0 version or the latest previous run?

args = parser.parse_args()

# Prepare Datasets and max_seq
if args.verbose:
    print('Preparing dataset...')    
dataset_dir = os.path.join('datasets')

csv_datasets = os.listdir(dataset_dir)
csv_datasets.remove('corpus.txt')

dataset_paths = [(csv, os.path.join(dataset_dir, csv)) for csv in csv_datasets]

train, test = None, None

model_dir = os.path.join('models','FakeNewsDetector')
tokenizer_dir = os.path.join('models','bpe')

for df_name, dataset_path in dataset_paths:
    df = pd.read_csv(dataset_path)
    
    if train is None and test is None:
        train, test = data_processing.process_data(
            df = df,
            feature_col = 'Content',
            label_col = 'Label',
            dataset_dir = dataset_dir,
            tokenizer_dir = tokenizer_dir,
            random_state = args.random_state,
            df_name = df_name,
            vocab_size = args.vocab_size
        )
    else:
        train_subset, test_subset = data_processing.process_data(
            df = df,
            feature_col = 'Content',
            label_col = 'Label',
            random_state = args.random_state,
            df_name = df_name,
            vocab_size = args.vocab_size
        )
        
        train = pd.concat([
            train,
            train_subset
        ])
        
        test = pd.concat([
            test,
            test_subset
        ])

if args.verbose:
    print('Train and Test Set Loaded.')

# Get max seq for model config
max_seq = max(pd.concat([
    train['Content'],
    test['Content']
]).apply(len))


# Build the Model
config = {
    'vocab_size': args.vocab_size,
    'embed_dim': args.embed_dim,
    'pad_id': args.pad_id,
    'conv_dim': args.conv_dim,
    'kernel_size': args.kernel_size,
    'max_seq': max_seq,
}

model = model_building.FakeNewsDetector(**config)

# Check if the model has the same params as recently trained models, then increment model_version if true
model_dir = os.path.join('models','FakeNewsDetector')
os.listdir(model_dir)

experiment = model_building.get_experiment(
    exp_name = 'FakeNewsDetector',
    uri_path = os.path.join(model_dir, 'mlflow.db')
)
mlflow.set_experiment(experiment_id = experiment.experiment_id)
 
# Get finished runs database
recent_runs = mlflow.search_runs(
    filter_string = "status = 'FINISHED'",
    order_by = ['end_time DESC', 'metrics.test_loss ASC', 'metrics.train_loss ASC']
)

# Get most recent version configuration
versions = recent_runs['tags.version'].apply(lambda x: float(x) if x is not None else x)
latest_version = max(versions)

expected_params = model_building.get_config(latest_version)

param_mismatch = 0
for k,v in config.items():    
    if not v == expected_params[k]:
        param_mismatch += 1
        
if args.verbose:
    print(f'{param_mismatch} Parameter Mismatch Detected.')

model_version = model_building.get_version(config, args.verbose)

if model_version is None:
    # Get last trained model version
    model_version = float(latest_run['tags.version'])

    if param_mismatch:
        # Increment depend on whether there are massive changes or minor
        model_version += 0.1 if param_mismatch < 6 else 1.0

if args.verbose:
    print(f'Model Version set to {model_version}.')
    
# Check for existing verision runs in database
existing_version_runs = recent_runs[recent_runs['tags.version'] == f'{model_version}'].sort_values(
    by = ['end_time', 'metrics.test_loss', 'metrics.train_loss'], 
    ascending = [False, True, True]
)

# Load Best Recently Trained Model if an existing model version was trained before
if len(existing_version_runs) > 0:
    if args.verbose:
        print(f'Existing Model runs with version {model_version} detected. Loading Best v{model_version} Model.')
        
    model_run_id = existing_version_runs.loc[0, 'run_id']
    artifacts_dir = os.path.join(model_dir, model_run_id, 'artifacts')
    
    model = model_building.load_model_weights(artifacts_dir)
else:
    if args.verbose:
        print(f'No existing model runs with version {model_version} detected.')
        

# Train the Model
model_building.train_model(
    model = model,
    model_config = config,
    epochs = args.epochs,
    datasets = (train, test),
    batch_size = args.batch_size,
    verbose = args.verbose,
    model_version = model_version
)

if args.verbose:
    print('Model Training Finished.')
