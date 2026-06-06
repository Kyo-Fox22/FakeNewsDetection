import pandas as pd
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.pardir))

from src import data_processing
from src import model_building

# Setup Argument Parser for command-line interface
parser = argparse.ArgumentParser(description = 'Train a Filipino Fake News Detector')
parser.add_argument('-e', '--epochs', type = int, help = 'Number of times the model sees the whole dataset in training')
parser.add_argument('-ed', '--embed_dim', type = int, default = 5, help = 'Number of embedding dimensions for the model')
parser.add_argument('-cd', '--conv_dim', type = int, default = 4, help = 'Number of convolutional dimensions for the model')
parser.add_argument('-k', '--kernel_size', type = int, default = 5, help = 'Kernel size of the Convolutional and Pooling Layers')
parser.add_argument('-vc', '--vocab_size', type = int, default = 8000, help = 'Size of the vocabulary the model is trained on')
parser.add_argument('-pid', '--pad_id', type = int, default = 3, help = 'Integer used to register as the padding token')
parser.add_argument('-b', '--batch_size', type = int, default = 32, help = 'Size of the batches for the dataloader the model uses')
parser.add_argument('-v', '--verbose', type = bool, default = True, help = 'Whether or not the program should output progress reports')

