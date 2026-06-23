from PyQt6.QtWidgets import QApplication

import os
import mlflow
import argparse
from src.model_building import get_experiment, load_latest_model
from src.gui import MainWindow

model_dir = os.path.join('models','FakeNewsDetector')
tokenizer_dir = os.path.join('models', 'bpe')

# Set experiment
experiment = get_experiment('FakeNewsDetector', os.path.join(model_dir, 'mlflow.db'))
mlflow.set_experiment(experiment_id = experiment.experiment_id)

# Parse Arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', type = float, default = None, help = 'Specify the model version to use in the application.')
parser.add_argument('--verbose', type = bool, default = True, help = 'Determine whether to print out progress or status on program.')

args = parser.parse_args()

model_version = args.version

model, config = load_latest_model(
    model_dir = model_dir,
    model_version = model_version,
    verbose = args.verbose
)

if args.verbose:
    print(f'Starting up program.')

app = QApplication([])

window = MainWindow(
    model,
    config,
    tokenizer_dir
)
window.show()

app.exec()