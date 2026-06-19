from PyQt6.QtWidgets import QApplication

import os
import mlflow
from src.model_building import get_experiment, load_latest_model
from src.gui import MainWindow

model_dir = os.path.join('models','FakeNewsDetector')
tokenizer_dir = os.path.join('models', 'bpe')

# Set experiment
experiment = get_experiment('FakeNewsDetector', os.path.join(model_dir, 'mlflow.db'))
mlflow.set_experiment(experiment_id = experiment.experiment_id)

model, config = load_latest_model(model_dir)

app = QApplication([])

window = MainWindow(
    model,
    config,
    tokenizer_dir
)
window.show()

app.exec()