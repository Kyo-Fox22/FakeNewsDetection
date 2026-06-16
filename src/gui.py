from PyQt6.QtWidgets import (QApplication,  QMainWindow, QWidget, QLabel, 
                             QLineEdit, QTextEdit, QPushButton, QVBoxLayout)

import os
import mlflow
from model_building import get_experiment, load_latest_model, model_predict, FakeNewsDetector

# Prepare GUI
class MainWindow(QMainWindow):
    def __init__(self, model: FakeNewsDetector, config: dict):
        super().__init__()
        
        # For Prediction
        self.model = model
        self.model_config = config
        
        self.setWindowTitle('Detect Fake News')
            
        # Create Layout
        layout = QVBoxLayout()
        
        self.authlabel = QLabel('<h4>Enter Author:</h4>')
        self.authbox = QLineEdit()
        self.authbox.setFixedSize(200,30)
        
        self.contentlabel = QLabel('<h4>Enter Content:</h4>')
        self.contentbox = QTextEdit()
        self.contentbox.setFixedSize(250,200)
        
        self.predictbutton = QPushButton('Make Prediction')
        self.predictbutton.clicked.connect(self.predict)
        self.predictionbox = QLabel('<h4>Output:</h4>No prediction yet.')
        # If fake: The given news is likely fake news.
        # If not fake: The given news is likely authentic news.
        
        layout.addWidget(self.authlabel)
        layout.addWidget(self.authbox)
        layout.addWidget(self.contentlabel)
        layout.addWidget(self.contentbox)
        layout.addWidget(self.predictbutton)
        layout.addWidget(self.predictionbox)
        
        # Create Main Window
        window = QWidget()
        window.setLayout(layout)
        self.setCentralWidget(window)
        self.setFixedSize(270,380)
        
    def predict(self):
        author = self.authbox.text()
        content = self.contentbox.toPlainText()
        
        content_threshold = 100
        
        try:
            if content == '':
                self.predictionbox.setText('<h4>Output:</h4>No given content detected.')
            elif len(content) <= content_threshold:
                self.predictionbox.setText('<h4>Output:</h4>Given content too short.')
            
            if content != '' and len(content) > content_threshold:
                is_fake = model_predict(
                    inputs = {
                        'author': author if author != '' else 'Unknown', 
                        'content': content
                        },
                    model = self.model,
                    max_seq = self.model_config['max_seq']
                )
                
                if is_fake:
                    self.predictionbox.setText('<h4>Output:</h4>The given news is likely fake news.')
                else:
                    self.predictionbox.setText('<h4>Output:</h4>The given news is likely authentic news.')
                
        except:
            self.predictionbox.setText('<h4>Output:</h4>An unexpected error occurred.')
         
        return
    

model_dir = os.path.join('..','models','FakeNewsDetector')

# Set experiment
experiment = get_experiment('FakeNewsDetector', os.path.join(model_dir, 'mlflow.db'))
mlflow.set_experiment(experiment_id = experiment.experiment_id)

model, config = load_latest_model(model_dir)

app = QApplication([])

window = MainWindow(model, config)
window.show()

app.exec()