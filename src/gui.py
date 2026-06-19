from PyQt6.QtWidgets import (QMainWindow, QWidget, QLabel, QLineEdit, 
                             QTextEdit, QPushButton, QVBoxLayout)

from src.model_building import model_predict, FakeNewsDetector

# Prepare GUI
class MainWindow(QMainWindow):
    def __init__(self, model: FakeNewsDetector, config: dict, tokenizer_dir):
        super().__init__()
        
        # Model Details
        self.model = model
        self.model_config = config
        self.tokenizer_dir = tokenizer_dir
        
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
                    max_seq = self.model_config['max_seq'],
                    tokenizer_dir = self.tokenizer_dir
                )
                
                if is_fake:
                    self.predictionbox.setText('<h4>Output:</h4>The given news is likely fake news.')
                else:
                    self.predictionbox.setText('<h4>Output:</h4>The given news is likely authentic news.')
                
        except Exception as e:
            self.predictionbox.setText('<h4>Output:</h4>An unexpected error occurred.')
            print(f'ERROR: {e}')
         
        return
