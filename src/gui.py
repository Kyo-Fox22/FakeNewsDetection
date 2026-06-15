from PyQt6.QtWidgets import (QApplication,  QMainWindow, QWidget, QLabel, 
                             QLineEdit, QTextEdit, QPushButton, QVBoxLayout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
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
        self.predictbutton.clicked.connect(self.placeholder_func)
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
        
    def placeholder_func(self):
        author = self.authbox.text()
        content = self.contentbox.toPlainText()
        print(f'{author}: {content}')
        return
    
        
app = QApplication([])

window = MainWindow()
window.show()

app.exec()