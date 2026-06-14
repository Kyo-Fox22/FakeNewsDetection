from PyQt6.QtWidgets import QApplication,  QMainWindow, QWidget, QLabel, QTextEdit, QVBoxLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('Fake News Detector App')
            
        # Create Layout
        layout = QVBoxLayout()
        
        textlabel = QLabel('<h1>Enter Author: </h1>')
        textbox = QTextEdit()
        
        layout.addWidget(textlabel)
        layout.addWidget(textbox)
        
        # Create Window
        window = QWidget()
        window.setLayout(layout)
        self.setCentralWidget(window)
    
        
app = QApplication([])

window = MainWindow()
window.show()

app.exec()