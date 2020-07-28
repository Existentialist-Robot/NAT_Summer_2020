import sys

from multiprocessing import Process
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from Image_Manipulation.artScreen import artScreen
from Image_Manipulation.randomArt import randomArt
import stroopy_words
import record

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
       
        appName = 'RemBRAINdt'

        self.setWindowTitle(appName)
        self.resize(800, 400)
        self.setStyleSheet("QMainWindow { background-color: black; color: white }")

       
        label = QLabel(appName)
        label.setFont(QFont('Arial', 20))
        label.setStyleSheet("QLabel { color: white}")
        label.setAlignment(Qt.AlignCenter)

        btn1 = QPushButton('Set Baseline')
        btn2 = QPushButton('Start')
        btn1.setStyleSheet("QPushButton { max-width: 15em}")
        btn2.setStyleSheet("QPushButton { max-width: 15em}")

        beta = QLabel('β')
        beta.setStyleSheet("QLabel { color: white}")
        beta.setFont(QFont('Arial', 22))
        alpha = QLabel('α')
        alpha.setStyleSheet("QLabel { color: white}")
        alpha.setFont(QFont('Arial', 22))
        theta = QLabel('θ')
        theta.setStyleSheet("QLabel { color: white}")
        theta.setFont(QFont('Arial', 22))
        delta = QLabel('δ')
        delta.setStyleSheet("QLabel { color: white}")
        delta.setFont(QFont('Arial', 22))



        def addFeatures(self):
            for f in features:
                self.addItem(f)

        artFeatures1 = QComboBox()
        addFeatures(artFeatures1)

        artFeatures2 = QComboBox()
        addFeatures(artFeatures2)

        artFeatures3 = QComboBox()
        addFeatures(artFeatures3)

        artFeatures4 = QComboBox()
        addFeatures(artFeatures4)

        currentStates = [0, 1, 2, 3]
        print(currentStates)

        def setStates():
            artFeatures1.setCurrentIndex(currentStates[0])
            artFeatures2.setCurrentIndex(currentStates[1])
            artFeatures3.setCurrentIndex(currentStates[2])
            artFeatures4.setCurrentIndex(currentStates[3])
        
        setStates()

        def changeFeatures(index):
            if index == 0:
                wave = artFeatures1
            if index == 1:
                wave = artFeatures2
            if index == 2:
                wave = artFeatures3
            if index == 3:
                wave = artFeatures4
            new = wave.currentIndex() # set new = 1
            old = currentStates[index] #get old value
            change = currentStates.index(new) #find last position of 1
            currentStates[change] = old # change found position to old value in map
            currentStates[index] = new # change old value to new value in map
            print(currentStates)
            setStates()

        artFeatures1.currentIndexChanged.connect(lambda: changeFeatures(0)) #if 0 changed to 1
        artFeatures2.currentIndexChanged.connect(lambda: changeFeatures(1)) #if 0 changed to 1
        artFeatures3.currentIndexChanged.connect(lambda: changeFeatures(2)) #if 0 changed to 1
        artFeatures4.currentIndexChanged.connect(lambda: changeFeatures(3)) #if 0 changed to 1


        layout1 = QVBoxLayout()

        layout2 = QHBoxLayout()

        layout2b = QGridLayout()
        layout2b.addWidget(beta, 0, 0)
        layout2b.addWidget(alpha, 1, 0)
        layout2b.addWidget(theta, 2, 0)
        layout2b.addWidget(delta, 3, 0)
        layout2b.addWidget(artFeatures1, 0, 1)
        layout2b.addWidget(artFeatures2, 1, 1)
        layout2b.addWidget(artFeatures3, 2, 1)
        layout2b.addWidget(artFeatures4, 3, 1)
    
        layout2.addLayout(layout2b)

        layout3 = QHBoxLayout()
        layout3.addWidget(btn1)
        layout3.addWidget(btn2)

        layout1.addWidget(label)
        layout1.addLayout(layout2)
        layout1.addLayout(layout3)

        self.widget = QWidget()
        self.widget.setLayout(layout1)
        self.setCentralWidget(self.widget)
        
        btn1.clicked.connect(self.RecordBaseline)
        btn2.clicked.connect(self.open_artScreen)


    def open_artScreen(self):
        art_screen_size = [526,526]
        screen = artScreen(art_screen_size)
        screen.updateScreen(randomArt(art_screen_size))

    def RecordBaseline(self):
        default = 120
        minTime = 5
        maxTime = 300
        increment = 5
        time,ok = QInputDialog.getInt(self, "Enter recording duration", "Time (s):", default, minTime, maxTime, increment)
        if ok:
            stimulus = Process(target=stroopy_words.present, args=(time,))
            recording = Process(target=record.record, args=(time, 'data\csvtest.txt'))
            stimulus.start()
            recording.start()

  
features = ["Feature 1",
            "Feature 2",
            "Feature 3",
            "Feature 4",
            ]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())