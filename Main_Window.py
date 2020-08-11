import sys
import time

from multiprocessing import Process, cpu_count
from multiprocessing import Pool
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QLabel, QPushButton, QComboBox, QMessageBox
from PyQt5.QtWidgets import QInputDialog, QDialog
from PyQt5.QtGui import QFont, QPixmap, QIcon, QImage, QPalette, QBrush, QPainter
from PyQt5.QtCore import Qt, QSize
from Image_Manipulation.artScreen import artScreen
#from Image_Manipulation.showArtScreen import showScreen
from Image_Manipulation.randomArt import randomArt
import stroopy_words
import record
import send

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
       
        appName = 'RemBRAINdt'

        # Main window setup
        self.setWindowTitle(appName)
        self.resize(800, 400)
        #self.setStyleSheet("QMainWindow { background: url(Images/Background.jpg) center center cover no-repeat fixed; color: white }")
        self.setStyleSheet("QMainWindow { border-image: url(Images/Background.jpg) center center cover no-repeat; color: white }")
        self.setWindowIcon(QIcon('Images\Icon.png'))

        #background-color: black;

        #painter = QPainter(self)
        #painter.drawRect(self.rect())
        #background = QPixmap('Images\Background.jpg')
        #painter.drawPixmap(self.rect(), background)
        #sbackground = background.scaled(QSize(800, 400))
        #palette = QPalette()
        #palette.setBrush(10, QBrush(background))
        #self.setPalette(palette)

        # ~~~Main window contents~~~

        # Create title label
        title = QLabel(appName)
        title.setFont(QFont('Arial', 20))
        title.setStyleSheet("QLabel { color: white}")
        title.setAlignment(Qt.AlignCenter)

        # Create buttons
        btn1 = QPushButton('Set Baseline')
        btn1.setStyleSheet("QPushButton { max-width: 15em}")

        btn2 = QPushButton('Start')
        btn2.setStyleSheet("QPushButton { max-width: 15em}")

        btn3 = QPushButton()
        btn3.setCheckable(True)
        btn3.setStyleSheet("QPushButton {max-width: 26; max-height: 26; border-radius: 13; border: 2px solid black; background-color: red;}")
        
        def toggle(button):
            if button.isChecked():
                button.setStyleSheet("QPushButton"
                                "{"
                                "max-height : 26; max-width : 26;"
                                "border-radius : 13;  border : 1px solid black;"
                                "background-color : green;"
                                "}"
                                ) 
                sendData()
            else:
                button.setStyleSheet("QPushButton"
                                "{"
                                "max-height : 26; max-width : 26;"
                                "border-radius : 13;  border : 1px solid black;"
                                "background-color : red;"
                                "}"
                                ) 
                stopData()

        # Create brainwave labels
        beta = QLabel('β   -   ')
        beta.setStyleSheet("QLabel { color: white}")
        beta.setFont(QFont('Times', 23))
        alpha = QLabel('α   -   ')
        alpha.setStyleSheet("QLabel { color: white}")
        alpha.setFont(QFont('Times', 23)) 
        theta = QLabel('θ   -   ')
        theta.setStyleSheet("QLabel { color: white}")
        theta.setFont(QFont('Times', 23))
        delta = QLabel('δ   -   ')
        delta.setStyleSheet("QLabel { color: white}")
        delta.setFont(QFont('Times', 23))


        def addFeatures(self):
            for f in features:
                self.addItem(f)
            self.setFixedWidth(100)

        # Create drop-downs
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

        # Create window layout and add components

        #                   __________________________
        #          {  layout1|   Logo    |  Title     |
        #          {       __|________________________|
        #          {         |           |            |
        #          {         |           |            |
        #  layout0 {  layout2| layout2b  |   Image    |
        #          {         |           |            |
        #          {       __|___________|____________|
        #          {         |                        |
        #          {  layout3|        buttons         |
        #                    |________________________|
        

        layout0 = QVBoxLayout()
        layout0.setContentsMargins(50, 25, 50, 50)

        layout2 = QHBoxLayout()

        layout2b = QGridLayout()
        layout2b.addWidget(beta, 0, 0, 1, 1)
        layout2b.addWidget(alpha, 1, 0, 1, 1)
        layout2b.addWidget(theta, 2, 0, 1, 1)
        layout2b.addWidget(delta, 3, 0, 1, 1)
        layout2b.addWidget(artFeatures1, 0, 1, 1, 2)
        layout2b.addWidget(artFeatures2, 1, 1, 1, 2)
        layout2b.addWidget(artFeatures3, 2, 1, 1, 2)
        layout2b.addWidget(artFeatures4, 3, 1, 1, 2)

        layout2b.setAlignment(Qt.AlignCenter)
    
        layout1 = QHBoxLayout()
        RBimg = QLabel(self)
        pixmap = QPixmap('Images\RemBRAINdt_Framed.png')
        #pixmap = QPixmap('Images\RemBRAINdt.jpg')
        RBimg.setPixmap(pixmap)
        RBimg.setAlignment(Qt.AlignCenter)

        logo = QLabel(self)
        pixmap2 = QPixmap('Images\Icon.png')
        logo.setPixmap(pixmap2)
        logo.setAlignment(Qt.AlignCenter)

        layout2.addLayout(layout2b)
        layout2.addWidget(RBimg)

        layout3 = QHBoxLayout()
        layout3.addWidget(btn1)
        layout3.addWidget(btn3)
        layout3.addWidget(btn2)

        layout1.addWidget(logo)
        layout1.addWidget(title)

        layout0.addLayout(layout1)
        layout0.addLayout(layout2)
        layout0.addLayout(layout3)

        self.widget = QWidget()
        self.widget.setLayout(layout0)
        self.setCentralWidget(self.widget)

        
        btn1.clicked.connect(self.RecordBaseline)
        btn2.clicked.connect(self.open_artScreen)
        btn3.clicked.connect(lambda: toggle(btn3))


    def open_artScreen(self):
        art_screen_size = [526,526]
        art = artScreen()
        art.artDialog(art_screen_size)
        art.exec_()
        

    def RecordBaseline(self):
        global stimulus
        global recording
        global ready_baseline
        if ready_baseline == 0:
            error = QMessageBox.warning(self.widget, "No datastream!", "Please start the datastream first.")
        else:
            default = 120
            minTime = 5
            maxTime = 300
            increment = 5

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = timestamp + "_csv.txt"
            path = "data\\" + filename
            duration,ok = QInputDialog.getInt(self, "Enter recording duration", "Time (s):", default, minTime, maxTime, increment)
            if ok:
                stimulus = Process(target=stroopy_words.present, args=(duration,))
                recording = Process(target=record.record, args=(duration, path))
                stimulus.start()
                recording.start()
                print(stimulus, stimulus.is_alive())
                print(recording, recording.is_alive())
                time.sleep(duration)
                while stimulus.is_alive() == True or recording.is_alive() == True :
                    time.sleep(0.5)
                    print("waiting")
                print("ok, finished")
                #stimulus.close()
                #recording.close()
                print("baseline task complete")

  
features = ["Feature 1",
            "Feature 2",
            "Feature 3",
            "Feature 4",
            ]

sending = 0
stimulus = 0
recording = 0
ready_baseline = 0

def sendData():
    global sending
    global ready_baseline
    sending = Process(target=send.sendingData)
    sending.daemon = True
    print(sending, sending.is_alive())
    sending.start()
    print('sending started')
    print(sending, sending.is_alive())
    ready_baseline = 1
    
def stopData():
    global sending
    global ready_baseline
    print(sending, sending.is_alive())
    sending.terminate()
    while sending.is_alive() == True:
        time.sleep(0.1)
    sending.close()
    print("sending terminated")
    ready_baseline = 0


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())    