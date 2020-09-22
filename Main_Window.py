import sys
import os
import time
from multiprocessing import Process, Queue
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QLabel, QPushButton, QComboBox, QMessageBox, QRadioButton, QDialogButtonBox, QLineEdit, QSpacerItem
from PyQt5.QtWidgets import QInputDialog, QDialog, QDesktopWidget, QProgressBar
from PyQt5.QtGui import QFont, QPixmap, QIcon, QIntValidator
from PyQt5.QtCore import Qt
from Image_Manipulation.artScreen import launchArtScreen
import stroopy_words
import record
import send_art_stream

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

        self.screenResolution = QDesktopWidget().screenGeometry()
        self.width, self.height = self.screenResolution.width(), self.screenResolution.height()

        # ~~~Main window contents~~~

        # Create title label
        title = QLabel(appName)
        title.setFont(QFont('Arial', 20))
        title.setStyleSheet("QLabel { color: white}")
        title.setAlignment(Qt.AlignCenter)

        # Create buttons
        btn1 = QPushButton('Set Baseline')
        btn1.setStyleSheet("QPushButton { max-width: 15em}")

        btn2 = QPushButton("Start Datastream")
        btn2.setCheckable(True)
        btn2.setStyleSheet("QPushButton { max-width: 15em}")
        #btn2.setStyleSheet("QPushButton {max-width: 26; max-height: 26; border-radius: 13; border: 2px solid black; background-color: red;}")
        
        btn3 = QPushButton('Start')
        btn3.setStyleSheet("QPushButton { max-width: 15em}")
        btn3.setEnabled(False)
                
        btn4 = QPushButton('New User')
        btn4.setStyleSheet("QPushButton { max-width: 15em}")

        btn5 = QPushButton('Train Model')
        btn5.setStyleSheet("QPushButton { max-width: 15em}")


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

        def addListItems(self, list, width):
            for l in list:
                self.addItem(l)
            self.setFixedWidth(width)        
            

        # Create drop-downs
        artFeatures1 = QComboBox()
        #addFeatures(artFeatures1)
        addListItems(artFeatures1, features, 100)

        artFeatures2 = QComboBox()
        #addFeatures(artFeatures2)
        addListItems(artFeatures2, features, 100)

        artFeatures3 = QComboBox()
        #addFeatures(artFeatures3)
        addListItems(artFeatures3, features, 100)

        artFeatures4 = QComboBox()
        #addFeatures(artFeatures4)
        addListItems(artFeatures4, features, 100)

        userList = QComboBox()
        addListItems(userList, users, 100)

        def setFeatures():
            artFeatures1.setCurrentIndex(currentFeatures[0])
            artFeatures2.setCurrentIndex(currentFeatures[1])
            artFeatures3.setCurrentIndex(currentFeatures[2])
            artFeatures4.setCurrentIndex(currentFeatures[3])
        
        setFeatures()

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
            old = currentFeatures[index] #get old value
            change = currentFeatures.index(new) #find last position of 1
            currentFeatures[change] = old # change found position to old value in map
            currentFeatures[index] = new # change old value to new value in map
            print(currentFeatures)
            setFeatures()

        artFeatures1.currentIndexChanged.connect(lambda: changeFeatures(0)) #if 0 changed to 1
        artFeatures2.currentIndexChanged.connect(lambda: changeFeatures(1)) #if 0 changed to 1
        artFeatures3.currentIndexChanged.connect(lambda: changeFeatures(2)) #if 0 changed to 1
        artFeatures4.currentIndexChanged.connect(lambda: changeFeatures(3)) #if 0 changed to 1

        def updateUser():
            global users
            global path
            global currentUser
            user = userList.currentText()
            userDir = path + '/' + user
            if os.path.exists(userDir):
                currentUser = user
                print(currentUser)
            else:
                users.clear()
                populateUsers()
                userList.clear()
                addListItems(userList, users, 100)

        userList.currentTextChanged.connect(lambda: updateUser())

        blankSpace = QSpacerItem(40, 30)

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
        layout2b.addItem(blankSpace, 4, 0, 1, 1)
        layout2b.addItem(blankSpace, 4, 1, 1, 1)
        layout2b.addItem(blankSpace, 5, 0, 1, 1)
        layout2b.addItem(blankSpace, 5, 1, 1, 1)
        layout2b.addWidget(btn4, 6, 0, 1, 1)
        layout2b.addWidget(userList, 6, 1, 1, 1)

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

        onOff = QLabel(self)
        pixmap3 = QPixmap('Images\Off.png')
        #pixmap3 = QPixmap('Images\On.png')
        onOff.setPixmap(pixmap3)
        onOff.setAlignment(Qt.AlignCenter)

        layout2.addLayout(layout2b)
        layout2.addWidget(RBimg)

        layout3 = QHBoxLayout()
        layout3.addWidget(btn1)
        #layout3.addWidget(btn3)
        layout3.addWidget(onOff)
        layout3.addWidget(btn2)

        layout4 = QHBoxLayout()
        layout4.addWidget(btn5)
        layout4.addWidget(btn3)

        layout1.addWidget(logo)
        layout1.addWidget(title)

        layout0.addLayout(layout1)
        layout0.addLayout(layout2)
        layout0.addLayout(layout3)
        layout0.addLayout(layout4)
        #layout0.addWidget(btn5)


        self.widget = QWidget()
        self.widget.setLayout(layout0)
        self.setCentralWidget(self.widget)

        def toggle(button):
            if button.isChecked():
                onOff.setPixmap(QPixmap('Images\On.png'))
                button.setText("Stop Datastream")
                sendData()
            else:
                onOff.setPixmap(QPixmap('Images\Off.png'))
                button.setText("Start Datastream")
                stopData()

        def newUser(self):
            global currentUser
            new,ok = QInputDialog.getText(self, "Enter name", "Name")
            if ok:
                newDir = './data/' + new
                if not os.path.exists(newDir):
                    os.makedirs(newDir)
                    print(users)
                    users.append(new)
                    userList.clear()
                    userList.addItems(sorted(users))
                    currentUser = new
                    userList.setCurrentText(new)
                else:
                    error = QMessageBox.warning(self.widget, "User already exists!", "Please try a different name.")

        btn1.clicked.connect(self.RecordBaseline)
        btn3.clicked.connect(self.open_artScreen)
        btn2.clicked.connect(lambda: toggle(btn2))
        btn4.clicked.connect(lambda: newUser(self))
        btn5.clicked.connect(lambda: trainModel(self))
        #btn5.clicked.connect(self.startTraining)

        def trainModel(self):
            if btn3.isEnabled():
                btn3.setEnabled(False)    
            else:
                btn3.setEnabled(True)

    def startTraining(self):
        training = ProgressDialog(self)
        #training.accepted.connect(lambda: MainWindow.trainModel(self))
        training.trainDialog()
        #print(ok)

    def open_artScreen(self):
        global currentFeatures
        screen_resolution = [self.width, self.height]
        screenSize = RadioDialog(self)
        screenSize.sizeDialog(screen_resolution, currentFeatures)

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
            path = "data/" + currentUser + '/' + filename
            duration,ok = QInputDialog.getInt(self, "Enter recording duration", "Time (s):", default, minTime, maxTime, increment)
            if ok:
                q = Queue(5)
                stimulus = Process(target=stroopy_words.present, args=(q,duration))
                recording = Process(target=record.record, args=(duration, path))
                stimulus.start()
                recording.start()
                #while stimulus.is_alive():
                #    print(q.get())
                print(stimulus, stimulus.is_alive())
                print(recording, recording.is_alive())
                time.sleep(duration)
                while stimulus.is_alive() == True or recording.is_alive() == True :
                    time.sleep(0.5)
                    print("waiting")
                print("ok, finished")
                stimulus.close()
                recording.close()
                print("baseline task complete")
    


  
features = ["Red",
            "Green",
            "Blue",
            "Shape",
            ]


class RadioDialog(QDialog):
    def sizeDialog(self, size, features):
        self.size = size
        print(self.size)
        self.currentFeatures = features
        self.w, self.h = int(self.size[0]*0.5), int(self.size[1]*0.5) #default small dimensions
        self.custom = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Screen Size")
        self.setStyleSheet("QDialog { border-image: url(Images/Background.jpg) center center cover no-repeat; color: white }")

        self.validatorW = QIntValidator(50, self.size[0])
        self.validatorH = QIntValidator(50, self.size[1])
        self.customW = QLineEdit(self)
        self.customW.setDisabled(True)
        self.customW.setValidator(self.validatorW)
        self.labelW = QLabel('W:')
        self.labelW.setStyleSheet("QLabel { color: white}")
        self.customH = QLineEdit(self)
        self.customH.setDisabled(True)
        self.customH.setValidator(self.validatorH)
        self.labelH = QLabel('H:')
        self.labelH.setStyleSheet("QLabel { color: white}")
        self.choice1 = QRadioButton("Small")
        self.choice1.setStyleSheet("QRadioButton { color: white}")
        self.choice1.setChecked(True)
        self.choice2 = QRadioButton("Medium")
        self.choice2.setStyleSheet("QRadioButton { color: white}")
        self.choice3 = QRadioButton("Large")
        self.choice3.setStyleSheet("QRadioButton { color: white}")
        self.choice4 = QRadioButton("Custom")
        self.choice4.setStyleSheet("QRadioButton { color: white}")

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.layout = QVBoxLayout()
        self.layoutWH = QHBoxLayout()
        self.layoutWH.addWidget(self.labelW)
        self.layoutWH.addWidget(self.customW)
        self.layoutWH.addWidget(self.labelH)
        self.layoutWH.addWidget(self.customH)

        self.layout.addWidget(self.choice1)
        self.layout.addWidget(self.choice2)
        self.layout.addWidget(self.choice3)
        self.layout.addWidget(self.choice4)
        self.layout.addLayout(self.layoutWH)
        self.layout.addWidget(self.buttons)
        self.setLayout(self.layout)

        self.choice1.toggled.connect(lambda: self.btnState(self.choice1))
        self.choice2.toggled.connect(lambda: self.btnState(self.choice2))
        self.choice3.toggled.connect(lambda: self.btnState(self.choice3))
        self.choice4.toggled.connect(lambda: self.btnState(self.choice4))

        self.buttons.accepted.connect(self.acceptSize)
        self.buttons.rejected.connect(self.reject)

        self.exec_()

    def getSize(self):
        if self.custom:
            W = int(self.customW.text())
            H = int(self.customH.text())

            if W < 50:
                W = 50
            elif W > self.size[0]:
                W = int(self.size[0])
            else:
                W = int(self.customW.text())

            if H < 50:
                H = 50
            elif H > self.size[1]:
                H = int(self.size[1])
            else:
                H = int(self.customH.text())
        else:
            W = self.w
            H = self.h
        self.art_screen_size = [H, W]
        print(self.art_screen_size)
        launchArtScreen(self.art_screen_size, self.currentFeatures)

    def acceptSize(self):
        self.getSize()
        self.accept()

    def btnState(self, b):
        print(b.text())
        if b.text() == "Small":
            self.customH.setDisabled(True)
            self.customW.setDisabled(True)
            self.custom = False
            self.w, self.h = int(self.size[0]*0.5), int(self.size[1]*0.5)
        if b.text() == "Medium":
            self.customH.setDisabled(True)
            self.customW.setDisabled(True)
            self.custom = False
            self.w, self.h = int(self.size[0]*0.75), int(self.size[1]*0.75)
        if b.text() == "Large":
            self.customH.setDisabled(True)
            self.customW.setDisabled(True)
            self.custom = False
            self.w, self.h = int(self.size[0]*0.9), int(self.size[1]*0.9)
        if b.text() == "Custom":
            self.customH.setDisabled(False)
            self.customW.setDisabled(False)
            self.custom = True

class ProgressDialog(QDialog):
    def trainDialog(self):
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Training Model")
        self.setStyleSheet("QDialog { border-image: url(Images/Background.jpg) center center cover no-repeat; color: white }")
        self.label = QLabel("Please wait while the model is training.")
        self.label.setStyleSheet("QLabel { color: white}")
        self.progressBar = QProgressBar()
        self.progressBar.setGeometry(0, 0, 300, 25)
        self.progressBar.setMaximum(100)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.progressBar)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.buttons)
        self.setLayout(self.layout)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.exec_()

users = []
defaultUser = '(default)'
currentUser = defaultUser
path = './data/'


def populateUsers():
    if not os.path.exists(path + defaultUser):
        os.makedirs(path + defaultUser)
    contents = os.listdir(path)
    print(contents)
    for item in contents:
        if os.path.isdir(os.path.join(path, item)):
            users.append(item)
            print(users)
            
populateUsers()

currentFeatures = [0, 1, 2, 3]   # 0 = Red, 1 = Green, 2 = Blue, 3 = Shape

sending = 0
stimulus = 0
recording = 0
ready_baseline = 0

def sendData():
    global sending
    global ready_baseline
    sending = Process(target=send_art_stream.sendingData)
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