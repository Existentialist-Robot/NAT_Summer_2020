import sys
import os
import time
from multiprocessing import Process, Queue
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QLabel, QPushButton, QComboBox, QMessageBox, QRadioButton, QDialogButtonBox, QLineEdit, QSpacerItem
from PyQt5.QtWidgets import QInputDialog, QDialog, QDesktopWidget, QProgressBar
from PyQt5.QtGui import QFont, QPixmap, QIcon, QIntValidator
from PyQt5.QtCore import Qt, QTimer
from Image_Manipulation.artScreen import launchArtScreen
import stroopy_words
import record
import send_art_stream
import classifier


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
       
        appName = 'RemBRAINdt'

        # ~~~Main window setup~~~

        self.setWindowTitle(appName)
        self.resize(800, 400)
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
        btnFont = QFont('Arial', 11)
        btnStyle = ("QPushButton {"
                        "max-width: 15em;"
                        "color: black;"
                        "background-color: #f5eadf;"
                    "}")

        btn1 = QPushButton('Set Baseline')
        btn1.setFont(btnFont)
        btn1.setStyleSheet(btnStyle)

        btn2 = QPushButton("Start Datastream")
        btn2.setFont(btnFont)
        btn2.setStyleSheet(btnStyle)
        btn2.setCheckable(True) # allows button to be toggled on/off

        btn3 = QPushButton('Start')
        btn3.setFont(btnFont)
        btn3.setStyleSheet(btnStyle)
        btn3.setEnabled(False)  # button disabled until a trained model is available
                
        btn4 = QPushButton('New User')
        btn4.setFont(btnFont)
        btn4.setStyleSheet(btnStyle)

        btn5 = QPushButton('Train Model')
        btn5.setFont(btnFont)
        btn5.setStyleSheet(btnStyle)


        # Create brainwave labels
        labelFont = QFont('Times', 23)

        beta = QLabel('β   -   ')
        beta.setStyleSheet("QLabel { color: red}")
        beta.setFont(labelFont)

        alpha = QLabel('α   -   ')
        alpha.setStyleSheet("QLabel { color: green}")
        alpha.setFont(labelFont) 
        
        theta = QLabel('θ   -   ')
        theta.setStyleSheet("QLabel { color: blue}")
        theta.setFont(labelFont)
        
        delta = QLabel('δ   -   ')
        delta.setStyleSheet("QLabel { color: yellow}")
        delta.setFont(labelFont)

        # Create spacer
        blankSpace = QSpacerItem(40, 30)

        # Create drop-down lists
        featuresWidth = 100
        usersWidth = 100

        artFeatures1 = QComboBox()
        self.addListItems(artFeatures1, features, featuresWidth)

        artFeatures2 = QComboBox()
        self.addListItems(artFeatures2, features, featuresWidth)

        artFeatures3 = QComboBox()
        self.addListItems(artFeatures3, features, featuresWidth)

        artFeatures4 = QComboBox()
        self.addListItems(artFeatures4, features, featuresWidth)

        userList = QComboBox()
        self.addListItems(userList, users, usersWidth)

        # Set default list items
        def setFeatures():
            artFeatures1.setCurrentIndex(currentFeatures[0])
            artFeatures2.setCurrentIndex(currentFeatures[1])
            artFeatures3.setCurrentIndex(currentFeatures[2])
            artFeatures4.setCurrentIndex(currentFeatures[3])
        
        setFeatures()

        # Swap feature list items when changed to prevent duplicate feature assignments
        def changeFeatures(index):
            if index == 0:
                wave = artFeatures1
            if index == 1:
                wave = artFeatures2
            if index == 2:
                wave = artFeatures3
            if index == 3:
                wave = artFeatures4
            new = wave.currentIndex() # set new = '1'
            old = currentFeatures[index] # get old index value
            change = currentFeatures.index(new) # find last index position of '1'
            currentFeatures[change] = old # change found position to old index value in feature list
            currentFeatures[index] = new # change old index value to new value in feature list
            setFeatures()

        artFeatures1.currentIndexChanged.connect(lambda: changeFeatures(0))
        artFeatures2.currentIndexChanged.connect(lambda: changeFeatures(1))
        artFeatures3.currentIndexChanged.connect(lambda: changeFeatures(2))
        artFeatures4.currentIndexChanged.connect(lambda: changeFeatures(3))

        # Updates the dropdown list of user profiles
        def updateUser():
            global users
            global path
            global currentUser
            user = userList.currentText()
            userDir = path + '/' + user
            if os.path.exists(userDir):
                currentUser = user
            else:
                users.clear()
                populateUsers()
                userList.clear()
                self.addListItems(userList, users, 100)

        userList.currentTextChanged.connect(lambda: updateUser())


        # ~~~Main Window Layout~~~

        #                   __________________________
        #          {  layout1|   Logo    |  Title     |
        #          {       __|________________________|
        #          {         |           |            |
        #          {         |           |            |
        #  layout0 {  layout2| layout2b  |   Image    |
        #          {         |           |            |
        #          {       __|___________|____________|
        #          {  layout3|        buttons         |
        #          {         |________________________|
        #          {  layout4|        buttons         |
        #                    |________________________|
        
        # Create window layout and add components

        # Create images
        RBimg = QLabel(self)
        pixmap = QPixmap('Images\RemBRAINdt_Framed.png')    
        RBimg.setPixmap(pixmap)
        RBimg.setAlignment(Qt.AlignCenter)

        logo = QLabel(self)
        pixmap2 = QPixmap('Images\Icon.png')
        logo.setPixmap(pixmap2)
        logo.setAlignment(Qt.AlignCenter)

        onOff = QLabel(self)
        pixmap3 = QPixmap('Images\Off.png')
        onOff.setPixmap(pixmap3)
        onOff.setAlignment(Qt.AlignCenter)

        # Main layout wrapper
        layout0 = QVBoxLayout()
        layout0.setContentsMargins(50, 25, 50, 50)

        # Logo and title
        layout1 = QHBoxLayout()
        layout1.addWidget(logo)
        layout1.addWidget(title)

        # Dropdowns and RemBRAINdt image
        layout2 = QHBoxLayout()

        layout2b = QGridLayout()
        layout2b.addWidget(beta, 0, 0, 1, 1)    # (widget, row, col, row_span, col_span)
        layout2b.addWidget(alpha, 1, 0, 1, 1)
        layout2b.addWidget(theta, 2, 0, 1, 1)
        layout2b.addWidget(delta, 3, 0, 1, 1)
        layout2b.addWidget(artFeatures1, 0, 1, 1, 1)
        layout2b.addWidget(artFeatures2, 1, 1, 1, 1)
        layout2b.addWidget(artFeatures3, 2, 1, 1, 1)
        layout2b.addWidget(artFeatures4, 3, 1, 1, 1)
        layout2b.addItem(blankSpace, 4, 0, 1, 1)
        layout2b.addItem(blankSpace, 4, 1, 1, 1)
        layout2b.addItem(blankSpace, 5, 0, 1, 1)
        layout2b.addItem(blankSpace, 5, 1, 1, 1)
        layout2b.addWidget(btn4, 6, 0, 1, 1)
        layout2b.addWidget(userList, 6, 1, 1, 1)

        layout2b.setAlignment(Qt.AlignCenter)

        layout2.addLayout(layout2b)
        layout2.addWidget(RBimg)

        # First row of buttons
        layout3 = QHBoxLayout()
        layout3.addWidget(btn1)
        layout3.addWidget(onOff)
        layout3.addWidget(btn2)

        # Second row of buttons
        layout4 = QHBoxLayout()
        layout4.addWidget(btn5)
        layout4.addWidget(btn3)

        # Add all layouts to main layout wrapper
        layout0.addLayout(layout1)
        layout0.addLayout(layout2)
        layout0.addLayout(layout3)
        layout0.addLayout(layout4)

        # Set layout0 to the central widget
        self.widget = QWidget()
        self.widget.setLayout(layout0)
        self.setCentralWidget(self.widget)

        #Assign functions to buttons
        btn1.clicked.connect(self.RecordBaseline)
        btn3.clicked.connect(self.open_artScreen)
        btn2.clicked.connect(lambda: toggle(self, btn2))
        btn4.clicked.connect(lambda: newUser(self))
        btn5.clicked.connect(lambda: startTraining(self))       # placeholder - to be combined with trainclassifier
        #btn5.clicked.connect(lambda: trainclassifier(self))    # Initiates classifier training


        # Toggle the the Datastream button and indicator light
        def toggle(self, button):
            if button.isChecked():
                onOff.setPixmap(QPixmap('Images\On.png'))
                button.setText("Stop Datastream")
                sendData()
            else:
                onOff.setPixmap(QPixmap('Images\Off.png'))
                button.setText("Start Datastream")
                stopData()

        # Create new user profile
        def newUser(self):
            global currentUser
            new,ok = QInputDialog.getText(self, "Enter name", "Name")
            if ok:
                newDir = './data/' + new
                baselineDir = newDir + '/baseline'
                modelDir = newDir + '/model'
                if not os.path.exists(newDir):
                    os.makedirs(newDir)
                    os.makedirs(baselineDir)
                    os.makedirs(modelDir)
                    users.append(new)
                    userList.clear()
                    userList.addItems(sorted(users))
                    currentUser = new
                    userList.setCurrentText(new)
                else:
                    error = QMessageBox.warning(self.widget, "User already exists!", "Please try a different name.")

        # Initiates classifier training
        # ***To be connected to training progresss dialog
        def trainclassifier(self):
            global currentUser
            classifier.make_model(currentUser)

        # Loads training progress dialog, and enables Start button once trained
        # ***Needs to be connected to actual time estimate from classifier training.
        def startTraining(self):
            training = ProgressDialog(self)
            if training.exec_():
                print("OK")
                if not btn3.isEnabled():
                    btn3.setEnabled(True)    
            else:
                print("CANCEL")


    # Populates a dropdown from a list
    def addListItems(self, itemList, list, width):
        for l in list:
            itemList.addItem(l)
        itemList.setFixedWidth(width)

    # Launches the art screen via the screen size select dialog
    def open_artScreen(self):
        global currentFeatures
        screen_resolution = [self.width, self.height]
        screenSize = SizeRadioDialog(self)
        screenSize.sizeDialog(screen_resolution, currentFeatures)

    # Starts baseline recording
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
            path = "data/" + currentUser + '/baseline/' + filename

            duration,ok = QInputDialog.getInt(self, "Enter recording duration", "Time (s):", default, minTime, maxTime, increment)
            if ok:
                q = Queue(5)
                stimulus = Process(target=stroopy_words.present, args=(q,duration, self.width, self.height))  # present word list
                recording = Process(target=record.record, args=(duration, path))    # start recording
                stimulus.start()
                recording.start()
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

#~~~~Additional window classes~~~~

# Screen size select dialog window
class SizeRadioDialog(QDialog):
    def sizeDialog(self, size, features):
        self.size = size
        self.currentFeatures = features    # needed to pass to art screen
        self.w, self.h = int(self.size[0]*0.5), int(self.size[1]*0.5) #default small dimensions
        self.custom = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Screen Size")
        self.setStyleSheet("QDialog { border-image: url(Images/Background.jpg) center center cover no-repeat; color: white }")

        # custom inputs
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

        # size presets
        self.choice1 = QRadioButton("Small")
        self.choice1.setStyleSheet("QRadioButton { color: white}")
        self.choice1.setChecked(True)
        self.choice2 = QRadioButton("Medium")
        self.choice2.setStyleSheet("QRadioButton { color: white}")
        self.choice3 = QRadioButton("Large")
        self.choice3.setStyleSheet("QRadioButton { color: white}")
        self.choice4 = QRadioButton("Custom")
        self.choice4.setStyleSheet("QRadioButton { color: white}")

        # add OK and Cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        # Set layouts
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

        # Assign size function to radio buttons
        self.choice1.toggled.connect(lambda: self.btnState(self.choice1))
        self.choice2.toggled.connect(lambda: self.btnState(self.choice2))
        self.choice3.toggled.connect(lambda: self.btnState(self.choice3))
        self.choice4.toggled.connect(lambda: self.btnState(self.choice4))

        self.buttons.accepted.connect(self.acceptSize)
        self.buttons.rejected.connect(self.reject)

        self.exec_()

    # sets screen size based no button states
    def btnState(self, b):
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

    # calls getSize and closes Size Select dialog
    def acceptSize(self):
        self.getSize()
        self.accept()

    # Gets the chosen size and launches the art screen
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


# Progress bar dialog window
class ProgressDialog(QDialog):

    def __init__(self, *args, **kwargs):
        super(ProgressDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Training Model")
        self.setStyleSheet("QDialog { border-image: url(Images/Background.jpg) center center cover no-repeat; color: white }")
        self.label = QLabel("Please wait while the model is training.")
        self.label.setStyleSheet("QLabel { color: white}")

        self.progressBar = QProgressBar()
        self.progressBar.setGeometry(0, 0, 300, 25)
        self.progressBar.setMaximum(0)
        self.progressBar.setMinimum(0)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.progressBar)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.buttons)
        self.setLayout(self.layout)
        
        time = 3000
        self.timer = QTimer()
        self.timer.start(time)
        self.timer.timeout.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

# define user list and default users
users = []
defaultUser = '(default)'
currentUser = defaultUser
path = './data/'

# populate user list from existing directories
def populateUsers():
    if not os.path.exists(path + defaultUser):
        os.makedirs(path + defaultUser)
    contents = os.listdir(path)
    for item in contents:
        if os.path.isdir(os.path.join(path, item)):
            users.append(item)
            
populateUsers()

# define art features
features = ["Red",
            "Green",
            "Blue",
            "Shape",
            ]

# index feature order
currentFeatures = [0, 1, 2, 3]   # 0 = Red, 1 = Green, 2 = Blue, 3 = Shape

# define variables which will be used as processes
sending = 0
stimulus = 0
recording = 0

# switch to prevent baseline recording when datastream is off
ready_baseline = 0

# Start datastream and enable baseline recording
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
    
# Stop datastream and disable baseline recording    
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


# initiate application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())    