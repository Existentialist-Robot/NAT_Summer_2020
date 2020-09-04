from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QApplication, QMessageBox, QDialog)
from PyQt5.QtCore import QTimer
from PIL.ImageQt import ImageQt
import sys
import numpy as np
import pdb
from circleArt import circleArt

class artScreen(QDialog):

    """ This is a window that will draw an art on the screen """
    """ inputSize is a list or a tuple of the width and the height of the screen """
    """ App is the PyQt5.QtCore.QApplication object for the main app """

    def artDialog(self, inputSize):

        # initialize an array for the image -- x by y arrays of 3-item arrays for the HSV values of each pixel in the image
        self.imageArray = np.zeros((inputSize[0],inputSize[1],3),dtype=np.uint8)
        
        # set timer for updating the whole image
        updateTimer = QTimer(self)
        updateTimer.timeout.connect(lambda: self.updateScreen())
        updateTimer.start(1000)  # in milliseconds e.g. 1000 = 1 sec

        # set timer for creating a pulsating illusion on the image
        pulseTimer = QTimer(self)
        pulseTimer.timeout.connect(lambda: self.updateScreen(pulse=True))
        pulseTimer.start(100)

        self.size = inputSize
        self.initUI()

    def initUI(self):
        self.hbox = QHBoxLayout(self)
        self.setWindowTitle('Art Screen')

        pixmap = QPixmap()

        self.imageLabel = QLabel(self)
        self.imageLabel.setPixmap(pixmap)

        self.hbox.addWidget(self.imageLabel)
        self.setBaseSize(self.size[0],self.size[1])
        self.setLayout(self.hbox)
        
        self.raise_()
        self.show()
   
    def updateScreen(self,pulse=False):

        ''' Update the art screen

        funcName is the art function to be implemented. hsvArt produces a new image, and pulseArt creates a pulsing effect
        '''

        # pdb.set_trace()
        newImage = circleArt(self.imageArray,pulse=pulse)
        # newImage = randomArt(self.size)
        self.qim = ImageQt(newImage)
        pix = QPixmap.fromImage(self.qim)
        self.imageLabel.setPixmap(pix)
        self.resize(pix.width(), pix.height())
        self.raise_()
        self.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
            print('Window closed')
        else:
            event.ignore()

def main():
    if QApplication.instance():
        qapp = QApplication.instance()
    else:
        qapp = QApplication(sys.argv)
    inputSize = input(
        'Type in the size of art screen as two numbers with a comma inbetween (e.g. 1920,1060): ')
    inputSize = inputSize.split(',')
    inputSize[0] = int(inputSize[0])
    inputSize[1] = int(inputSize[1])
    # pdb.set_trace()
    # showScreen(inputSize)
    art = artScreen()
    art.artDialog(inputSize)
    sys.exit(qapp.exec_())

if __name__=='__main__':
    main()
