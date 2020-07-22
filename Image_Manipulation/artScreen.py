from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL.ImageQt import ImageQt
import ctypes
from randomArt import randomArt
import sys

class artScreen:

    """ This is a window that will draw an art on the screen """
    """ inputSize is a list or a tuple of the width and the height of the screen """
    """ App is the PyQt5.QtCore.QApplication object for the main app """

    def __init__(self,inputSize,App):
        self.size = inputSize
        self.win = QWidget()
        self.win.setWindowTitle('Art Screen')
        user32 = ctypes.windll.user32
        screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        # self.scene = QtWidgets.QGraphicsScene(screensize[0]//2,screensize[1]//2,inputSize[0],inputSize[1])
        # self.view = QtWidgets.QGraphicsView(self.scene)
        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(QPixmap())
        self.win.show()
   
    def updateScreen(self,newImage):
        """ Update the art screen"""
        """ newImage is a PIL Image object of the new image to be displayed """
        qim = ImageQt(newImage)
        pix = QPixmap.fromImage(qim)
        self.imageLabel.setPixmap(pix)
        self.win.show()

    def clearScreen(self):
        """clear the screen"""
        self.imageLabel.setPixmap(QPixmap())
        self.win.show()

if __name__ == '__main__':
    screenSize = input(
        'Type in the size of art screen as two numbers with a comma inbetween (e.g. 1920,1060): ')
    screenSize = screenSize.split(',')
    screenSize[0] = int(screenSize[0])
    screenSize[1] = int(screenSize[1])
    myApp = QApplication(sys.argv)
    myScreen = artScreen(screenSize,myApp)
    art = randomArt(screenSize)
    myScreen.updateScreen(art)
    sys.exit(myApp.exec_())
