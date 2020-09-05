from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QApplication, QMessageBox, QDialog)
from PyQt5.QtCore import QTimer, Qt
from multiprocessing import Process, Queue
from running_stream import *
from PIL.ImageQt import ImageQt
from PIL import Image
import sys
import numpy as np
import pdb
# from circleArt import circleArt
from Image_Manipulation.circleArt import circleArt
import random
import time

class artScreen(QDialog):

    """ This is a window that will draw an art on the screen """
    """ inputSize is a list or a tuple of the width and the height of the screen """
    """ App is the PyQt5.QtCore.QApplication object for the main app """

    def artDialog(self, inputSize, artFeatures, q):
        
        self.size = inputSize
        # initialize an array for the image -- x by y arrays of 3-item arrays for the HSV values of each pixel in the image
        self.imageArray = np.zeros(
            (self.size[0], self.size[1], 3), dtype=np.uint8)
        self.artFeatures = artFeatures

        self.noise_dict = {
            'Delta': False,
            'Theta': False,
            'Alpha': False,
            'Beta': False
            }
        
        self.state_dict = {
            'Delta': 'Low',
            'Theta': 'Low',
            'Alpha': 'Low',
            'Beta': 'Low'
            }

        self.initUI()

        self.pulseIndex = 0
        self.pulseMax = 8
        self.pulseChannel = random.randint(0,2)
        self.pulseDir = 1
        self.dicts = []

        # self.updateScreen(artFeatures,q)

        # Set timer for a set interval
        timer = QTimer(self)
        timer.timeout.connect(lambda: self.updateScreen(artFeatures,q))
        timer.start(1000)  # in milliseconds e.g. 1000 = 1 sec

        # set timer for creating a pulsating illusion on the image
        pulseTimer = QTimer(self)
        pulseTimer.timeout.connect(lambda: self.pulseScreen())
        pulseTimer.start(80)
        
        
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

    def updateScreen(self,artFeatures,q):

        ''' Update the art screen

        funcName is the art function to be implemented. hsvArt produces a new image, and pulseArt creates a pulsing effect
        '''
        self.imageArray = np.zeros(
            (self.size[0], self.size[1], 3), dtype=np.uint8)    # clear the screen first
        
        if not q.empty():
            state_dict,noise_dict = q.get()
            if state_dict != self.state_dict or noise_dict != self.noise_dict:
                self.state_dict = state_dict
                self.noise_dict = noise_dict
                self.imageArray = circleArt(self.imageArray,self.noise_dict,self.state_dict,artFeatures) #create newImage from hsvArt fun
            else:
                self.imageArray = circleArt(self.imageArray,self.noise_dict,self.state_dict,artFeatures)
        else:
            self.imageArray = circleArt(self.imageArray,self.noise_dict,self.state_dict,artFeatures)
            # newImage = Image.fromarray(self.imageArray,mode='RGB')

        self.qim = ImageQt(Image.fromarray(self.imageArray, mode='RGB'))
        pix = QPixmap.fromImage(self.qim)
        self.imageLabel.setPixmap(pix)
        self.resize(pix.width(), pix.height())
        self.raise_()
        self.show()

        self.pulseChannel = random.randint(0,2)

    def pulseScreen(self):

        # pdb.set_trace()

        print(self.pulseChannel)

        # pdb.set_trace()
        self.imageArray[:, :, self.pulseChannel] = np.clip(
            self.imageArray[:, :, self.pulseChannel] + 30*self.pulseDir, 0, 255)
        self.pulseIndex += self.pulseDir

        if self.pulseIndex == self.pulseMax:
            self.pulseDir *= -1
        elif self.pulseIndex == 0:
            self.pulseDir *= -1

        self.qim = ImageQt(Image.fromarray(self.imageArray, mode='RGB'))
        pix = QPixmap.fromImage(self.qim)
        self.imageLabel.setPixmap(pix)
        self.resize(pix.width(), pix.height())
        self.raise_()
        self.show()

        # pdb.set_trace()

    def closeEvent(self, event):
        #global run_process
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            run_process.terminate() #if we close the window the spawned child process terminates
            while run_process.is_alive() == True:
                time.sleep(0.1)
            # run_process.close()
            print("run_process terminated")
            event.accept()
            print('Window closed')
        else:
            event.ignore()

def spawned_process(q):
    '''
    initialize stream object and run stream.run method
    this will be the target of the stream_process
    '''
    stream = Stream()
    stream.run(q)

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
    artFeatures = [0,1,2,3]
    launchArtScreen(inputSize,artFeatures)
    # pdb.set_trace()
    # showScreen(inputSize)
    sys.exit(qapp.exec_())

def launchArtScreen(size, artFeatures):    
    #initialize multiprocessing queue to allow data transfer between child process
    #initialize run_process for parallelism and make the variable global to allow 
    #exiting to make terminate the child process
    global run_process
    run_process = Process(target = spawned_process, args = (q,))
    run_process.start() 

    art = artScreen()
    art.setAttribute(Qt.WA_DeleteOnClose)
    art.artDialog(size,artFeatures, q)

q = Queue(5)


if __name__=='__main__':
    main()

