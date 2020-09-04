from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QApplication, QMessageBox, QDialog)

from PyQt5.QtCore import QTimer, Qt
from multiprocessing import Process, Queue
from running_stream import *

from PIL.ImageQt import ImageQt
import sys
import numpy as np
import pdb
from circleArt import circleArt

class artScreen(QDialog):

    """ This is a window that will draw an art on the screen """
    """ inputSize is a list or a tuple of the width and the height of the screen """
    """ App is the PyQt5.QtCore.QApplication object for the main app """

    def artDialog(self, inputSize, artFeatures, q):
        
        # initialize an array for the image -- x by y arrays of 3-item arrays for the HSV values of each pixel in the image
        self.imageArray = np.zeros((inputSize[0],inputSize[1],3),dtype=np.uint8)

        self.size = inputSize
        self.artFeatures = artFeatures
        self.initUI()
        # Set timer for a set interval
        timer = QTimer(self)
        timer.timeout.connect(lambda: self.updateScreen(q))
        timer.start(1000)  # in milliseconds e.g. 1000 = 1 sec

            # set timer for creating a pulsating illusion on the image
        pulseTimer = QTimer(self)
        pulseTimer.timeout.connect(lambda: self.updateScreen(pulse=True))
        pulseTimer.start(100)
        
        
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

    def updateScreen(self,q,pulse=True):

        ''' Update the art screen

        funcName is the art function to be implemented. hsvArt produces a new image, and pulseArt creates a pulsing effect
        '''
        if not q.empty():
            state_dict,noise_dict = q.get()
            print(state_dict)
            newImage = hsvArt(self.size, noise_dict, state_dict, self.artFeatures) #create newImage from hsvArt fun
        else:
            newImage = circleArt(self.imageArray,pulse=pulse)

        self.qim = ImageQt(newImage)
        pix = QPixmap.fromImage(self.qim)
        self.imageLabel.setPixmap(pix)
        self.resize(pix.width(), pix.height())
        self.raise_()
        self.show()


    def closeEvent(self, event):
        #global run_process
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            run_process.terminate() #if we close the window the spawned child process terminates
            while run_process.is_alive() == True:
                time.sleep(0.1)
            run_process.close()
            print("run_process terminated")
            event.accept()
            run_process.terminate() #if we close the window the spawned child process terminates
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
    launchArtScreen(inputSize)
    # pdb.set_trace()
    # showScreen(inputSize)
    sys.exit(qapp.exec_())
    
    

def launchArtScreen(size, artFeatures):    
    #initialize multiprocessing queue to allow data transfer between child process
    inputSize = size
    #initialize run_process for parallelism and make the variable global to allow 
    #exiting to make terminate the child process
    global run_process
    run_process = Process(target = spawned_process, args = (q,))
    run_process.start() 

    art = artScreen()
    art.setAttribute(Qt.WA_DeleteOnClose)
    art.artDialog(inputSize,artFeatures, q)

q = Queue(5)


if __name__=='__main__':
    main()
