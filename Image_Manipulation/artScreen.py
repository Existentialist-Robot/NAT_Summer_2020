from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QApplication, QMessageBox, QDialog)
from PyQt5.QtCore import QTimer
from randomArt import *
#from randomArt import randomArt
from multiprocessing import Process, Queue
from running_stream import *
from PIL.ImageQt import ImageQt
import sys
import faulthandler
import pdb
from hsvArt import *



class artScreen(QDialog):

    """ This is a window that will draw an art on the screen """
    """ inputSize is a list or a tuple of the width and the height of the screen """
    """ App is the PyQt5.QtCore.QApplication object for the main app """

    def artDialog(self, inputSize,q):
        self.size = inputSize
        self.initUI()
        # Set timer for a set interval
        timer = QTimer(self)
        timer.timeout.connect(lambda: self.updateScreen(q))
        timer.start(300)  # in milliseconds e.g. 1000 = 1 sec

        
        

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
   
    def updateScreen(self,q):
        """ Update the art screen"""
        """ newImage is a PIL Image object of the new image to be displayed """
        # pdb.set_trace()
        
        
        artFeatures = [0,1,2,3] #artFeature copied from hsvArt.py main function

        if not q.empty():
            state_dict,noise_dict = q.get()
            print(state_dict)
            newImage = hsvArt(self.size, noise_dict, state_dict, artFeatures) #create newImage from hsvArt fun
        else:
            newImage = randomArt(self.size)

        
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

    # pdb.set_trace()
    # showScreen(inputSize)

    #initialize multiprocessing queue to allow data transfer between child process
    q = Queue(5)

    #initialize run_process for parallelism and make the variable global to allow 
    #exiting to make terminate the child process
    global run_process
    run_process = Process(target = spawned_process, args = (q,))

    
    run_process.start() 
    
    art = artScreen()
    art.artDialog(inputSize,q)

    sys.exit(qapp.exec_())
    
    

if __name__=='__main__':
    main()
    
