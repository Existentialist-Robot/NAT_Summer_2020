from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QApplication, QMessageBox, QDialog)
from PyQt5.QtCore import QTimer
from randomArt import *
#from randomArt import randomArt
from multiprocessing import Process,Manager,set_start_method,Pool,Queue
from running_stream import *
from PIL.ImageQt import ImageQt
import sys
import faulthandler
import pdb
from hsvArt import *

set_start_method('spawn', force=True) #needed to run on UNIX 

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
        #newImage = randomArt(self.size)
        
        print(q.get())
        artFeatures = [0,1,2,3] #artFeature copied from hsvArt.py main function

        #newImage = hsvArt(self.size, shared_noise_dict, shared_state_dict, artFeatures) #create newImage from hsvArt func

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

    #initialize shared dictionaries for hsvArt and stream.run to use
    
    '''
    shared_noise_dict = manager.dict()
    shared_noise_dict['Delta']= False
    shared_noise_dict['Theta']= False
    shared_noise_dict['Alpha']=False 
    shared_noise_dict['Beta']= False
    

    
    shared_state_dict = manager.dict()
    shared_state_dict['Delta'] = 'low'
    shared_state_dict['Theta'] = 'low'                
    shared_state_dict['Alpha'] = 'low'
    shared_state_dict['Beta'] = 'low'
    shared_state_dict['Random'] = False
    '''
    stream = Stream()
    q = Queue(5)

    #run_process = Process(target = stream.run)
    #run_process.start() 
    pool = Pool(processes = 4)
    pool.apply_async(stream.run, args = (q,))



    
    
    art = artScreen()
    art.artDialog(inputSize,q)
    #art_process = Process(target = art.artDialog, args = (inputSize,shared_noise_dict,shared_state_dict))
    #art_process.start()
    sys.exit(qapp.exec_())

    pool.close()
    #run_process.join()
if __name__=='__main__':
    main()
    
