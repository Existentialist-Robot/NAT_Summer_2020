from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QApplication, QMessageBox, QDialog, QDesktopWidget)
from PyQt5.QtCore import QTimer, Qt, QRect
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
import math
from classifier import LiveModel

class artScreen(QDialog):

    def artDialog(self, inputSize, artFeatures, band_q,art_q):

        ''' This is a window that will draw an art on the screen
    
        inputSize is an iterable (list, tuple) of the width and the height of the screen
        artFeatures is an interable (list, tuple) of ints indicating which art features assigned to each bandwidth (beta, alpha, theta, delta)
        q is a multiprocessing.Process.Queue object that stores the noise level and state of each bandwidth as dictionaries

        (optional) mood_q is a multiprocessing.Process.Queue object that stores numpy arrays for the probabilities of the stimulus being positive, negative, or neutral '''

        self.size = inputSize
        # initialize an array for the image -- x by y arrays of 3-item arrays for the HSV values of each pixel in the image
        self.imageArray = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        self.artFeatures = artFeatures

        # initialize noise levels and states of bandwidths as default values
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

        # for keeping tracking of the pulsing animation inbetween updates
        self.pulseIndex = 0 # the number of "frames" that the image has animated in one direction so far
        self.pulseMax = 8   # the max number of "frames" that the image should animate in one direction
        self.pulseChannel = random.randint(0,2) # randomly generate which channel should be animated for the next second
        self.pulseDir = 1   # increase (1) or decrease (-1) the channel value

        # Set timer for updating the image with the new data
        timer = QTimer(self)
        timer.timeout.connect(lambda: self.updateScreen(band_q=band_q,art_q=art_q, artFeatures=artFeatures))
        timer.start(1000)  # in milliseconds e.g. 1000 = 1 sec

        # set timer for creating the pulsing animation on the image
        pulseTimer = QTimer(self)
        pulseTimer.timeout.connect(lambda: self.pulseScreen())
        pulseTimer.start(100)
        
        
    def initUI(self):

        # initiating the UI layout
        self.hbox = QHBoxLayout(self)
        self.setWindowTitle('Art Screen')

        qtRectangle = QRect(0, 0, self.size[1], self.size[0])
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        
        pixmap = QPixmap()
        self.imageLabel = QLabel(self)
        self.imageLabel.setPixmap(pixmap)
        self.hbox.addWidget(self.imageLabel)
        self.setBaseSize(self.size[0],self.size[1])
        self.setLayout(self.hbox)
        self.raise_()
        self.show()

    def updateScreen(self,band_q,art_q,artFeatures=[0,1,2,3]):

        ''' Update the art screen

        artFeatures is an interable (list, tuple) of ints indicating which art features assigned to each bandwidth (beta, alpha, theta, delta)
        q is a multiprocessing.Process.Queue object that stores the noise level and state of each bandwidth as dictionaries

        (optional) mood_q is a multiprocessing.Process.Queue object that stores numpy arrays for the probabilities of the stimulus being positive, negative, or neutral
        '''
        
        self.imageArray = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)    # clear the image array first

        # only get new state_dict and noise_dict if q is not empty
        if not band_q.empty():
            state_dict,noise_dict = band_q.get()
            self.imageArray = circleArt(self.imageArray,noise_dict,state_dict,artFeatures) # create new image array from circleArt
        else:
            state_dict = self.state_dict
            noise_dict = self.noise_dict
            self.imageArray = circleArt(self.imageArray,noise_dict,state_dict,artFeatures) # create new image array from circleArt

        if state_dict != self.state_dict or noise_dict != self.noise_dict:  # only change the image if the noise levels and states of bandwidths have changed in any way
            self.state_dict = state_dict
            self.noise_dict = noise_dict
            self.imageArray = circleArt(self.imageArray,self.noise_dict,self.state_dict,artFeatures) # create new image array from circleArt

            if not art_q.empty():
                emotion_array = art_q.get() #shape [negative, neutral, positive] softmax output
                self.waveEffect(emotion_array)

            # convert the array into QPixmap and put it on the UI
            self.qim = ImageQt(Image.fromarray(self.imageArray, mode='RGB'))
            pix = QPixmap.fromImage(self.qim)
            self.imageLabel.setPixmap(pix)
            self.resize(pix.width(), pix.height())
            self.raise_()
            self.show()

            # change the channel to be pulsed for the next second
            self.pulseChannel = random.randint(0,2)

    def blendImage(self,moodArray):

        ''' set diferent blend modes on the image array depending on the probability of the mood being positive, negative, or neutral
        only run if mood_q is not empty
        '''

        # the image that will be blended into the existing image array
        blendLayer = np.zeros((self.size[0],self.size[1],4), dtype=np.uint8)

        if moodArray.max() > 0.6:   # only blend if we have a clear enough classification, i.e. probability above 0.6

            mood = np.where(moodArray == moodArray.max())   # find which mood had the largest probability

            if (mood[0] == 2)[0]:   # change the blend layer only if the mood is pos
                blendLayer.fill(255)

            blendLayer[:,:,3] = random.randint(100,200)   # randomly generate an alpha value for the blend layer

            newImage = np.zeros((size[0],size[1],4), dtype=np.uint8)
            newImage[:,:,0:3]= image
            newImage[:,:,3] = 255

            # alpha blending (https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending)
            self.imageArray = Image.alpha_composite(Image.fromarray(
                newImage, mode='RGBA'), Image.fromarray(blendLayer, mode='RGBA'))

    def pulseScreen(self):

        ''' create a pulsing animation by adding or subtracting 30 to any of the Red, Green, and Blue channels '''

        # add or subtract 30 to the pulse channel depending on self.pulseDir but limit the values to between 0 and 255
        self.imageArray[:, :, self.pulseChannel] = np.clip(self.imageArray[:, :, self.pulseChannel] + 30*self.pulseDir, 0, 255)
        self.pulseIndex += self.pulseDir    # update self.pulseIndex

        # if the pulse index reached 0 or the max
        if self.pulseIndex == self.pulseMax | self.pulseIndex == 0:
            self.pulseDir *= -1

        # convert the image into QPixmap and put it on the UI
        self.qim = ImageQt(Image.fromarray(self.imageArray, mode='RGB'))
        pix = QPixmap.fromImage(self.qim)
        self.imageLabel.setPixmap(pix)
        self.resize(pix.width(), pix.height())
        self.raise_()
        self.show()

    def waveEffect(self,moodArray):
        if moodArray.max() > 0.6:   # only blend if we have a clear enough classification, i.e. probability above 0.6

            mood = np.where(moodArray == moodArray.max())   # find which mood had the largest probability

            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
                    if i+offset_x < self.size[0]:
                        self.imageArray[i,j] = self.imageArray[(i+offset_x)%self.size[1],j]
                    else:
                        self.imageArray[i,j] = 0
    
    def closeEvent(self, event):
        #global run_process
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            run_process.terminate() 
            while run_process.is_alive() == True:
                time.sleep(0.1)
            
            print("run_process terminated")
            
            #if we close the window the spawned child processes terminate
             
            # running_model.terminate()
            # while running_model.is_alive() == True:
            #     time.sleep(0.1)
            # print("running_model terminated")
            event.accept()
            print('Window closed')
        else:
            event.ignore()

def spawned_stream_process(band_q,model_q):
    '''
    initialize stream object and run stream.run method
    this will be the target of the stream_process
    '''
    stream = Stream()
    stream.run(band_q,model_q)

def spawned_model_process(model_q,art_q):

    liveModel = LiveModel('data/Fred/model/cnn_time_dom.h5',model_q,art_q)
    liveModel.run()

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
    
    inputSize = size
    #initialize multiprocessing queue to allow data transfer between runningstream and artScreen
    band_q = Queue(5)
    #initialize queue for runningstream and classifer
    model_q = Queue(5)
    #initiealize queue for classifier and artScreen
    art_q = Queue(5)
    #initialize run_process for parallelism and make the variable global to allow 
    #exiting to make terminate the child process
    
    global run_process
    run_process = Process(target = spawned_stream_process, args = (band_q,model_q))
    run_process.start() 

    global running_model
    running_model = Process(target = spawned_model_process, args=(model_q,art_q))
    running_model.start()


    art = artScreen()
    art.setAttribute(Qt.WA_DeleteOnClose)
    art.artDialog(inputSize,artFeatures, band_q,model_q)




if __name__=='__main__':
    main()
