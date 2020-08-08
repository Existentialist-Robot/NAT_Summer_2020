from artScreen import artScreen
from PyQt5.QtWidgets import QApplication
import sys
from time import sleep
import pdb

def showScreen(inputSize):
    myscreen = artScreen(inputSize)
    myscreen.show()

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
    pdb.set_trace()
    # showScreen(inputSize)
    myscreen = artScreen(inputSize)
    sys.exit(qapp.exec_())

if __name__=='__main__':
    main()
