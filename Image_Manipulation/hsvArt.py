import numpy as np
from PIL import Image
import random
import pdb

# TODO: Incorporate recursion into the design

random.seed = 1

def safeDivide(a, b):
    return np.divide(a, np.maximum(b, 0.001))

def noFunc():
    return

# def applyFunctions(inputArray,bias):

#     ''' appy different functions to a given array, with the consideration of the bias

#     the possible functions that can be applied are: addition, subtraction, multiplication, division, sin, cos, tan, and modulo
#     '''

#     funcs = [(1, noFunc),
#             (1, np.sin),
#             (1, np.cos),
#             (1, np.tan),
#             (2, np.add),
#             (2, np.subtract),
#             (2, np.multiply),
#             (2, np.mod),
#             (2, safeDivide)]


def changeHue(inputArray,freqState,artFeature):

    ''' manipulate the hue levels of the image

    inputArray is a numpy array storing the hue value of the imate
    freqState is a dict, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data
    artFeatures is a dict, storing the art features (hue, saturation, value, line quality) assigned to each bandwidth of brain wave
    '''

    bias = 0
    if freqState == 'high': # if the frequency band has a high amplitude, push the colours toward red
        bias = -50

    # draw values for hue by randomly sampling from a normal distribution with mean and standard deviation that are randomly generated but biased depending on the amplitude level
    outputArray = np.random.normal(loc=random.randint(0,360)+bias,scale=360*0.05+bias*0.05,size=inputArray.shape)

    # reset values that are beyond boundaries
    outputArray[np.where(outputArray > 360)] = 360
    outputArray[np.where(outputArray < 0)] = 0

    return outputArray


def changeSaturation(inputArray,freqState,artFeature):

    ''' manipulate the saturation levels of the image

    inputArray is a numpy array storing the hue value of the imate
    freqState is a dict, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data
    artFeatures is a dict, storing the art features (hue, saturation, value, line quality) assigned to each bandwidth of brain wave
    '''

    bias = 0
    if freqState == 'high': # if the frequency band has a high amplitude, output higher saturation
        bias = 20

    # draw values for hue by randomly sampling from a normal distribution with mean and standard deviation that are randomly generated but biased depending on the amplitude level
    outputArray = np.random.normal(loc=random.randint(0,100)+bias,scale=100*0.05-bias*0.05,size=inputArray.shape)
    
    # reset values that are beyond boundaries
    outputArray[np.where(outputArray > 100)] = 100
    outputArray[np.where(outputArray < 0)] = 0

    return outputArray

def changeValue(inputArray,freqState,artFeature):

    ''' manipulate the value levels of the image

    inputArray is a numpy array storing the hue value of the imate
    freqState is a dict, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data
    artFeatures is a dict, storing the art features (hue, saturation, value, line quality) assigned to each bandwidth of brain wave
    '''

    bias = 0
    if freqState == 'high': # if the frequency band has a high amplitude, output higher value (0% = black)
        bias = 20

    # draw values for hue by randomly sampling from a normal distribution with mean and standard deviation that are randomly generated but biased depending on the amplitude level
    outputArray = np.random.normal(loc=random.randint(0,100)+bias,scale=100*0.05-bias*0.05,size=inputArray.shape)
    
    # reset values that are beyond boundaries
    outputArray[np.where(outputArray > 100)] = 100
    outputArray[np.where(outputArray < 0)] = 0

    return outputArray


def changeLineQual(inputArray,freqState,artFeature):

    ''' manipulate the saturation levels of the image

    inputArray is a numpy array storing the hue value of the imate
    freqState is a dict, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data
    artFeatures is a dict, storing the art features (hue, saturation, value, line quality) assigned to each bandwidth of brain wave
    '''

    # set the bias to the generated numbers depending on the state of the given frequency
    bias = 0
    if freqState == 'High':
        bias = int(len(inputArray)*0.1)    # take a percentage of the image size as the bias

    # divide the image up into a random number of sections (but higher if the given signal is high and lower if the given signal is lower)
    n_sections = random.randint(1,4)+bias
    section_size = len(inputArray)//n_sections  # just draw lines in one dimension (rows) for now

    # for each section, get a random range, i.e. the width of the line that will go in that section
    # (higher range if the given signal is high and lower if the given signal is low), and set it as 1 to be visible
    for i in range(n_sections):
        linewidth = random.randint(1,section_size)
        linestart = i*section_size+(section_size//2-linewidth//2)
        inputArray[linestart:linestart+linewidth,:]=1

    return inputArray


def getArtFunc(artFeature):
    
    ''' return the name of an appropriate function depending on the art feature being manipulated

    artFeature is an index indicating the art feature (0 = hue, 1 = saturation, 2 = value, 3 = line quality
    '''
    if artFeature == 0:
        return 'changeHue',artFeature
    elif artFeature == 1:
        return 'changeSaturation',artFeature
    elif artFeature == 2:
        return 'changeValue',artFeature
    else:
        return 'changeLineQual',artFeature

def hsvArt(size,freqNoise,freqState,artFeatures):

    ''' make an image in the HSV system based on the given data
    
    size is a tuple - the size of the image
    freqNoise is a dict, storing 'True' if there is noise and 'False' if there is not noise, for each bandwidth of brain wave for the last second of data
    freqState is a dict, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data
    artFeatures is a dict, storing the art features (hue, saturation, value, line quality) assigned to each bandwidth of brain wave

    '''
   
    # the bandwidths in the order of presentation in the Main Window
    freqOrder = ['Beta','Alpha','Theta','Delta']
    
    # unpack size into x and y dimensions
    xdim,ydim = size

    # initialize three x by y arrays, for the HSV values of each pixel in the image
    imageArray = np.zeros((xdim,ydim,1,4),dtype=np.uint8)

    # make the art
    for freq in freqNoise.keys():   # iterate through the bandwidths
        if not freqNoise[freq]: # if there is no noise in the wave signal

            # TODO: Decide on some sort of a default when there is noise in the signal -- e.g. if delta is noisy, then everything else is ignored but we don't wanna do that

            func,arrayIdx = getArtFunc(artFeatures[freqOrder.index(freq)])    # get appropriate function for the art feature to be manipulated
            inputArray = 'imageArray[:,:,:,%d]' % arrayIdx
            evalExpression = '%s(%s,\'%s\',%d)' % (func,inputArray,freqState[freq],artFeatures[freqOrder.index(freq)]) # create expression to be evaluated
            imageArray[:,:,:,arrayIdx] = eval(evalExpression)   # apply the function to the image array

    # based on the line quality layer, make the final array
    for i in range(3):
        imageArray[:,:,:,i][np.where(imageArray[:,:,:,3]==0)] = 0
    # generate image as PIL Image with the HSV mode, but return image in the RGB mode (ImageQt only accepts PIL Image in the RGB mode)
    im = Image.fromarray(imageArray[:,:,:,0:3], mode='HSV')
    
    return im.convert(mode='RGB')
    

def hsv_main():

    size=[400,400]

    freqNoise = {
        'Delta': False,
        'Theta': False,
        'Alpha': False,
        'Beta': False
    }

    freqState = {
        'Delta': 'Low',
        'Theta': 'Low',
        'Alpha': 'Low',
        'Beta': 'Low'
    }
    
    artFeatures = [0,1,2,3]
    
    while True:

        image = hsvArt(size,freqNoise,freqState,artFeatures)
        image.show()
        print('freqNoise:\n',freqNoise)
        print('freqState:\n',freqState)
        input()

        # waves = ['Delta','Theta','Alpha','Beta']
        waves = ['Theta','Alpha','Beta']

        change_noise = random.choice([0,1])
        if change_noise:
            wave = random.choice(waves)
            if freqNoise[wave] == True:
                freqNoise[wave] = False
            else:
                freqNoise[wave] = True

        if not change_noise:
            change_state = True
        else:
            change_state = random.choice(waves)
        if change_state:
            wave = random.choice(waves)
            if freqState[wave] == 'Low':
                freqState[wave] = 'High'
            else:
                freqState[wave] = 'High'

if __name__ == '__main__':
    hsv_main()
