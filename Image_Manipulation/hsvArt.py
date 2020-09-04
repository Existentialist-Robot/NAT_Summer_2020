import numpy as np
from PIL import Image
import random
import pdb

# TODO: Incorporate recursion into the design

random.seed = 1


def changeHue(inputArray,freqState,pulse=False,inc=True):

    ''' manipulate the hue levels of the image

    inputArray is a numpy array storing the hue value of the imate
    freqState is a dict, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data

    if pulse is True, only increment (default) or decrement (if inc == False) by a small amount
    '''

    outputArray = np.zeros(inputArray.shape)

    if pulse:
        if inc:
            inputArray += 10
        else:
            inputArray -= 10
        return

    if freqState == 'high': # if the frequency band has a high amplitude, push the colours toward red
        hue = random.choice((0,50),(300,360))
        hue = random.randint(hue)
    elif freqState == 'low':    # if the frequency band has a low amplitude, push the colours toward blue
        hue = random.randint(70,300)
    else:   # if default, pick from the yellow range
        hue = random.randint(50,70)

    # assign hue value to the input array
    outputArray.fill(hue)

    return outputArray


def changeSaturation(inputArray,freqState,pulse=False,inc=True):

    ''' manipulate the saturation levels of the image

    inputArray is a numpy array storing the hue value of the imate
    freqState is a dict, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data

    if pulse is True, only increment (default) or decrement (if inc == False) by a small amount
    '''

    outputArray = np.zeros(inputArray.shape)

    if pulse:
        if inc:
            inputArray += 10
        else:
            inputArray -= 10
        return

    if freqState == 'high': # if the frequency band has a high amplitude, bias toward higher saturation
        bias = 50
    else:   # if the frequency band has a low amplitude or is set to default, bias toward lower saturation
        bias = -50

    # draw a random value for saturation then add the bias
    sat = random.randint(0,100) + bias

    # cap the saturation to 0 for the lower bound and 100 for the upper bound
    if sat > 100:
        sat = 100
    elif sat < 0:
        sat = 0

    # assign the saturation value to the input array
    outputArray.fill(sat)

    return outputArray


def changeValue(inputArray,freqState,pulse=False,inc=True):

    ''' manipulate the value levels of the image

    inputArray is a numpy array storing the hue value of the imate
    freqState is a dict, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data

    if pulse is True, only increment (default) or decrement (if inc == False) by a small amount
    '''

    outputArray = np.zeros(inputArray.shape)

    if pulse:
        if inc:
            inputArray += 10
        else:
            inputArray -= 10
        return

    if freqState == 'high': # if the frequency band has a high amplitude, output higher value (0% = black)
        bias = 50
    else:   # if the frequency band has a high amplitude, output lower value
        bias = -50

    # draw a random value for the value then add the bias
    value = random.randint(0, 100) + bias

    # cap the saturation to 0 for the lower bound and 100 for the upper bound
    if value > 100:
        value = 100
    elif value < 0:
        value = 0

    # assign the saturation value to the input array
    outputArray.fill(value)

    return outputArray

def changeLineQual(inputArray, freqState):

    ''' manipulate the saturation levels of the image

    inputArray is a numpy array storing the hue value of the imate
    freqState is a string, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data
    '''

    outputArray = np.zeros(inputArray.shape)

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
        outputArray[linestart:linestart+linewidth,:]=1

    return outputArray


def changeBlendLayer(inputArray,freqState):

    ''' manipulate the blend layer of the image
    
    inputArray is a numpy array storing the hue value of the imate
    freqState is a string, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data
    '''

    pass


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


def hsvArt(imageArray,blendArray,freqNoise,freqState,artFeatures):

    ''' make an image in the HSV system based on the given data
    
    size is a tuple - the size of the image
    freqNoise is a dict, storing 'True' if there is noise and 'False' if there is not noise, for each bandwidth of brain wave for the last second of data
    freqState is a dict, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data
    artFeatures is a dict, storing the art features (hue, saturation, value, line quality) assigned to each bandwidth of brain wave

    '''
   
    # the bandwidths in the order of presentation in the Main Window
    freqOrder = ['Beta','Alpha','Theta','Delta']
    
    # unpack size into x and y dimensions
    xdim,ydim = imageArray.shape[:2]

    # make the art
    for freq in freqNoise.keys():   # iterate through the bandwidths

        func,arrayIdx = getArtFunc(artFeatures[freqOrder.index(freq)])    # get appropriate function for the art feature to be manipulated

        # if there is noise in the bandwidth, set the bandwidth state to 'default'. if not, just set it to what it is (high or low)
        if freqNoise[freq]:
            state = 'default'
        else:
            state = freqState[freq]

        if arrayIdx == 3:   # if changing line quality
            inputArray = 'blendArray'

            evalExpression = '%s(%s,\'%s\')' % (func,inputArray,state) # create expression to be evaluated
            blendArray = eval(evalExpression)   # apply the function to the blend array

        else:
            inputArray = 'imageArray[:,:,:,%d]' % arrayIdx

            evalExpression = '%s(%s,\'%s\')' % (func,inputArray,state) # create expression to be evaluated
            imageArray[:,:,:,arrayIdx] = eval(evalExpression)   # apply the function to the image array

    # based on the line quality layer, make the final array
    for i in range(3):
        imageArray[:,:,:,i][np.where(blendArray==0)] = 0
    # generate image as PIL Image with the HSV mode, but return image in the RGB mode (ImageQt only accepts PIL Image in the RGB mode)
    im = Image.fromarray(imageArray[:,:,:,0:3], mode='HSV')
<<<<<<< HEAD
    
=======
    #return im
>>>>>>> upstream/master
    return im.convert(mode='RGB')
    

def hsv_main():

    size=[400,400]

    # initialize an array for the image -- x by y arrays of 3-item arrays for the HSV values of each pixel in the image
    imageArray = np.zeros((size[0],size[1],1,3),dtype=np.uint8)
    # initialize an x by y array for making patterns on the image by using blend modes
    imageBlend = np.zeros((size[0], size[1]), dtype=np.uint8)

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

        image = hsvArt(imageArray,imageBlend,freqNoise,freqState,artFeatures)
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
