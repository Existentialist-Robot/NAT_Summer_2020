import numpy as np
from PIL import Image
import random
import pdb
import cv2
import matplotlib.pyplot as plt


def circleArt(imageArray,freqNoise=[],freqState=[],artFeatures=[0,1,2,3],pulse=False):

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

    if pulse:
        imageArray = imageArray[0]+5

    else:
        lineQualBandwidth = freqOrder[artFeatures.index(3)]

        # pdb.set_trace()
        if not freqNoise[lineQualBandwidth]:
            if freqState[lineQualBandwidth] == 'High':
                thicknessRange = (xdim*0.3,xdim*0.4)
                nCircles = random.randint(20,30)
                alpha = random.uniform(0.6,1)
                
            else:
                thicknessRange = (xdim*0.1,xdim*0.2)
                nCircles = random.randint(5,10)
                alpha = random.uniform(0.2, 0.4)
        else:
            thicknessRange = (5,xdim*0.05)
            nCircles = random.randint(0,3)
            alpha = random.uniform(0,0.1)

        rBandwidth = freqOrder[artFeatures.index(0)]
        if not freqNoise[rBandwidth]:
            if freqState[rBandwidth] == 'High':
                rRange = (200,255)
            else:
                rRange = (100, 180)
        else:
            rRange = (30,60)

        gBandwidth = freqOrder[artFeatures.index(1)]
        if not freqNoise[gBandwidth]:
            if freqState[gBandwidth] == 'High':
                gRange = (200,255)
            else:
                gRange = (100, 180)
        else:
            gRange = (30,60)

        bBandwidth = freqOrder[artFeatures.index(2)]
        if not freqNoise[bBandwidth]:
            if freqState[bBandwidth] == 'High':
                bRange = (200,255)
            else:
                bRange = (100, 180)
        else:
            bRange = (30,60)

        print(thicknessRange)
        print(nCircles)
        print(rRange)
        print(gRange)
        print(bRange)

        for i in range(nCircles):
            random.choice([
                cv2.circle(imageArray,(random.randint(xdim*0.2,xdim*0.8),random.randint(ydim*0.2,ydim*0.8)),random.randint(xdim*0.1,xdim*0.5),(random.randint(rRange[0],rRange[1]),random.randint(gRange[0],gRange[1]),random.randint(bRange[0],bRange[1]))),
                cv2.circle(imageArray,(random.randint(xdim*0.2,xdim*0.8),random.randint(ydim*0.2,ydim*0.8)),random.randint(xdim*0.1,xdim*0.5),(random.randint(rRange[0],rRange[1]),random.randint(gRange[0],gRange[1]),random.randint(bRange[0],bRange[1])),thickness = random.randint(thicknessRange[0],thicknessRange[1]))
                ])

            imageArray = cv2.addWeighted(imageArray, alpha, imageArray, 1 - alpha, 0)

        # generate image as PIL Image with the HSV mode, but return image in the RGB mode (ImageQt only accepts PIL Image in the RGB mode)
        # im = Image.fromarray(imageArray, mode='RGB')
        return imageArray
        # return im.convert(mode='RGB')
    

def main():

    size=[800,800]

    # initialize an array for the image -- x by y arrays of 3-item arrays for the HSV values of each pixel in the image
    imageArray = np.zeros((size[0],size[1],3),dtype=np.uint8)
    # initialize an x by y array for making patterns on the image by using blend modes
    # imageBlend = np.zeros((size[0], size[1]), dtype=np.uint8)

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
        'Beta': 'High'
    }

    artFeatures = [0,1,2,3]
    
    image = circleArt(imageArray,freqNoise=freqNoise,freqState=freqState,artFeatures=artFeatures)
    
    # image.show()
    print('freqNoise:\n',freqNoise)
    print('freqState:\n',freqState)

    plt.imshow(image)
    plt.show()

    image = circleArt(imageArray,pulse=True)

    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    main()
